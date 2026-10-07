"""Small JSON vector index; no database or vector-search service required."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import tempfile

from .pdfs import Chunk


def normalize(vector: list[float]) -> list[float]:
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("Embedding must contain finite values.")
    length = math.sqrt(sum(value * value for value in vector))
    if length == 0:
        raise ValueError("Embedding cannot be a zero vector.")
    return [value / length for value in vector]


def embed(client, texts: list[str], model: str) -> list[list[float]]:
    # Byte count conservatively bounds token count without a tokenizer dependency.
    if any(not text.strip() or len(text.encode("utf-8")) > 8192 for text in texts):
        raise ValueError("Embedding text must be nonempty and at most 8192 UTF-8 bytes; "
                         "reduce --chunk-size or shorten the question.")
    vectors = []
    # 32 * 8192 <= 300,000 tokens across the batch.
    for start in range(0, len(texts), 32):
        batch = texts[start:start + 32]
        result = client.embeddings.create(model=model, input=batch, encoding_format="float")
        items = sorted(result.data, key=lambda item: item.index)
        if [item.index for item in items] != list(range(len(batch))):
            raise ValueError("Embedding response did not match the input batch.")
        vectors.extend(normalize(item.embedding) for item in items)
    return vectors


@dataclass
class Index:
    model: str
    chunks: list[Chunk]
    vectors: list[list[float]]

    def validate(self):
        if not self.model or not self.chunks or len(self.chunks) != len(self.vectors):
            raise ValueError("Index is empty or has mismatched chunks and vectors; run ingest again.")
        dimensions = len(self.vectors[0])
        if any(len(vector) != dimensions for vector in self.vectors):
            raise ValueError("Index embeddings have inconsistent dimensions.")
        for vector in self.vectors:
            normalize(vector)

    def save(self, path: Path):
        self.validate()
        payload = {"version": 1, "model": self.model,
                   "chunks": [asdict(chunk) for chunk in self.chunks], "vectors": self.vectors}
        path.parent.mkdir(parents=True, exist_ok=True)
        # Replace only after a successful write, preserving an existing index on failure.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as handle:
            temporary = Path(handle.name)
            try:
                json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["version"] != 1:
                raise ValueError("Unsupported index version.")
            index = cls(payload["model"], [Chunk(**item) for item in payload["chunks"]],
                        [normalize(vector) for vector in payload["vectors"]])
            index.validate()
            return index
        except FileNotFoundError as exc:
            raise ValueError(f"Index not found at {path}; run ingest first.") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid index at {path}; run ingest again.") from exc

    def retrieve(self, vector: list[float], top_k: int = 4, min_score: float = 0.0):
        self.validate()
        if top_k < 1 or not -1 <= min_score <= 1:
            raise ValueError("top_k must be positive and min_score must be between -1 and 1.")
        query = normalize(vector)
        if len(query) != len(self.vectors[0]):
            raise ValueError("Query embedding dimensions do not match the index; re-ingest.")
        matches = [(chunk, sum(a * b for a, b in zip(query, normalize(values))))
                   for chunk, values in zip(self.chunks, self.vectors)]
        return sorted((match for match in matches if match[1] >= min_score),
                      key=lambda match: match[1], reverse=True)[:top_k]


def build_index(client, chunks: list[Chunk], model: str) -> Index:
    if not chunks:
        raise ValueError("Cannot index an empty collection.")
    index = Index(model, chunks, embed(client, [chunk.text for chunk in chunks], model))
    index.validate()
    return index
