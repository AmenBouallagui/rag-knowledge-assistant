"""HTTP interface for the local PDF question-answering pipeline."""

import os
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from .answer import (
    AnswerResult,
    CITATION_PATTERN,
    DEFAULT_OLLAMA_MODEL,
    OllamaUnavailableError,
    generate_answer,
)
from .index import Index


app = FastAPI(title="PDF Knowledge Assistant")


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1)
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> AnswerResult:
    index_path = Path(os.getenv("RAG_INDEX_PATH") or ".rag/index.json")
    try:
        index = Index.load(index_path)
    except (ValueError, OSError, OverflowError) as exc:
        raise HTTPException(503, "RAG index is unavailable. Check RAG_INDEX_PATH and run rag ingest.") from exc

    client = None
    if index.provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(503, "OPENAI_API_KEY is required for this index's embeddings.")
        if len(request.question.encode("utf-8")) > 8192:
            raise HTTPException(422, "Question must be at most 8192 UTF-8 bytes for OpenAI embeddings.")
        client = OpenAI(timeout=60.0, max_retries=2)

    try:
        result = generate_answer(
            client, index, request.question, DEFAULT_OLLAMA_MODEL,
            request.top_k, request.min_score, answer_provider="ollama",
            ollama_host=os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434",
        )
    except OllamaUnavailableError as exc:
        raise HTTPException(503, "Ollama is unavailable. Check that it is running and the model is pulled.") from exc
    except OpenAIError as exc:
        raise HTTPException(503, "OpenAI embeddings are unavailable.") from exc
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(502, "The RAG pipeline could not produce a valid answer.") from exc

    # Expose citation labels in the structured sources, without CLI inline markers.
    answer = re.sub(r"[ \t]*" + CITATION_PATTERN, "", result.answer).strip()
    return AnswerResult(answer, result.sources, result.insufficient_evidence)
