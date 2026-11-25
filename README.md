# PDF Knowledge Assistant

A minimal Python RAG prototype: local PDFs → page-aware chunks → embeddings →
cosine retrieval → an answer with numbered PDF/page citations.

## Folder structure

```text
rag_assistant/
  pdfs.py       Extract PDF text and create overlapping chunks within each page
  index.py      Embed chunks, persist JSON, and rank by cosine similarity
  answer.py     Retrieve evidence, generate an answer, and resolve citations
  cli.py        The ingest and ask commands
  api.py        FastAPI health and question endpoints
  __main__.py   Support python -m rag_assistant
data/pdfs/      Put local PDFs here (ignored by Git)
.rag/          Generated index.json (ignored by Git)
tests/         Offline pipeline, retrieval, and citation tests
pyproject.toml Dependencies and rag CLI entry point
.env.example   Environment-variable examples (not loaded automatically)
```

No framework, database, web server, or background service. `index.json` holds
chunk text, relative PDF paths, 1-based PDF page numbers, normalized vectors,
and the embedding provider and model used to build them. Retrieval always uses that saved
embedding model, even if you later change the environment variable.

## Setup

Requires Python 3.10+. Set `OPENAI_API_KEY` if you use OpenAI embeddings or select OpenAI answers.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Local PDF ingestion was tested on an Intel MacBook Pro (macOS x86_64) with
Python 3.12.10, torch 2.2.2, NumPy 1.26.4, transformers 4.57.6, and
sentence-transformers 3.4.1. To use Python 3.12 explicitly, create the virtual
environment with `python3.12 -m venv .venv`. The torch 2.2.x constraint applies
only to Intel macOS.

On Windows, activate with `.venv\Scripts\activate` and set the environment
variable using your shell's syntax. `.env.example` documents the available
variables; the application reads exported environment variables directly.

## Use

Start Ollama and pull the default model before asking a question.

```bash
ollama pull llama3.2:3b
```

The first inference may be noticeably slower while Ollama loads the model.

Ollama must be running at `http://127.0.0.1:11434` when you ask. Set
`OLLAMA_HOST` to override that URL. If you choose another Ollama model with
`--model`, pull that model first as well.

Place PDFs in `data/pdfs/` (nested directories are supported), then run:

```bash
rag ingest data/pdfs
rag ask "How many days of annual leave are allowed?"
```

To use OpenAI for generation instead, set your API key and select the provider.

```bash
export OPENAI_API_KEY="your-api-key"
rag ask "How many days of annual leave are allowed?" --answer-provider openai
```

Illustrative output for a PDF containing that policy:

```text
Employees receive 25 days of annual leave. [1]

Sources:
[1] policies/handbook.pdf, page 2
```

You can also use `python -m rag_assistant` in place of `rag`.

```bash
rag ingest /path/to/pdfs --chunk-size 1200 --overlap 200 --index .rag/index.json
rag ask "What is the reimbursement policy?" --top-k 4 --min-score 0.25
```

Ingestion defaults to local Sentence Transformers embeddings. The first run
downloads `sentence-transformers/all-MiniLM-L6-v2` unless already cached.
Use `--embedding-model /absolute/path/to/model` for a model installed offline.
Keep that path available when asking questions. To select OpenAI embeddings:

```bash
rag ingest data/pdfs --embedding-provider openai
```

Asking uses the provider and model saved during ingestion. Existing indexes
without a provider continue to use OpenAI. Re-ingest to switch providers.
Set `RAG_EMBEDDING_MODEL` only to a model compatible with the selected provider.

Defaults: 1,200 characters per chunk, 200 characters of overlap, four retrieved
chunks, `text-embedding-3-small` for the OpenAI embedding option, and `llama3.2:3b` for Ollama
answers. OpenAI answers default to `gpt-4.1-mini`.
Override with `--embedding-model` during ingestion, `--model` during asking, or
`RAG_EMBEDDING_MODEL` / `RAG_ANSWER_MODEL`. Re-run ingestion whenever PDFs change;
it rebuilds the complete index and replaces it only after successful embedding.

## HTTP API

Build the index with `rag ingest` and start Ollama with `llama3.2:3b` available.
From the project directory, run:

```bash
uvicorn rag_assistant.api:app --reload
```

`GET /health` returns the API's status and works independently of Ollama.
`POST /ask` uses `.rag/index.json` by default. Set `RAG_INDEX_PATH` to use
another index. The API uses the saved embedding provider and Ollama for answers.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What experience does Amen have with Microsoft Fabric?"}'
```

The request accepts optional `top_k` and `min_score` fields, which default to
`4` and `0.0`. The response includes `answer`, `sources` with citation labels,
filenames and page numbers, and `insufficient_evidence`. The interactive
[FastAPI docs](http://127.0.0.1:8000/docs) show the request and response schemas.

## How it works

1. `pypdf` extracts each page separately, preserving its source path and page.
2. The chunker groups lines and bullets within each page. For oversized units,
   it prefers sentence boundaries. Overlap carries complete trailing units
   when they fit with the next unit.
3. The selected provider embeds chunks in batches of 32. Vectors and metadata
   are stored in one local JSON file.
4. The question is embedded using the same model. A linear cosine-similarity
   search ranks the chunks and selects up to `--top-k` above `--min-score`.
5. The selected answer provider receives only the retrieved excerpts and the question.
   It is instructed to cite supported claims or abstain. The application rejects
   missing/out-of-range citation labels and appends filenames/pages itself.

With local embeddings and the default Ollama server, PDF text and questions
stay on your machine and asking requires no OpenAI key. OpenAI embeddings send
chunk text and the question to OpenAI. OpenAI answers send the question and
retrieved excerpts to OpenAI with `store=False`. The local index contains
document text. PDFs, keys, and generated indexes are ignored by Git.

## Limits

- Scanned/image-only PDFs need OCR first. Empty pages produce warnings;
  unreadable/password-protected PDFs stop ingestion with the filename.
- PDF page numbers are physical pages, not necessarily printed page labels.
- Extraction may lose table layout. Oversized sentences may split across chunks.
  Local models truncate inputs beyond their token limit. Each OpenAI embedding input is limited conservatively to 8,192 UTF-8 bytes; reduce
  `--chunk-size` if a multilingual document exceeds that bound.
- Linear search and JSON suit small document collections; the entire index is
  loaded into memory. There is no incremental indexing or reranking.
- Similarity is not a calibrated confidence score. Tune `--min-score` against
  your documents; its default of 0.0 is permissive.
- Citation validation checks that labels refer to retrieved chunks. It cannot
  prove that every claim is supported. Inspect the cited pages for critical use.
  The prompt treats document text as untrusted, but is not an injection guarantee.

## Tests

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
```

Tests generate a real two-page PDF and exercise extraction, chunking, embedding
batching/order, index persistence, cosine ranking, CLI commands, citation
validation, abstention, and Ollama error handling. The tests mock generation and
embeddings, so no API key, Ollama server, or model download is needed. GitHub Actions runs the same tests.

API references: [Embeddings](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create)
and [Responses](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
