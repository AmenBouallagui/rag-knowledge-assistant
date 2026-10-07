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
  __main__.py   Support python -m rag_assistant
data/pdfs/      Put local PDFs here (ignored by Git)
.rag/          Generated index.json (ignored by Git)
tests/         Offline pipeline, retrieval, and citation tests
pyproject.toml Dependencies and rag CLI entry point
.env.example   Environment-variable examples (not loaded automatically)
```

No framework, database, web server, or background service. `index.json` holds
chunk text, relative PDF paths, 1-based PDF page numbers, normalized vectors,
and the embedding model used to build them. Retrieval always uses that saved
embedding model, even if you later change the environment variable.

## Setup

Requires Python 3.10+ and an OpenAI API key with access to the configured models.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export OPENAI_API_KEY="your-api-key"
```

On Windows, activate with `.venv\Scripts\activate` and set the environment
variable using your shell's syntax. `.env.example` documents the available
variables; the application reads exported environment variables directly.

## Use

Place PDFs in `data/pdfs/` (nested directories are supported), then run:

```bash
rag ingest data/pdfs
rag ask "How many days of annual leave are allowed?"
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

Defaults: 1,200 characters per chunk, 200 characters of overlap, four retrieved
chunks, `text-embedding-3-small` for embeddings, and `gpt-4.1-mini` for answers.
Override with `--embedding-model` during ingestion, `--model` during asking, or
`RAG_EMBEDDING_MODEL` / `RAG_ANSWER_MODEL`. Re-run ingestion whenever PDFs change;
it rebuilds the complete index and replaces it only after successful embedding.

## How it works

1. `pypdf` extracts each page separately, preserving its source path and page.
2. Overlapping character windows split the text; chunks never cross pages.
3. The OpenAI embeddings API embeds chunks in batches of 32. Vectors and metadata
   are stored in one local JSON file.
4. The question is embedded using the same model. A linear cosine-similarity
   search ranks the chunks and selects up to `--top-k` above `--min-score`.
5. The Responses API receives only the retrieved excerpts and the question.
   It is instructed to cite supported claims or abstain. The application rejects
   missing/out-of-range citation labels and appends filenames/pages itself.

Extracted PDF text is sent to OpenAI for embedding; questions and retrieved
excerpts are sent for answer generation. These requests require network access
and incur API usage. The index is local but contains document text. PDFs, keys,
and generated indexes are ignored by Git. Response generation uses `store=False`.

## Limits

- Scanned/image-only PDFs need OCR first. Empty pages produce warnings;
  unreadable/password-protected PDFs stop ingestion with the filename.
- PDF page numbers are physical pages, not necessarily printed page labels.
- Extraction may lose table layout. Character chunking can split sentences.
  Each embedding input is limited conservatively to 8,192 UTF-8 bytes; reduce
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
python -m unittest discover -s tests -v
```

Tests generate a real two-page PDF and exercise extraction, chunking, embedding
batching/order, index persistence, cosine ranking, CLI commands, citation
validation, and abstention. API responses are deterministic doubles: no key,
network, or API spend is needed. GitHub Actions runs the same tests.

API references: [Embeddings](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create)
and [Responses](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
