"""Two commands: ingest PDFs once, then ask questions against the saved index."""

import argparse
import os
from pathlib import Path

from .answer import answer_question
from .index import Index, build_index
from .pdfs import load_chunks


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ask cited questions about local PDFs.")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Build or replace the local PDF index")
    ingest.add_argument("pdf_dir", nargs="?", type=Path, default=Path("data/pdfs"))
    ingest.add_argument("--index", type=Path, default=Path(".rag/index.json"))
    ingest.add_argument("--chunk-size", type=int, default=1200)
    ingest.add_argument("--overlap", type=int, default=200)
    ingest.add_argument("--embedding-provider", choices=("local", "openai"), default="local")
    ingest.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL"))
    ask = commands.add_parser("ask", help="Retrieve evidence and answer a question")
    ask.add_argument("question")
    ask.add_argument("--index", type=Path, default=Path(".rag/index.json"))
    ask.add_argument("--top-k", type=int, default=4)
    ask.add_argument("--min-score", type=float, default=0.0)
    ask.add_argument("--answer-provider", choices=("ollama", "openai"), default="ollama")
    ask.add_argument("--model", default=os.getenv("RAG_ANSWER_MODEL"))
    args = parser.parse_args(argv)
    try:
        # Validate local input before constructing the API client.
        if args.command == "ingest":
            chunks = load_chunks(args.pdf_dir, args.chunk_size, args.overlap)
            needs_openai = args.embedding_provider == "openai"
        else:
            index = Index.load(args.index)
            needs_openai = index.provider == "openai" or args.answer_provider == "openai"
            if args.top_k < 1 or not -1 <= args.min_score <= 1:
                raise ValueError("--top-k must be positive; --min-score must be between -1 and 1.")
            if not args.question.strip() or len(args.question.strip()) > 4000:
                raise ValueError("Question must contain 1 to 4000 characters.")
        client = None
        if needs_openai:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("Set OPENAI_API_KEY in your environment before running this command.")
            from openai import OpenAI
            client = OpenAI(timeout=60.0, max_retries=2)
        if args.command == "ingest":
            defaults = {"local": "sentence-transformers/all-MiniLM-L6-v2", "openai": "text-embedding-3-small"}
            model = args.embedding_model or defaults[args.embedding_provider]
            index = build_index(client, chunks, model, args.embedding_provider)
            index.save(args.index)
            print(f"Indexed {len(chunks)} chunks from {len({c.source for c in chunks})} PDFs at {args.index}.")
        else:
            defaults = {"ollama": "llama3.2:3b", "openai": "gpt-4.1-mini"}
            model = args.model or defaults[args.answer_provider]
            print(answer_question(
                client, index, args.question, model, args.top_k, args.min_score,
                answer_provider=args.answer_provider,
                ollama_host=os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434",
            ))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    except ImportError:
        parser.exit(1, 'Error: Install dependencies with python -m pip install -e "."\n')
    except Exception as exc:
        # API failures should be concise CLI errors, not unhandled tracebacks.
        from openai import OpenAIError
        if isinstance(exc, OpenAIError):
            parser.exit(1, f"OpenAI API error: {exc}\n")
        raise
