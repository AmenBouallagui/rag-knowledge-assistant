"""Offline HTTP tests using the real answer pipeline and mocked model calls."""

from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from fastapi.testclient import TestClient

from rag_assistant.answer import NO_ANSWER, answer_question
from rag_assistant.api import app
from rag_assistant.index import Index
from rag_assistant.pdfs import Chunk


class ApiTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        directory = stack.enter_context(tempfile.TemporaryDirectory())
        self.path = Path(directory) / "index.json"
        stack.enter_context(patch.dict(os.environ, {"RAG_INDEX_PATH": str(self.path)}, clear=True))
        self.client = stack.enter_context(TestClient(app))
        self.index = Index(
            "test-model",
            [Chunk("other.pdf", 1, "Overview."), Chunk("Resume_EN.pdf", 2, "Migrated data to Fabric.")],
            [[1.0, 0.0], [1.0, 0.0]],
            provider="local",
        )
        self.embed = stack.enter_context(patch("rag_assistant.answer.embed", return_value=[[1.0, 0.0]]))
        self.ollama = stack.enter_context(patch("rag_assistant.answer.urlopen"))

    def ask(self, **options):
        return self.client.post("/ask", json={"question": "What work was done?", **options})

    def mock_answer(self, text):
        self.ollama.return_value = io.BytesIO(json.dumps({"message": {"content": text}}).encode())

    def test_health_without_index_or_ollama(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.ollama.assert_not_called()
        self.embed.assert_not_called()

    def test_ask_returns_answer_and_cited_sources(self):
        self.index.save(self.path)
        self.mock_answer("Amen migrated data to Fabric. [2] [2]")
        response = self.ask()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "answer": "Amen migrated data to Fabric.",
            "sources": [{"label": 2, "source": "Resume_EN.pdf", "page": 2}],
            "insufficient_evidence": False,
        })
        self.assertEqual(self.embed.call_args.args, (None, ["What work was done?"], "test-model", "local"))
        payload = json.loads(self.ollama.call_args.args[0].data)
        self.assertEqual(payload["model"], "llama3.2:3b")
        self.assertEqual(payload["options"], {"temperature": 0, "seed": 42})
        self.assertEqual(self.ollama.call_args.kwargs["timeout"], 180)

    def test_insufficient_evidence_from_model(self):
        self.index.save(self.path)
        self.mock_answer("INSUFFICIENT_EVIDENCE")
        response = self.ask()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "answer": NO_ANSWER, "sources": [], "insufficient_evidence": True,
        })

    def test_min_score_can_skip_generation(self):
        self.index.vectors = [[0.0, 1.0], [0.0, 1.0]]
        self.index.save(self.path)
        response = self.ask(min_score=0.1)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["insufficient_evidence"])
        self.ollama.assert_not_called()

    def test_invalid_request(self):
        for fields in (
            {"question": " "}, {"question": "x" * 4001}, {"top_k": 0},
            {"top_k": 1.5}, {"min_score": -1.1}, {"min_score": 1.1},
        ):
            with self.subTest(fields=fields):
                self.assertEqual(self.ask(**fields).status_code, 422)
        self.assertEqual(self.client.post("/ask", json={}).status_code, 422)
        self.embed.assert_not_called()

    def test_missing_or_invalid_index(self):
        self.assertEqual(self.ask().status_code, 503)
        self.path.write_text("invalid json")
        self.assertEqual(self.ask().status_code, 503)
        self.ollama.assert_not_called()

    def test_ollama_unavailable(self):
        self.index.save(self.path)
        self.ollama.side_effect = URLError(ConnectionRefusedError("private connection detail"))
        response = self.ask()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private connection detail", response.text)

    def test_top_k_and_invalid_citation(self):
        self.index.save(self.path)
        self.mock_answer("Amen migrated data. [2]")
        self.assertEqual(self.ask(top_k=1).status_code, 502)

    def test_cli_format_preserves_inline_citations_and_source_footer(self):
        self.mock_answer("Amen migrated data to Fabric. [2] [2]")
        result = answer_question(
            None, self.index, "What work was done?", "llama3.2:3b", answer_provider="ollama"
        )
        self.assertEqual(result, "Amen migrated data to Fabric. [2]\n\nSources:\n[2] Resume_EN.pdf, page 2")
