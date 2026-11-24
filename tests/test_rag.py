"""Offline checks with real PDF extraction and deterministic API doubles."""

import io
import json
from urllib.error import HTTPError, URLError
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from rag_assistant.answer import NO_ANSWER, OLLAMA_INSTRUCTIONS, answer_question, ollama_answer
from rag_assistant.cli import main
from rag_assistant.index import Index, build_index, embed
from rag_assistant.pdfs import Chunk, load_chunks, split_text


def write_pdf(path):
    """Create two extractable pages without requiring a PDF-generation dependency."""
    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    font_ref = writer._add_object(font)
    for text in ("The office is in Vienna.", "Annual leave is 25 days."):
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


class FakeClient:
    def __init__(self, output="Annual leave is 25 days. [1]", status="completed"):
        self.calls = []
        self.embeddings = SimpleNamespace(create=self.embedding_response)
        self.responses = SimpleNamespace(create=self.answer_response)
        self.output = output
        self.status = status

    def embedding_response(self, **kwargs):
        self.calls.append(("embeddings", kwargs))
        data = [SimpleNamespace(index=i, embedding=[1.0, 0.0] if "leave" in text else [0.0, 1.0])
                for i, text in enumerate(kwargs["input"])]
        return SimpleNamespace(data=list(reversed(data)))

    def answer_response(self, **kwargs):
        self.calls.append(("responses", kwargs))
        return SimpleNamespace(output_text=self.output, status=self.status)


class RagTests(unittest.TestCase):
    def test_chunk_coverage_overlap_and_limits(self):
        text = "abcdefghij" * 31
        chunks = split_text(text, chunk_size=100, overlap=20)
        reconstructed = chunks[0] + "".join(chunk[20:] for chunk in chunks[1:])
        self.assertEqual(reconstructed, text)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertEqual(split_text("   \n"), [])
        for size, overlap in [(0, 0), (100, 100), (100, -1), (4001, 1)]:
            with self.assertRaises(ValueError):
                split_text("", size, overlap)

    def test_pdf_to_persisted_index_to_cited_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "policies").mkdir()
            write_pdf(root / "policies" / "handbook.PDF")
            chunks = load_chunks(root)
            self.assertEqual([c.page for c in chunks], [1, 2])
            self.assertEqual(chunks[1].source, "policies/handbook.PDF")
            client = FakeClient()
            index = build_index(client, chunks, "test-embedding")
            path = root / "index.json"
            index.save(path)
            loaded = Index.load(path)
            answer = answer_question(client, loaded, "How much annual leave?", "test-answer", top_k=1)
            self.assertIn("Annual leave is 25 days. [1]", answer)
            self.assertIn("[1] policies/handbook.PDF, page 2", answer)
            self.assertEqual(client.calls[-2][1]["model"], "test-embedding")
            prompt = json.loads(client.calls[-1][1]["input"])
            self.assertEqual(len(prompt["excerpts"]), 1)
            self.assertIn("25 days", prompt["excerpts"][0]["text"])
            self.assertFalse(client.calls[-1][1]["store"])

    def test_cosine_ranking_threshold_and_dimensions(self):
        index = Index("test", [Chunk("a.pdf", 1, "A"), Chunk("b.pdf", 4, "B")],
                      [[1.0, 0.0], [0.0, 20.0]])
        matches = index.retrieve([0.0, 2.0], top_k=10, min_score=0.5)
        self.assertEqual([c.source for c, _ in matches], ["b.pdf"])
        for vector in ([1.0], [0.0, 0.0], [float("nan"), 1.0]):
            with self.assertRaises(ValueError):
                index.retrieve(vector)

    def test_citation_validation_and_abstention(self):
        index = Index("test", [Chunk("a.pdf", 2, "leave policy")], [[1.0, 0.0]])
        for output in ("No citations", "Wrong citation [2]", "Wrong citation [0]", ""):
            with self.assertRaises(ValueError):
                answer_question(FakeClient(output), index, "leave?", "test")
        self.assertEqual(answer_question(FakeClient("INSUFFICIENT_EVIDENCE"), index,
                                         "leave?", "test"), NO_ANSWER)
        client = FakeClient()
        self.assertEqual(answer_question(client, index, "office?", "test", min_score=0.5), NO_ANSWER)
        self.assertEqual([name for name, _ in client.calls], ["embeddings"])
        with self.assertRaises(ValueError):
            answer_question(FakeClient(status="incomplete"), index, "leave?", "test")

    def test_batch_order_and_size(self):
        client = FakeClient()
        vectors = embed(client, ["leave"] * 33, "test")
        self.assertEqual(len(vectors), 33)
        self.assertEqual([len(args["input"]) for _, args in client.calls], [32, 1])
        self.assertTrue(all(vector == [1.0, 0.0] for vector in vectors))
        for text in (" ", "\U0001f600" * 2049):
            with self.assertRaises(ValueError):
                embed(client, [text], "test")

    def test_empty_pdf_missing_input_and_bad_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                load_chunks(root)
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            writer.write(root / "blank.pdf")
            with self.assertWarns(UserWarning), self.assertRaises(ValueError):
                load_chunks(root)
            with self.assertRaises(ValueError):
                Index.load(root / "missing.json")
            (root / "bad.json").write_text("{}")
            with self.assertRaises(ValueError):
                Index.load(root / "bad.json")

    def test_default_local_ingest_and_query(self):
        from unittest.mock import Mock

        encoder = Mock()
        encoder.encode.side_effect = [
            Mock(tolist=lambda: [[0, 2], [3, 0]]),
            Mock(tolist=lambda: [[4, 0]]),
        ]
        constructor = Mock(return_value=encoder)
        module = SimpleNamespace(SentenceTransformer=constructor)
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", clear=True), patch.dict(
            "sys.modules", {"sentence_transformers": module, "openai": None}
        ):
            root = Path(tmp)
            write_pdf(root / "handbook.pdf")
            path = root / "index.json"
            main(["ingest", tmp, "--index", str(path)])
            index = Index.load(path)
            self.assertEqual(index.provider, "local")
            response = io.BytesIO(b'{"message":{"content":"Annual leave is 25 days. [1]"}}')
            with patch.dict("os.environ", {"OLLAMA_HOST": "http://127.0.0.1:12345/"}), patch(
                "rag_assistant.answer.urlopen", return_value=response
            ) as http, patch("builtins.print") as output:
                main(["ask", "annual leave?", "--index", str(path), "--top-k", "1"])
            self.assertIn("handbook.pdf, page 2", output.call_args.args[0])
            request = http.call_args.args[0]
            self.assertEqual(request.full_url, "http://127.0.0.1:12345/api/chat")
            self.assertEqual(request.get_method(), "POST")
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], "gemma3:1b")
            self.assertIs(payload["stream"], False)
            self.assertEqual(payload["options"], {"temperature": 0, "seed": 42})
            system, user = payload["messages"]
            self.assertEqual(system, {"role": "system", "content": OLLAMA_INSTRUCTIONS})
            self.assertEqual(user["role"], "user")
            self.assertEqual(json.loads(user["content"]), {
                "question": "annual leave?",
                "excerpts": [{"label": 1, "text": "Annual leave is 25 days."}],
            })
            self.assertEqual([call.args for call in constructor.call_args_list],
                             [("sentence-transformers/all-MiniLM-L6-v2",)] * 2)

    def test_ollama_errors_and_citation_rules(self):
        host = "http://127.0.0.1:11434"
        model = "gemma3:1b"
        for error, message in (
            (URLError(ConnectionRefusedError("Connection refused")), "Check that it is running"),
            (HTTPError(host, 404, "Not Found", {}, None), "ollama pull"),
        ):
            with patch("rag_assistant.answer.urlopen", side_effect=error):
                with self.assertRaisesRegex(ValueError, message):
                    ollama_answer(host, model, "question")
        for body in (b"not JSON", b"{}", b'{"message":{"content":null}}'):
            with patch("rag_assistant.answer.urlopen", return_value=io.BytesIO(body)):
                with self.assertRaisesRegex(ValueError, "Malformed Ollama response"):
                    ollama_answer(host, model, "question")

        index = Index("test", [Chunk("handbook.pdf", 2, "leave")], [[1.0, 0.0]])
        for content in ("No labels", "Invalid [1] [2] [2]", "INSUFFICIENT_EVIDENCE"):
            body = json.dumps({"message": {"content": content}}).encode()
            with patch("rag_assistant.answer.urlopen", return_value=io.BytesIO(body)):
                if content == "INSUFFICIENT_EVIDENCE":
                    self.assertEqual(answer_question(
                        FakeClient(), index, "leave?", model, answer_provider="ollama"
                    ), NO_ANSWER)
                else:
                    with self.assertRaisesRegex(ValueError, "missing or invalid citations"):
                        answer_question(FakeClient(), index, "leave?", model, answer_provider="ollama")

    def test_ollama_rejects_cited_question_echo(self):
        question = "What experience does Amen have with Power BI?"
        index = Index("test", [Chunk("profile.pdf", 2, "Amen developed dashboards.")],
                      [[0.0, 1.0]])
        for content in (f"{question} [1]", f" ‘{question.lower().rstrip('?')}’ [1] "):
            with self.subTest(content=content):
                body = json.dumps({"message": {"content": content}}).encode()
                with patch("rag_assistant.answer.urlopen", return_value=io.BytesIO(body)):
                    with self.assertRaisesRegex(ValueError, "missing or invalid citations"):
                        answer_question(FakeClient(), index, question, "gemma3:1b",
                                        answer_provider="ollama")

    def test_ollama_deduplicates_labels_only_within_each_group(self):
        index = Index("test", [Chunk("handbook.pdf", 2, "leave"),
                               Chunk("handbook.pdf", 3, "policy")], [[1.0, 0.0], [0.0, 1.0]])
        generated = (
            "Annual leave is 25 days. [1] [1]  [2] Another sentence. [2] [2]\n"
            "- A separate bullet. [1]\n- Another bullet. [1]"
        )
        expected = (
            "Annual leave is 25 days. [1]  [2] Another sentence. [2]\n"
            "- A separate bullet. [1]\n- Another bullet. [1]\n\n"
            "Sources:\n[1] handbook.pdf, page 2\n[2] handbook.pdf, page 3"
        )
        body = json.dumps({"message": {"content": generated}}).encode()
        with patch("rag_assistant.answer.urlopen", return_value=io.BytesIO(body)):
            self.assertEqual(answer_question(
                FakeClient(), index, "leave?", "gemma3:1b", answer_provider="ollama"
            ), expected)

    def test_ask_requires_key_for_either_openai_provider(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", clear=True):
            path = Path(tmp) / "index.json"
            for embedding_provider, answer_provider in (("local", "openai"), ("openai", "ollama")):
                Index("test", [Chunk("handbook.pdf", 2, "leave")], [[1.0, 0.0]],
                      embedding_provider).save(path)
                with patch("sys.stderr", new_callable=io.StringIO) as stderr, self.assertRaises(SystemExit):
                    main(["ask", "leave?", "--index", str(path), "--answer-provider", answer_provider])
                self.assertIn("Set OPENAI_API_KEY", stderr.getvalue())

    def test_cli_ingest_and_ask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_pdf(root / "handbook.pdf")
            index = root / "index.json"
            client = FakeClient()
            module = SimpleNamespace(OpenAI=lambda **kwargs: client)
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only"}, clear=True), \
                 patch.dict("sys.modules", {"openai": module}), patch("builtins.print") as output:
                main(["ingest", str(root), "--embedding-provider", "openai", "--index", str(index)])
                payload = json.loads(index.read_text())
                self.assertEqual(payload.pop("provider"), "openai")
                index.write_text(json.dumps(payload))  # Original version-1 format.
                main(["ask", "annual leave?", "--index", str(index), "--top-k", "1",
                      "--answer-provider", "openai"])
                self.assertEqual(client.calls[-1][1]["model"], "gpt-4.1-mini")
                self.assertIn("handbook.pdf, page 2", output.call_args.args[0])
                self.assertTrue(index.exists())


if __name__ == "__main__":
    unittest.main()
