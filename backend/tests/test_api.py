"""
NyayaGuide AI — Phase 4/5 API Tests
Unit tests (mocked pipeline) and integration test (real OpenRouter).
"""
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.api.app import app
from app.models.document import RAGResponse, SourceCitation, RetrievalResult
from app.config import ABSTENTION_MESSAGE, MAX_QUESTION_LENGTH, is_openrouter_configured


# ──────────────────────────────────────────────
# Helpers: build mock RAG responses
# ──────────────────────────────────────────────
def _mock_rti_response() -> RAGResponse:
    return RAGResponse(
        question="How can I file an RTI application?",
        answer=(
            "To file an RTI application, submit a written request to the Public Information Officer "
            "with a fee of Rs 10.\n\nSources:\n\n1. The Right to Information Act, 2005\n"
            "   Section 6\n   Document: RTI_Act_2005.pdf (Page 6)\n   Source: India Code"
        ),
        sources=[
            SourceCitation(
                document="RTI_Act_2005.pdf",
                category="RTI",
                page=6,
                legal_reference="Section 6",
                title="The Right to Information Act, 2005",
                source="India Code",
                source_url="https://cic.gov.in",
                chunk_id="RTI_Act_2005_p6_c1",
            )
        ],
        retrieval_results=[
            RetrievalResult(
                rank=1, score=0.85, chunk_id="RTI_Act_2005_p6_c1",
                document="RTI_Act_2005.pdf", category="RTI", page=6,
                legal_reference="Section 6",
                title="The Right to Information Act, 2005",
                source="India Code", source_url="https://cic.gov.in",
                text="A person who desires to obtain any information shall make a request."
            )
        ],
        is_abstention=False,
        model_used="google/gemini-2.5-flash",
        top_score=0.85,
        follow_up_questions=[
            "What is the application fee for RTI?",
            "What is the time limit for a PIO to respond?",
            "What can I do if my RTI request is rejected?"
        ]
    )


def _mock_consumer_response() -> RAGResponse:
    return RAGResponse(
        question="What are my rights as a consumer?",
        answer=(
            "Under the Consumer Protection Act, 2019, consumers have the right to be protected "
            "against unfair trade practices.\n\nSources:\n\n1. Consumer Protection Act, 2019\n"
            "   Section 2\n   Document: Consumer_Protection_Act_2019.pdf (Page 5)\n"
            "   Source: Department of Consumer Affairs"
        ),
        sources=[
            SourceCitation(
                document="Consumer_Protection_Act_2019.pdf",
                category="CONSUMER",
                page=5,
                legal_reference="Section 2",
                title="Consumer Protection Act, 2019",
                source="Department of Consumer Affairs, Government of India",
                source_url="https://consumeraffairs.nic.in",
                chunk_id="CPA_2019_p5_c1",
            )
        ],
        retrieval_results=[],
        is_abstention=False,
        model_used="google/gemini-2.5-flash",
        top_score=0.78,
        follow_up_questions=[
            "How can I file a consumer complaint?",
            "What does the District Consumer Commission do?"
        ]
    )


def _mock_abstention_response(question: str) -> RAGResponse:
    return RAGResponse(
        question=question,
        answer=ABSTENTION_MESSAGE,
        sources=[],
        retrieval_results=[],
        is_abstention=True,
        model_used=None,
        top_score=0.42,
        follow_up_questions=[]
    )


# ──────────────────────────────────────────────
# Unit Tests (mocked pipeline — no OpenRouter)
# ──────────────────────────────────────────────
class TestAPIUnit(unittest.TestCase):
    """Unit tests using a mocked NyayaRAGPipeline."""

    @classmethod
    def setUpClass(cls):
        cls.mock_pipeline = MagicMock()
        app.state.pipeline = cls.mock_pipeline
        cls.client = TestClient(app, raise_server_exceptions=False)

    # 1. Health endpoint
    def test_health_returns_ok(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")

    # 2. RTI question
    def test_ask_rti_question(self):
        self.mock_pipeline.ask.return_value = _mock_rti_response()
        resp = self.client.post("/api/ask", json={"question": "How can I file an RTI application?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["question"], "How can I file an RTI application?")
        self.assertIn("RTI", data["answer"])
        self.assertFalse(data["is_abstention"])
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["document"], "RTI_Act_2005.pdf")
        self.assertEqual(data["sources"][0]["page"], 6)
        self.assertEqual(data["sources"][0]["legal_reference"], "Section 6")
        self.assertEqual(data["sources"][0]["source_url"], "https://cic.gov.in")
        self.assertEqual(data["model_used"], "google/gemini-2.5-flash")
        self.assertGreater(data["top_score"], 0.0)
        self.assertEqual(len(data["follow_up_questions"]), 3)

    # 3. Consumer question
    def test_ask_consumer_question(self):
        self.mock_pipeline.ask.return_value = _mock_consumer_response()
        resp = self.client.post("/api/ask", json={"question": "What are my rights as a consumer?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("Consumer", data["answer"])
        self.assertFalse(data["is_abstention"])
        self.assertEqual(data["sources"][0]["category"], "CONSUMER")
        self.assertEqual(data["sources"][0]["source_url"], "https://consumeraffairs.nic.in")
        self.assertEqual(len(data["follow_up_questions"]), 2)

    # 4. Empty question
    def test_empty_question_returns_422(self):
        resp = self.client.post("/api/ask", json={"question": ""})
        self.assertEqual(resp.status_code, 422)

    # 5. Whitespace-only question
    def test_whitespace_only_question_returns_422(self):
        resp = self.client.post("/api/ask", json={"question": "   "})
        self.assertEqual(resp.status_code, 422)

    # 6. Missing question field (invalid body)
    def test_missing_question_field_returns_422(self):
        resp = self.client.post("/api/ask", json={"query": "test"})
        self.assertEqual(resp.status_code, 422)

    # 7. Too-long question
    def test_too_long_question_returns_422(self):
        long_q = "a" * (MAX_QUESTION_LENGTH + 1)
        resp = self.client.post("/api/ask", json={"question": long_q})
        self.assertEqual(resp.status_code, 422)

    # 8. Out-of-domain question (abstention)
    def test_out_of_domain_abstention(self):
        self.mock_pipeline.ask.return_value = _mock_abstention_response("What is the weather?")
        resp = self.client.post("/api/ask", json={"question": "What is the weather?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["is_abstention"])
        self.assertEqual(data["answer"], ABSTENTION_MESSAGE)
        self.assertEqual(len(data["sources"]), 0)
        self.assertEqual(len(data["follow_up_questions"]), 0)
        self.assertIsNone(data["model_used"])

    # 9. Pipeline error → 500 with safe message
    def test_unexpected_error_returns_500(self):
        self.mock_pipeline.ask.side_effect = Exception("Kaboom!")
        resp = self.client.post("/api/ask", json={"question": "test question"})
        self.assertEqual(resp.status_code, 500)
        data = resp.json()
        self.assertIn("internal server error", data["detail"].lower())
        self.assertNotIn("Kaboom", data["detail"])
        self.mock_pipeline.ask.side_effect = None

    # 10. OpenRouter auth error → 502
    def test_openrouter_auth_error_returns_502(self):
        self.mock_pipeline.ask.side_effect = PermissionError("Auth failed")
        resp = self.client.post("/api/ask", json={"question": "test question"})
        self.assertEqual(resp.status_code, 502)
        data = resp.json()
        self.assertIn("authentication", data["detail"].lower())
        self.mock_pipeline.ask.side_effect = None

    # 11. OpenRouter timeout → 504
    def test_openrouter_timeout_returns_504(self):
        self.mock_pipeline.ask.side_effect = TimeoutError("Timed out")
        resp = self.client.post("/api/ask", json={"question": "test question"})
        self.assertEqual(resp.status_code, 504)
        data = resp.json()
        self.assertIn("timed out", data["detail"].lower())
        self.mock_pipeline.ask.side_effect = None

    # 12. OpenRouter rate limit → 502
    def test_openrouter_rate_limit_returns_502(self):
        self.mock_pipeline.ask.side_effect = RuntimeError("rate limit reached")
        resp = self.client.post("/api/ask", json={"question": "test question"})
        self.assertEqual(resp.status_code, 502)
        data = resp.json()
        self.assertIn("rate limit", data["detail"].lower())
        self.mock_pipeline.ask.side_effect = None

    # 13. No secrets in error responses
    def test_no_secrets_in_error_responses(self):
        import os
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        hf_token = os.getenv("HF_TOKEN", "")

        self.mock_pipeline.ask.side_effect = RuntimeError(f"Error with key {api_key}")
        resp = self.client.post("/api/ask", json={"question": "test question"})
        data = resp.json()
        if api_key:
            self.assertNotIn(api_key, data["detail"])
        if hf_token:
            self.assertNotIn(hf_token, data["detail"])
        self.mock_pipeline.ask.side_effect = None

    # 14. Health does NOT have secrets
    def test_health_no_secrets(self):
        resp = self.client.get("/health")
        body = resp.text
        self.assertNotIn("OPENROUTER_API_KEY", body)
        self.assertNotIn("HF_TOKEN", body)

    # 15. CORS headers present for allowed origin
    def test_cors_headers_for_allowed_origin(self):
        resp = self.client.options(
            "/api/ask",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertIn(resp.status_code, [200, 204])
        self.assertIn("access-control-allow-origin", resp.headers)

    # 16. OpenAPI schema available
    def test_openapi_json_available(self):
        resp = self.client.get("/openapi.json")
        self.assertEqual(resp.status_code, 200)
        schema = resp.json()
        self.assertIn("openapi", schema)
        self.assertIn("/health", schema["paths"])
        self.assertIn("/api/ask", schema["paths"])

    # 17. Docs page available
    def test_docs_available(self):
        resp = self.client.get("/docs")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers.get("content-type", ""))

    # 18. Structured JSON parsing & follow-up validation in RAG pipeline
    def test_rag_pipeline_follow_up_parsing(self):
        from app.rag.rag_pipeline import NyayaRAGPipeline
        pipeline = NyayaRAGPipeline(
            retriever=MagicMock(),
            llm_client=MagicMock()
        )
        # Test valid JSON with markdown code blocks
        raw = '```json\n{"answer": "Test answer", "follow_up_questions": ["Q1?", "Q2?", "Q1?", "", "Q3?", "Q4?", "Q5?"]}\n```'
        ans, follow_ups = pipeline._parse_llm_response(raw)
        self.assertEqual(ans, "Test answer")
        self.assertEqual(len(follow_ups), 4) # Capped at 4, duplicates & empties removed
        self.assertEqual(follow_ups, ["Q1?", "Q2?", "Q3?", "Q4?"])


# ──────────────────────────────────────────────
# Integration Test (real OpenRouter)
# ──────────────────────────────────────────────
class TestAPIIntegration(unittest.TestCase):
    """Live integration test — requires OPENROUTER_API_KEY configured."""

    @classmethod
    def setUpClass(cls):
        if not is_openrouter_configured():
            raise unittest.SkipTest(
                "OPENROUTER_API_KEY not configured. Skipping live API integration test."
            )
        real_pipeline = __import__("app.rag.rag_pipeline", fromlist=["NyayaRAGPipeline"]).NyayaRAGPipeline()
        app.state.pipeline = real_pipeline
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_live_rti_question(self):
        resp = self.client.post(
            "/api/ask",
            json={"question": "How can I file an RTI application?"},
        )
        if resp.status_code == 502:
            raise unittest.SkipTest(f"OpenRouter upstream service error during live test: {resp.json().get('detail')}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["is_abstention"])
        self.assertIn("Sources:", data["answer"])
        self.assertGreater(len(data["sources"]), 0)
        self.assertIsNotNone(data["model_used"])
        self.assertGreater(data["top_score"], 0.50)
        print(f"\n[LIVE] RTI Answer received: {len(data['answer'])} chars, "
              f"{len(data['sources'])} sources, top_score={data['top_score']:.4f}")
        print(f"[LIVE] Follow-up questions ({len(data.get('follow_up_questions', []))}): {data.get('follow_up_questions', [])}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
