"""
Live Evaluation of NyayaGuide AI End-to-End RAG Generation via OpenRouter.
Evaluates the 5 legal questions and 2 out-of-domain questions.
"""
import sys
import time
import unittest
from pathlib import Path

# Reconfigure stdout for UTF-8 compatibility on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import is_openrouter_configured, ABSTENTION_MESSAGE, OPENROUTER_MODEL
from app.rag.rag_pipeline import NyayaRAGPipeline


class TestRAGLiveEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not is_openrouter_configured():
            raise unittest.SkipTest("OPENROUTER_API_KEY is not configured in .env. Skipping live LLM evaluation.")
        print("\n" + "=" * 65)
        print("Initializing NyayaGuide AI RAG Pipeline for Live Generation...")
        print(f"Active Model: {OPENROUTER_MODEL}")
        print("=" * 65)
        cls.pipeline = NyayaRAGPipeline()

    def _run_and_display(self, query: str, expect_abstention: bool = False):
        print(f"\n" + "=" * 65)
        print(f"QUESTION: \"{query}\"")
        print("-" * 65)

        try:
            response = self.pipeline.ask(query)
        except RuntimeError as e:
            if "402" in str(e) or "429" in str(e) or "quota" in str(e).lower() or "payment" in str(e).lower() or "rate limit" in str(e).lower():
                raise unittest.SkipTest(f"OpenRouter credit/quota/rate-limit hit: {e}")
            raise

        print(f"Top Score     : {response.top_score:.4f}")
        print(f"Abstention    : {response.is_abstention}")
        print(f"Model Used    : {response.model_used or 'N/A'}")
        print(f"Sources Count : {len(response.sources)}")
        print(f"Follow-ups    : {len(response.follow_up_questions)}")
        print("-" * 65)
        print("GENERATED RESPONSE:")
        print(response.answer)
        print("=" * 65)

        if expect_abstention:
            self.assertTrue(response.is_abstention)
            self.assertEqual(response.answer, ABSTENTION_MESSAGE)
            self.assertEqual(len(response.sources), 0)
            self.assertEqual(len(response.follow_up_questions), 0)
        else:
            self.assertFalse(response.is_abstention)
            self.assertNotEqual(response.answer, ABSTENTION_MESSAGE)
            self.assertGreater(len(response.sources), 0)
            self.assertIn("Sources:", response.answer)

        return response

    def test_q1_file_rti_application(self):
        self._run_and_display("How can I file an RTI application?", expect_abstention=False)

    def test_q2_info_requested_under_rti(self):
        self._run_and_display("What information can a citizen request under RTI?", expect_abstention=False)

    def test_q3_consumer_rights(self):
        self._run_and_display("What are my rights as a consumer?", expect_abstention=False)

    def test_q4_file_consumer_complaint(self):
        self._run_and_display("How can I file a consumer complaint?", expect_abstention=False)

    def test_q5_district_consumer_commission(self):
        self._run_and_display("What does the District Consumer Commission do?", expect_abstention=False)

    def test_q6_out_of_domain_weather(self):
        self._run_and_display("What is the weather tomorrow?", expect_abstention=True)

    def test_q7_out_of_domain_python_sort(self):
        self._run_and_display("Write a Python program to sort an array.", expect_abstention=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
