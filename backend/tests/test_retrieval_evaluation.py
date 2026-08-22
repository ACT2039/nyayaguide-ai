"""
Retrieval Evaluation Tests for NyayaGuide AI
Evaluates semantic retrieval across 5 target civic/legal queries:
1. "How can I file an RTI application?"
2. "What information can a citizen request under RTI?"
3. "What are my rights as a consumer?"
4. "How can I file a consumer complaint?"
5. "What does the District Consumer Commission do?"
"""
import sys
import unittest
from pathlib import Path

# Reconfigure stdout for UTF-8 compatibility on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.retrieval.retriever import NyayaRetriever, retrieve
from app.models.document import RetrievalResult


class TestRetrievalEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 65)
        print("Setting up FAISS Index & Retriever for Evaluation...")
        print("=" * 65)
        cls.retriever = NyayaRetriever()
        cls.retriever.ensure_index_loaded()

    def _evaluate_query(self, query: str, expected_category: str, top_k: int = 5):
        print(f"\nQUERY: \"{query}\" [Expected Domain: {expected_category}]")
        print("-" * 65)

        results = self.retriever.retrieve(query=query, top_k=top_k)

        # Assertion 1: Valid count
        self.assertEqual(len(results), top_k)

        # Assertion 2: All results have valid structure and scores
        matching_domain_count = 0
        for res in results:
            self.assertIsInstance(res, RetrievalResult)
            self.assertGreater(res.score, 0.0)
            self.assertLessEqual(res.score, 1.0)
            self.assertTrue(bool(res.chunk_id))
            self.assertTrue(bool(res.document))
            self.assertTrue(bool(res.category))
            self.assertTrue(bool(res.text))
            self.assertGreater(res.page, 0)

            if res.category.upper() == expected_category.upper():
                matching_domain_count += 1

            # Format safe snippet
            snippet = res.text[:140].replace("\n", " ") + "..."
            print(
                f"  Rank {res.rank} | Score: {res.score:.4f} | "
                f"[{res.category}] {res.document} (p.{res.page}) | "
                f"Ref: {res.legal_reference or 'N/A'}\n"
                f"         Preview: \"{snippet}\""
            )

        # Assertion 3: Domain relevance (majority of top results should match expected legal category)
        self.assertGreaterEqual(
            matching_domain_count, 3,
            f"Expected at least 3/{top_k} results from {expected_category}, got {matching_domain_count}"
        )

        # Top 1 result must match domain
        self.assertEqual(
            results[0].category.upper(), expected_category.upper(),
            f"Top-1 result for '{query}' should be {expected_category}, got {results[0].category} ({results[0].document})"
        )

        return results

    def test_query_1_file_rti_application(self):
        self._evaluate_query("How can I file an RTI application?", expected_category="RTI")

    def test_query_2_information_requested_under_rti(self):
        self._evaluate_query("What information can a citizen request under RTI?", expected_category="RTI")

    def test_query_3_consumer_rights(self):
        self._evaluate_query("What are my rights as a consumer?", expected_category="CONSUMER")

    def test_query_4_file_consumer_complaint(self):
        self._evaluate_query("How can I file a consumer complaint?", expected_category="CONSUMER")

    def test_query_5_district_consumer_commission(self):
        self._evaluate_query("What does the District Consumer Commission do?", expected_category="CONSUMER")


if __name__ == "__main__":
    unittest.main(verbosity=2)
