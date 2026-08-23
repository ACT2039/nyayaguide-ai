"""
NyayaGuide AI — RAG Domain Classification & Duplicate Suggestion Tests

Tests:
1. "How can I file an RTI application?" -> recognised as RTI/in-domain
2. Valid RTI question with matching indexed context -> grounded answer (not abstention)
3. RTI question with initially weak retrieval (border-line) -> secondary pass picks it up
4. Out-of-domain question (weather) -> remains abstained
5. Exact duplicate suggestion rejected
6. Case/whitespace/punctuation variation treated as duplicate
7. Suggestions contain useful alternatives after filtering
8. Consumer Protection questions continue to pass
9. _filter_duplicate_suggestions edge cases
10. _detect_domain keyword coverage
"""
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.rag.rag_pipeline import (
    NyayaRAGPipeline,
    _normalise_text,
    _detect_domain,
    _filter_duplicate_suggestions,
    _SECONDARY_RELEVANCE_THRESHOLD,
)
from app.config import MIN_RELEVANCE_THRESHOLD, ABSTENTION_MESSAGE
from app.models.document import RAGResponse, RetrievalResult, SourceCitation


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────

def _make_retrieval_result(score: float, doc: str = "RTI_Act_2005.pdf", cat: str = "RTI", page: int = 6) -> RetrievalResult:
    return RetrievalResult(
        rank=1, score=score, chunk_id="test_chunk_001",
        document=doc, category=cat, page=page,
        legal_reference="Section 6",
        title="The Right to Information Act, 2005",
        source="India Code", source_url="https://cic.gov.in",
        text="A person who desires to obtain any information shall make a request to the PIO."
    )


def _make_consumer_result(score: float) -> RetrievalResult:
    return RetrievalResult(
        rank=1, score=score, chunk_id="consumer_chunk_001",
        document="Consumer_Protection_Act_2019.pdf", category="CONSUMER", page=5,
        legal_reference="Section 2",
        title="Consumer Protection Act, 2019",
        source="Department of Consumer Affairs", source_url="https://consumeraffairs.nic.in",
        text="A consumer means any person who buys goods for consideration."
    )


# ─────────────────────────────────────────────────
# Test 1-4: Domain detection utility
# ─────────────────────────────────────────────────
class TestDomainDetection(unittest.TestCase):

    def test_rti_application_question_detected_as_rti(self):
        """Test 1: 'How can I file an RTI application?' must be recognised as RTI."""
        result = _detect_domain("How can I file an RTI application?")
        self.assertEqual(result, "RTI", "RTI application question must be detected as RTI domain")

    def test_rti_keyword_variants_detected(self):
        """Test 1b: Various RTI phrasings must be detected as in-domain."""
        rti_questions = [
            "What is the procedure for filing an RTI?",
            "Who is the Public Information Officer?",
            "How do I file a right to information request?",
            "What is the RTI fee?",
            "What are RTI Rules 2012?",
            "How to appeal an RTI decision?",
        ]
        for q in rti_questions:
            with self.subTest(question=q):
                result = _detect_domain(q)
                self.assertEqual(result, "RTI", f"Expected RTI domain for: {q!r}")

    def test_consumer_questions_detected(self):
        """Test 8: Consumer Protection questions must be detected as CONSUMER domain."""
        consumer_questions = [
            "What are my rights as a consumer?",
            "How do I file a consumer complaint?",
            "What does the District Consumer Commission do?",
            "What is consumer protection act?",
        ]
        for q in consumer_questions:
            with self.subTest(question=q):
                result = _detect_domain(q)
                self.assertEqual(result, "CONSUMER", f"Expected CONSUMER domain for: {q!r}")

    def test_out_of_domain_returns_none(self):
        """Test 4: Out-of-domain questions must return None from domain detection."""
        ood_questions = [
            "What is the weather in Mumbai today?",
            "Who is the prime minister of India?",
            "How do I write a Python script?",
            "What is the stock price of Reliance?",
        ]
        for q in ood_questions:
            with self.subTest(question=q):
                result = _detect_domain(q)
                self.assertIsNone(result, f"Expected None domain for: {q!r}")


# ─────────────────────────────────────────────────
# Test 5-7: Duplicate suggestion filter
# ─────────────────────────────────────────────────
class TestDuplicateSuggestionFilter(unittest.TestCase):

    def test_exact_duplicate_suggestion_rejected(self):
        """Test 5: Suggestion exactly equal to current question must be rejected."""
        question = "How can I file an RTI application?"
        suggestions = [
            "How can I file an RTI application?",  # exact duplicate -> must be removed
            "What is the RTI fee?",
            "How long does a PIO have to respond?",
        ]
        filtered = _filter_duplicate_suggestions(question, suggestions)
        self.assertNotIn("How can I file an RTI application?", filtered)
        self.assertIn("What is the RTI fee?", filtered)
        self.assertIn("How long does a PIO have to respond?", filtered)

    def test_case_whitespace_punctuation_variation_treated_as_duplicate(self):
        """Test 6: Case/whitespace/punctuation variants must be treated as duplicates."""
        question = "How can I file an RTI application?"
        duplicate_variants = [
            " how can i file an rti application? ",  # leading/trailing whitespace + lowercase
            "How  Can  I  File  An  RTI  Application?",  # double spaces
            "how can i file an rti application",  # no punctuation
            "HOW CAN I FILE AN RTI APPLICATION?",  # uppercase
        ]
        for variant in duplicate_variants:
            with self.subTest(variant=variant):
                filtered = _filter_duplicate_suggestions(question, [variant, "What is the RTI fee?"])
                self.assertNotIn(variant, filtered,
                    f"Variant {variant!r} should have been filtered as duplicate of {question!r}")

    def test_useful_suggestions_remain_after_filtering(self):
        """Test 7: After duplicate filtering, useful alternative suggestions remain."""
        question = "How can I file an RTI application?"
        suggestions = [
            "How can I file an RTI application?",  # duplicate -> removed
            "What is the application fee for filing an RTI request?",
            "What information can a citizen request under RTI?",
            "What is the time limit for a Public Information Officer to respond?",
            "What can I do if my RTI request is rejected?",
        ]
        filtered = _filter_duplicate_suggestions(question, suggestions)
        self.assertGreater(len(filtered), 0, "Should have useful suggestions after filtering")
        self.assertNotIn("How can I file an RTI application?", filtered)
        self.assertIn("What is the application fee for filing an RTI request?", filtered)

    def test_empty_suggestions_filtered(self):
        """Test 9a: Empty suggestions must be removed."""
        question = "test question"
        suggestions = ["", "   ", "Valid suggestion?", ""]
        filtered = _filter_duplicate_suggestions(question, suggestions)
        self.assertEqual(filtered, ["Valid suggestion?"])

    def test_deduplication_within_suggestions_list(self):
        """Test 9b: Duplicate suggestions within the list itself must be removed."""
        question = "test question"
        suggestions = ["Same suggestion?", "Same suggestion?", "Other question?"]
        filtered = _filter_duplicate_suggestions(question, suggestions)
        self.assertEqual(filtered.count("Same suggestion?"), 1)

    def test_no_false_positive_filtering(self):
        """Test 9c: Non-duplicate suggestions must NOT be filtered."""
        question = "How can I file an RTI application?"
        suggestions = [
            "What is the RTI fee?",
            "What are consumer rights?",
            "How to file an appeal?",
        ]
        filtered = _filter_duplicate_suggestions(question, suggestions)
        self.assertEqual(len(filtered), 3)


# ─────────────────────────────────────────────────
# Test normalise_text utility
# ─────────────────────────────────────────────────
class TestNormaliseText(unittest.TestCase):

    def test_lowercase(self):
        self.assertEqual(_normalise_text("HOW CAN I FILE AN RTI?"), "how can i file an rti")

    def test_trim_whitespace(self):
        self.assertEqual(_normalise_text("  hello world  "), "hello world")

    def test_collapse_whitespace(self):
        self.assertEqual(_normalise_text("how  can   i   file"), "how can i file")

    def test_remove_trailing_punctuation(self):
        self.assertEqual(_normalise_text("how can i file an rti?"), "how can i file an rti")
        self.assertEqual(_normalise_text("hello."), "hello")
        self.assertEqual(_normalise_text("hello!"), "hello")

    def test_combined_normalization(self):
        a = _normalise_text("How can I file an RTI application?")
        b = _normalise_text(" how can i file an rti application? ")
        self.assertEqual(a, b)


# ─────────────────────────────────────────────────
# Test Pipeline: two-pass retrieval logic
# ─────────────────────────────────────────────────
class TestTwoPassRetrieval(unittest.TestCase):

    def _build_pipeline(self, pass1_results, pass2_results=None):
        """Build a pipeline with a mocked retriever."""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        mock_llm.model = "test-model"  # must be a string for Pydantic

        if pass2_results is not None:
            mock_retriever.retrieve.side_effect = [pass1_results, pass2_results]
        else:
            mock_retriever.retrieve.return_value = pass1_results

        mock_retriever.reload_index.return_value = True

        pipeline = NyayaRAGPipeline(
            retriever=mock_retriever,
            llm_client=mock_llm
        )
        return pipeline, mock_retriever, mock_llm

    def test_rti_question_with_good_score_answers_directly(self):
        """Test 2: RTI question with score >= threshold -> answers without secondary pass."""
        results = [_make_retrieval_result(score=0.74)]
        pipeline, mock_retriever, mock_llm = self._build_pipeline(results)
        mock_llm.generate.return_value = '{"answer": "You can file RTI at the PIO.", "follow_up_questions": ["What is the RTI fee?"]}'

        resp = pipeline.ask("How can I file an RTI application?")
        self.assertFalse(resp.is_abstention)
        self.assertEqual(mock_retriever.retrieve.call_count, 1)  # only one pass needed

    def test_rti_question_borderline_uses_secondary_pass(self):
        """Test 3: RTI question with borderline score triggers secondary pass."""
        # Pass 1: score below threshold
        pass1 = [_make_retrieval_result(score=0.42)]
        # Pass 2: broader retrieval returns better score
        pass2 = [
            _make_retrieval_result(score=0.55),
            _make_retrieval_result(score=0.52, page=7),
        ]
        pipeline, mock_retriever, mock_llm = self._build_pipeline(pass1, pass2)
        mock_llm.generate.return_value = '{"answer": "RTI grounded answer.", "follow_up_questions": ["What is the RTI fee?"]}'

        resp = pipeline.ask("How can I file an RTI application?")
        self.assertFalse(resp.is_abstention, "In-domain RTI question with secondary pass score should not abstain")
        self.assertEqual(mock_retriever.retrieve.call_count, 2)  # both passes used

    def test_out_of_domain_question_abstains_without_secondary_pass(self):
        """Test 4: Weather question abstains without triggering secondary pass."""
        results = [_make_retrieval_result(score=0.45, doc="RTI_Act_2005.pdf")]
        pipeline, mock_retriever, _ = self._build_pipeline(results)

        resp = pipeline.ask("What is the weather in Mumbai today?")
        self.assertTrue(resp.is_abstention)
        # Should NOT attempt secondary pass since domain is None
        self.assertEqual(mock_retriever.retrieve.call_count, 1)

    def test_both_passes_fail_leads_to_abstention(self):
        """Test 3b: If secondary pass also fails threshold, correctly abstains."""
        pass1 = [_make_retrieval_result(score=0.30)]
        pass2 = [_make_retrieval_result(score=0.32)]
        pipeline, mock_retriever, _ = self._build_pipeline(pass1, pass2)

        resp = pipeline.ask("How can I file an RTI application?")
        self.assertTrue(resp.is_abstention)
        self.assertEqual(mock_retriever.retrieve.call_count, 2)

    def test_consumer_question_passes(self):
        """Test 8: Consumer Protection question with good score answers correctly."""
        results = [_make_consumer_result(score=0.70)]
        pipeline, mock_retriever, mock_llm = self._build_pipeline(results)
        mock_llm.generate.return_value = '{"answer": "As a consumer you have rights under CPA 2019.", "follow_up_questions": ["How to file complaint?"]}'

        resp = pipeline.ask("What are my rights as a consumer?")
        self.assertFalse(resp.is_abstention)

    def test_duplicate_suggestion_filtered_in_pipeline(self):
        """Test 5: Pipeline filters follow-up suggestions that duplicate the current question."""
        results = [_make_retrieval_result(score=0.75)]
        pipeline, mock_retriever, mock_llm = self._build_pipeline(results)
        # LLM returns the user's own question as a suggestion
        mock_llm.generate.return_value = (
            '{"answer": "RTI answer.", "follow_up_questions": '
            '["How can I file an RTI application?", "What is the RTI fee?", "Who is the PIO?"]}'
        )

        resp = pipeline.ask("How can I file an RTI application?")
        self.assertFalse(resp.is_abstention)
        self.assertNotIn("How can I file an RTI application?", resp.follow_up_questions)
        self.assertIn("What is the RTI fee?", resp.follow_up_questions)

    def test_duplicate_suggestion_case_variation_filtered(self):
        """Test 6: Case/whitespace variation of current question is filtered from suggestions."""
        results = [_make_retrieval_result(score=0.75)]
        pipeline, _, mock_llm = self._build_pipeline(results)
        mock_llm.generate.return_value = (
            '{"answer": "RTI answer.", "follow_up_questions": '
            '[" how can i file an rti application? ", "What is the RTI fee?"]}'
        )

        resp = pipeline.ask("How can I file an RTI application?")
        # The whitespace/case variant should be filtered
        for suggestion in resp.follow_up_questions:
            self.assertNotEqual(
                _normalise_text(suggestion),
                _normalise_text("How can I file an RTI application?"),
                f"Suggestion {suggestion!r} is a duplicate of the current question"
            )

    def test_empty_index_abstains(self):
        """Production scenario: empty FAISS index (0 vectors) -> abstention."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []  # empty index
        mock_llm = MagicMock()
        pipeline = NyayaRAGPipeline(retriever=mock_retriever, llm_client=mock_llm)

        resp = pipeline.ask("How can I file an RTI application?")
        self.assertTrue(resp.is_abstention)
        self.assertEqual(resp.answer, ABSTENTION_MESSAGE)


# ─────────────────────────────────────────────────
# Test the secondary threshold constant
# ─────────────────────────────────────────────────
class TestThresholdConstants(unittest.TestCase):

    def test_secondary_threshold_lower_than_primary(self):
        """Secondary threshold must be lower than primary threshold."""
        self.assertLess(
            _SECONDARY_RELEVANCE_THRESHOLD,
            MIN_RELEVANCE_THRESHOLD,
            "Secondary threshold must be below primary threshold"
        )

    def test_secondary_threshold_reasonable(self):
        """Secondary threshold must be in a reasonable range (0.35 - 0.49)."""
        self.assertGreaterEqual(_SECONDARY_RELEVANCE_THRESHOLD, 0.35)
        self.assertLess(_SECONDARY_RELEVANCE_THRESHOLD, MIN_RELEVANCE_THRESHOLD)


if __name__ == "__main__":
    unittest.main(verbosity=2)
