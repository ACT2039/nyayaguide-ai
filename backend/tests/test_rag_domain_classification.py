"""
NyayaGuide AI — RAG Domain Classification & State Machine Tests

Tests all 14 required scenarios:
 1.  RTI query recognised as in-domain
 2.  RTI query with valid indexed context → grounded answer
 3.  Consumer Protection query recognised as in-domain
 4.  Unrelated query → OUT_OF_DOMAIN
 5.  Empty FAISS index → KNOWLEDGE_BASE_UNAVAILABLE (NOT OUT_OF_DOMAIN)
 6.  In-domain query + insufficient context → IN_DOMAIN_BUT_INSUFFICIENT
 7.  Secondary retrieval finds relevant context → ANSWERABLE
 8.  Exact duplicate suggestion filtered
 9.  Case/whitespace/punctuation duplicate filtered
10.  Useful suggestions remain after filtering
11.  Existing API tests continue passing (imported separately)
12.  Existing document management tests continue passing (imported separately)
13.  Existing HF persistence tests continue passing (imported separately)
14.  Existing RAG tests continue passing (imported separately)
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
    _KB_UNAVAILABLE_MESSAGE,
)
from app.models.document import (
    RAGResponse,
    RetrievalResult,
    SourceCitation,
    QueryResultState,
)
from app.config import MIN_RELEVANCE_THRESHOLD, ABSTENTION_MESSAGE


# ─────────────────────────────────────────────────
# Test data helpers
# ─────────────────────────────────────────────────

def _rti_result(score: float, page: int = 6) -> RetrievalResult:
    return RetrievalResult(
        rank=1, score=score, chunk_id=f"rti_chunk_p{page}",
        document="RTI_Act_2005.pdf", category="RTI", page=page,
        legal_reference="Section 6",
        title="The Right to Information Act, 2005",
        source="India Code",
        source_url="https://cic.gov.in",
        text=(
            "A person who desires to obtain any information under this Act shall "
            "make a request in writing or through electronic means in English or "
            "Hindi or in the official language of the area in which the application "
            "is being made, to the Central Public Information Officer or State Public "
            "Information Officer."
        ),
    )


def _consumer_result(score: float) -> RetrievalResult:
    return RetrievalResult(
        rank=1, score=score, chunk_id="consumer_chunk_p5",
        document="Consumer_Protection_Act_2019.pdf", category="CONSUMER", page=5,
        legal_reference="Section 2",
        title="The Consumer Protection Act, 2019",
        source="Department of Consumer Affairs, Government of India",
        source_url="https://consumeraffairs.nic.in",
        text=(
            "A consumer means any person who buys any goods for a consideration "
            "which has been paid or promised or partly paid and partly promised, "
            "or under any system of deferred payment."
        ),
    )


def _build_pipeline(pass1_results, pass2_results=None):
    """Build a NyayaRAGPipeline with mocked retriever and LLM."""
    mock_retriever = MagicMock()
    mock_retriever.reload_index.return_value = True

    # Mock the vector_store so total_vectors reflects the index state
    mock_vs = MagicMock()
    mock_vs.total_vectors = len(pass1_results)
    mock_retriever.vector_store = mock_vs

    if pass2_results is not None:
        mock_retriever.retrieve.side_effect = [pass1_results, pass2_results]
    else:
        mock_retriever.retrieve.return_value = pass1_results

    mock_llm = MagicMock()
    mock_llm.model = "test-model"

    pipeline = NyayaRAGPipeline(retriever=mock_retriever, llm_client=mock_llm)
    return pipeline, mock_retriever, mock_llm


# ─────────────────────────────────────────────────
# 1 & 3: Domain detection
# ─────────────────────────────────────────────────
class TestDomainDetection(unittest.TestCase):

    def test_rti_application_question_detected(self):
        """Test 1: 'How can I file an RTI application?' → RTI domain."""
        self.assertEqual(_detect_domain("How can I file an RTI application?"), "RTI")

    def test_rti_keyword_variants(self):
        """Test 1 (variants): Various RTI phrasings detected."""
        cases = [
            "What is the procedure for filing an RTI?",
            "Who is the Public Information Officer?",
            "How do I file a right to information request?",
            "What is the RTI fee for applying?",
            "How do I file a first appeal under RTI?",
            "What information can I request under RTI Act?",
        ]
        for q in cases:
            with self.subTest(question=q):
                self.assertEqual(_detect_domain(q), "RTI")

    def test_consumer_questions_detected(self):
        """Test 3: Consumer Protection questions → CONSUMER domain."""
        cases = [
            "What are my rights as a consumer?",
            "How do I file a consumer complaint?",
            "What does the District Consumer Commission do?",
            "What is the Consumer Protection Act?",
            "How can I get a refund for defective goods?",
        ]
        for q in cases:
            with self.subTest(question=q):
                self.assertEqual(_detect_domain(q), "CONSUMER")

    def test_out_of_domain_returns_none(self):
        """Test 4 (domain check): Unrelated queries return None."""
        cases = [
            "What is the weather in Mumbai today?",
            "Who is the prime minister of India?",
            "How do I write a Python script?",
            "What is the stock price of Reliance?",
            "Tell me a joke.",
        ]
        for q in cases:
            with self.subTest(question=q):
                self.assertIsNone(_detect_domain(q))


# ─────────────────────────────────────────────────
# 5: Empty index → KNOWLEDGE_BASE_UNAVAILABLE
# ─────────────────────────────────────────────────
class TestEmptyIndexGuard(unittest.TestCase):

    def test_empty_index_returns_knowledge_base_unavailable(self):
        """
        Test 5: When FAISS index has 0 vectors, any query must return
        KNOWLEDGE_BASE_UNAVAILABLE — NOT OUT_OF_DOMAIN.
        """
        mock_retriever = MagicMock()
        mock_vs = MagicMock()
        mock_vs.total_vectors = 0
        mock_retriever.vector_store = mock_vs

        mock_llm = MagicMock()
        mock_llm.model = "test-model"

        pipeline = NyayaRAGPipeline(retriever=mock_retriever, llm_client=mock_llm)

        resp = pipeline.ask("How can I file an RTI application?")

        self.assertTrue(resp.is_abstention)
        self.assertEqual(resp.query_result_state, QueryResultState.KNOWLEDGE_BASE_UNAVAILABLE)
        self.assertNotEqual(resp.query_result_state, QueryResultState.OUT_OF_DOMAIN)
        self.assertEqual(resp.answer, _KB_UNAVAILABLE_MESSAGE)
        self.assertEqual(resp.top_score, 0.0)
        self.assertEqual(resp.follow_up_questions, [])

        # Retriever.retrieve must NOT be called when index is empty
        mock_retriever.retrieve.assert_not_called()

    def test_empty_index_ood_question_also_returns_kb_unavailable(self):
        """
        Test 5b: Even an out-of-domain question returns KNOWLEDGE_BASE_UNAVAILABLE
        (not OUT_OF_DOMAIN) when the index is empty.
        """
        mock_retriever = MagicMock()
        mock_vs = MagicMock()
        mock_vs.total_vectors = 0
        mock_retriever.vector_store = mock_vs

        mock_llm = MagicMock()
        mock_llm.model = "test-model"

        pipeline = NyayaRAGPipeline(retriever=mock_retriever, llm_client=mock_llm)

        resp = pipeline.ask("What is the weather in Mumbai today?")
        self.assertEqual(resp.query_result_state, QueryResultState.KNOWLEDGE_BASE_UNAVAILABLE)

    def test_kb_unavailable_answer_is_not_abstention_message(self):
        """
        Test 5c: KNOWLEDGE_BASE_UNAVAILABLE response uses a service-condition message,
        NOT the standard abstention/out-of-domain message.
        """
        mock_retriever = MagicMock()
        mock_vs = MagicMock()
        mock_vs.total_vectors = 0
        mock_retriever.vector_store = mock_vs

        mock_llm = MagicMock()
        mock_llm.model = "test-model"

        pipeline = NyayaRAGPipeline(retriever=mock_retriever, llm_client=mock_llm)
        resp = pipeline.ask("How can I file an RTI application?")

        # The KB unavailable message is distinct from the generic abstention message
        self.assertNotEqual(resp.answer, ABSTENTION_MESSAGE)
        self.assertIn("unavailable", resp.answer.lower())


# ─────────────────────────────────────────────────
# 2: ANSWERABLE with valid RTI context
# ─────────────────────────────────────────────────
class TestAnswerablePath(unittest.TestCase):

    def test_rti_question_with_good_score_returns_answerable(self):
        """Test 2: RTI question with score >= threshold → ANSWERABLE, no abstention."""
        results = [_rti_result(score=0.74)]
        pipeline, mock_retriever, mock_llm = _build_pipeline(results)
        mock_llm.generate.return_value = (
            '{"answer": "To file an RTI application, submit a written request to the PIO. '
            '[SOURCE 1]", "follow_up_questions": ["What is the RTI fee?", '
            '"Who is the PIO?", "What is the RTI appeal process?"]}'
        )

        resp = pipeline.ask("How can I file an RTI application?")

        self.assertFalse(resp.is_abstention)
        self.assertEqual(resp.query_result_state, QueryResultState.ANSWERABLE)
        self.assertGreater(resp.top_score, 0.0)
        self.assertEqual(mock_retriever.retrieve.call_count, 1)  # single pass sufficient

    def test_consumer_question_with_good_score_returns_answerable(self):
        """Test 3 (full path): Consumer question with good score → ANSWERABLE."""
        results = [_consumer_result(score=0.70)]
        pipeline, _, mock_llm = _build_pipeline(results)
        mock_llm.generate.return_value = (
            '{"answer": "As a consumer you have rights under CPA 2019. [SOURCE 1]", '
            '"follow_up_questions": ["How to file complaint?", "What is District Commission?"]}'
        )

        resp = pipeline.ask("What are my rights as a consumer?")

        self.assertFalse(resp.is_abstention)
        self.assertEqual(resp.query_result_state, QueryResultState.ANSWERABLE)

    def test_answer_contains_citation_text(self):
        """Test 2b: ANSWERABLE response appends programmatic citations."""
        results = [_rti_result(score=0.74)]
        pipeline, _, mock_llm = _build_pipeline(results)
        mock_llm.generate.return_value = (
            '{"answer": "Submit a request to the PIO. [SOURCE 1]", '
            '"follow_up_questions": ["What is the RTI fee?"]}'
        )

        resp = pipeline.ask("How can I file an RTI application?")
        # Citation text should be appended (contains 'Sources:')
        self.assertIn("Sources:", resp.answer)


# ─────────────────────────────────────────────────
# 4: OUT_OF_DOMAIN
# ─────────────────────────────────────────────────
class TestOutOfDomain(unittest.TestCase):

    def test_weather_query_is_out_of_domain(self):
        """Test 4: Weather question → OUT_OF_DOMAIN (not KB_UNAVAILABLE)."""
        results = [_rti_result(score=0.45)]
        pipeline, mock_retriever, _ = _build_pipeline(results)

        resp = pipeline.ask("What is the weather in Mumbai today?")

        self.assertTrue(resp.is_abstention)
        self.assertEqual(resp.query_result_state, QueryResultState.OUT_OF_DOMAIN)
        # Secondary pass must NOT be attempted for truly unrelated queries
        self.assertEqual(mock_retriever.retrieve.call_count, 1)

    def test_coding_query_is_out_of_domain(self):
        """Test 4b: Programming question → OUT_OF_DOMAIN."""
        results = [_rti_result(score=0.42)]
        pipeline, _, _ = _build_pipeline(results)

        resp = pipeline.ask("How do I write a Python function?")
        self.assertEqual(resp.query_result_state, QueryResultState.OUT_OF_DOMAIN)

    def test_out_of_domain_uses_standard_abstention_message(self):
        """OUT_OF_DOMAIN must use the standard abstention message."""
        results = [_rti_result(score=0.45)]
        pipeline, _, _ = _build_pipeline(results)

        resp = pipeline.ask("What is the weather today?")
        self.assertEqual(resp.answer, ABSTENTION_MESSAGE)


# ─────────────────────────────────────────────────
# 6: IN_DOMAIN_BUT_INSUFFICIENT
# ─────────────────────────────────────────────────
class TestInDomainButInsufficient(unittest.TestCase):

    def test_in_domain_insufficient_context_classified_correctly(self):
        """
        Test 6: RTI question where both primary and secondary passes fail threshold
        → IN_DOMAIN_BUT_INSUFFICIENT (not OUT_OF_DOMAIN, not KB_UNAVAILABLE).
        """
        pass1 = [_rti_result(score=0.30)]
        pass2 = [_rti_result(score=0.35)]  # both below _SECONDARY_RELEVANCE_THRESHOLD=0.40
        pipeline, mock_retriever, _ = _build_pipeline(pass1, pass2)

        resp = pipeline.ask("How can I file an RTI application?")

        self.assertTrue(resp.is_abstention)
        self.assertEqual(resp.query_result_state, QueryResultState.IN_DOMAIN_BUT_INSUFFICIENT)
        self.assertNotEqual(resp.query_result_state, QueryResultState.OUT_OF_DOMAIN)
        self.assertNotEqual(resp.query_result_state, QueryResultState.KNOWLEDGE_BASE_UNAVAILABLE)
        # Both passes must have been tried
        self.assertEqual(mock_retriever.retrieve.call_count, 2)

    def test_in_domain_insufficient_uses_abstention_message(self):
        """IN_DOMAIN_BUT_INSUFFICIENT uses the standard abstention message."""
        pass1 = [_rti_result(score=0.30)]
        pass2 = [_rti_result(score=0.35)]
        pipeline, _, _ = _build_pipeline(pass1, pass2)

        resp = pipeline.ask("How can I file an RTI application?")
        self.assertEqual(resp.answer, ABSTENTION_MESSAGE)


# ─────────────────────────────────────────────────
# 7: Secondary retrieval succeeds
# ─────────────────────────────────────────────────
class TestSecondaryRetrievalFallback(unittest.TestCase):

    def test_borderline_rti_query_answered_via_secondary_pass(self):
        """
        Test 7: RTI query with borderline primary score succeeds via secondary pass.
        """
        pass1 = [_rti_result(score=0.42)]     # below 0.50
        pass2 = [
            _rti_result(score=0.55),            # above 0.40 secondary threshold
            _rti_result(score=0.52, page=7),
        ]
        pipeline, mock_retriever, mock_llm = _build_pipeline(pass1, pass2)
        mock_llm.generate.return_value = (
            '{"answer": "RTI grounded answer. [SOURCE 1]", '
            '"follow_up_questions": ["What is the RTI fee?", "Who is the PIO?"]}'
        )

        resp = pipeline.ask("How can I file an RTI application?")

        self.assertFalse(resp.is_abstention)
        self.assertEqual(resp.query_result_state, QueryResultState.ANSWERABLE)
        self.assertEqual(mock_retriever.retrieve.call_count, 2)

    def test_secondary_pass_not_triggered_for_ood_query(self):
        """Test 7b: Out-of-domain query does NOT trigger secondary pass."""
        pass1 = [_rti_result(score=0.45)]
        pipeline, mock_retriever, _ = _build_pipeline(pass1)

        pipeline.ask("What is the weather today?")
        self.assertEqual(mock_retriever.retrieve.call_count, 1)

    def test_secondary_pass_triggered_for_in_domain_query(self):
        """Test 7c: In-domain query below threshold DOES trigger secondary pass."""
        pass1 = [_rti_result(score=0.42)]
        pass2 = [_rti_result(score=0.35)]  # secondary also fails
        pipeline, mock_retriever, _ = _build_pipeline(pass1, pass2)

        pipeline.ask("How can I file an RTI application?")
        self.assertEqual(mock_retriever.retrieve.call_count, 2)


# ─────────────────────────────────────────────────
# 8–10: Duplicate suggestion filtering
# ─────────────────────────────────────────────────
class TestDuplicateSuggestionFilter(unittest.TestCase):

    def test_exact_duplicate_suggestion_rejected(self):
        """Test 8: Exact duplicate of current question filtered from suggestions."""
        question = "How can I file an RTI application?"
        suggestions = [
            "How can I file an RTI application?",  # exact duplicate
            "What is the RTI fee?",
            "Who is the PIO?",
        ]
        filtered = _filter_duplicate_suggestions(question, suggestions)
        self.assertNotIn("How can I file an RTI application?", filtered)
        self.assertIn("What is the RTI fee?", filtered)

    def test_case_whitespace_punctuation_variant_rejected(self):
        """Test 9: Case/whitespace/punctuation variations treated as duplicates."""
        question = "How can I file an RTI application?"
        variants = [
            " how can i file an rti application? ",
            "How  Can  I  File  An  RTI  Application?",
            "how can i file an rti application",
            "HOW CAN I FILE AN RTI APPLICATION?",
        ]
        for v in variants:
            with self.subTest(variant=v):
                filtered = _filter_duplicate_suggestions(question, [v, "What is the RTI fee?"])
                # The variant must be filtered
                for s in filtered:
                    self.assertNotEqual(
                        _normalise_text(s),
                        _normalise_text(question),
                        f"Variant {v!r} should have been filtered",
                    )

    def test_useful_suggestions_remain(self):
        """Test 10: Useful non-duplicate suggestions are preserved."""
        question = "How can I file an RTI application?"
        suggestions = [
            "How can I file an RTI application?",           # duplicate → removed
            "What is the application fee for RTI?",
            "What information can I request under RTI?",
            "What is the time limit for PIO to respond?",
            "What can I do if my RTI is rejected?",
        ]
        filtered = _filter_duplicate_suggestions(question, suggestions)
        self.assertGreater(len(filtered), 0)
        self.assertNotIn("How can I file an RTI application?", filtered)
        self.assertIn("What is the application fee for RTI?", filtered)

    def test_empty_suggestions_filtered(self):
        """Empty strings must be removed."""
        filtered = _filter_duplicate_suggestions("test", ["", "  ", "Valid?"])
        self.assertEqual(filtered, ["Valid?"])

    def test_intra_list_duplicates_removed(self):
        """Duplicate entries within the suggestion list itself are removed."""
        filtered = _filter_duplicate_suggestions("other", ["Same?", "Same?", "Different?"])
        self.assertEqual(filtered.count("Same?"), 1)

    def test_no_false_positives(self):
        """Non-duplicate suggestions are never filtered."""
        question = "How can I file an RTI application?"
        suggestions = ["What is the RTI fee?", "Who is the PIO?", "What are RTI Rules 2012?"]
        filtered = _filter_duplicate_suggestions(question, suggestions)
        self.assertEqual(len(filtered), 3)

    def test_pipeline_filters_llm_duplicate_suggestion(self):
        """Test 8 (full pipeline): Pipeline strips LLM suggestion that echoes the question."""
        results = [_rti_result(score=0.75)]
        pipeline, _, mock_llm = _build_pipeline(results)
        mock_llm.generate.return_value = (
            '{"answer": "RTI answer. [SOURCE 1]", "follow_up_questions": '
            '["How can I file an RTI application?", "What is the RTI fee?", "Who is the PIO?"]}'
        )

        resp = pipeline.ask("How can I file an RTI application?")
        self.assertNotIn("How can I file an RTI application?", resp.follow_up_questions)
        self.assertIn("What is the RTI fee?", resp.follow_up_questions)

    def test_pipeline_filters_case_variant_suggestion(self):
        """Test 9 (pipeline): Case variant of current question filtered from LLM output."""
        results = [_rti_result(score=0.75)]
        pipeline, _, mock_llm = _build_pipeline(results)
        mock_llm.generate.return_value = (
            '{"answer": "RTI answer. [SOURCE 1]", "follow_up_questions": '
            '[" how can i file an rti application? ", "What is the RTI fee?"]}'
        )

        resp = pipeline.ask("How can I file an RTI application?")
        for s in resp.follow_up_questions:
            self.assertNotEqual(
                _normalise_text(s),
                _normalise_text("How can I file an RTI application?"),
            )


# ─────────────────────────────────────────────────
# Normalise utility
# ─────────────────────────────────────────────────
class TestNormaliseText(unittest.TestCase):

    def test_lowercase(self):
        self.assertEqual(_normalise_text("HOW CAN I FILE AN RTI?"), "how can i file an rti")

    def test_trim(self):
        self.assertEqual(_normalise_text("  hello  "), "hello")

    def test_collapse_whitespace(self):
        self.assertEqual(_normalise_text("how  can   i  file"), "how can i file")

    def test_strip_trailing_punctuation(self):
        self.assertEqual(_normalise_text("how can i file an rti?"), "how can i file an rti")
        self.assertEqual(_normalise_text("hello."), "hello")
        self.assertEqual(_normalise_text("hello!"), "hello")
        self.assertEqual(_normalise_text("hello,"), "hello")

    def test_combined_equivalence(self):
        a = _normalise_text("How can I file an RTI application?")
        b = _normalise_text(" how can i file an rti application? ")
        self.assertEqual(a, b)


# ─────────────────────────────────────────────────
# Threshold constants
# ─────────────────────────────────────────────────
class TestThresholdConstants(unittest.TestCase):

    def test_secondary_threshold_lower_than_primary(self):
        self.assertLess(_SECONDARY_RELEVANCE_THRESHOLD, MIN_RELEVANCE_THRESHOLD)

    def test_secondary_threshold_in_reasonable_range(self):
        self.assertGreaterEqual(_SECONDARY_RELEVANCE_THRESHOLD, 0.35)
        self.assertLess(_SECONDARY_RELEVANCE_THRESHOLD, 0.50)


# ─────────────────────────────────────────────────
# QueryResultState enum
# ─────────────────────────────────────────────────
class TestQueryResultStateEnum(unittest.TestCase):

    def test_all_states_distinct(self):
        states = [
            QueryResultState.ANSWERABLE,
            QueryResultState.OUT_OF_DOMAIN,
            QueryResultState.IN_DOMAIN_BUT_INSUFFICIENT,
            QueryResultState.KNOWLEDGE_BASE_UNAVAILABLE,
        ]
        self.assertEqual(len(set(states)), 4)

    def test_kb_unavailable_not_equal_out_of_domain(self):
        self.assertNotEqual(
            QueryResultState.KNOWLEDGE_BASE_UNAVAILABLE,
            QueryResultState.OUT_OF_DOMAIN
        )

    def test_state_is_string_enum(self):
        self.assertIsInstance(QueryResultState.ANSWERABLE, str)


# ─────────────────────────────────────────────────
# Production simulation: Part 9 of the specification
# ─────────────────────────────────────────────────
class TestProductionSimulation(unittest.TestCase):

    def test_part9_empty_faiss_rti_question_gives_kb_unavailable(self):
        """
        Part 9 scenario A: FAISS unavailable + RTI question
        Expected: KNOWLEDGE_BASE_UNAVAILABLE (not OUT_OF_DOMAIN)
        """
        mock_retriever = MagicMock()
        mock_vs = MagicMock()
        mock_vs.total_vectors = 0
        mock_retriever.vector_store = mock_vs

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        pipeline = NyayaRAGPipeline(retriever=mock_retriever, llm_client=mock_llm)

        resp = pipeline.ask("How can I file an RTI application?")
        self.assertTrue(resp.is_abstention)
        self.assertEqual(resp.query_result_state, QueryResultState.KNOWLEDGE_BASE_UNAVAILABLE)

    def test_part9_populated_faiss_rti_question_gives_answerable(self):
        """
        Part 9 scenario B: FAISS available + RTI question with sufficient context
        Expected: ANSWERABLE
        """
        results = [_rti_result(score=0.74)]
        pipeline, _, mock_llm = _build_pipeline(results)
        mock_llm.generate.return_value = (
            '{"answer": "To file an RTI: submit to PIO. [SOURCE 1]", '
            '"follow_up_questions": ["What is the RTI fee?"]}'
        )

        resp = pipeline.ask("How can I file an RTI application?")
        self.assertFalse(resp.is_abstention)
        self.assertEqual(resp.query_result_state, QueryResultState.ANSWERABLE)

    def test_part9_weather_question_gives_out_of_domain(self):
        """
        Part 9 scenario C: Weather question
        Expected: OUT_OF_DOMAIN
        """
        results = [_rti_result(score=0.45)]
        pipeline, _, _ = _build_pipeline(results)

        resp = pipeline.ask("What is the weather in Mumbai today?")
        self.assertTrue(resp.is_abstention)
        self.assertEqual(resp.query_result_state, QueryResultState.OUT_OF_DOMAIN)


if __name__ == "__main__":
    unittest.main(verbosity=2)
