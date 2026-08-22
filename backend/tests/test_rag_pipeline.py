"""
Unit Tests for RAG Context Builder, Prompt Construction, and RAG Pipeline
"""
import sys
import unittest
from unittest.mock import MagicMock
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.models.document import RetrievalResult, RAGResponse
from app.rag.context_builder import ContextBuilder
from app.rag.prompt import SYSTEM_PROMPT, build_rag_messages
from app.rag.rag_pipeline import NyayaRAGPipeline
from app.config import ABSTENTION_MESSAGE


class TestRAGContextAndPrompts(unittest.TestCase):
    def setUp(self):
        self.mock_results = [
            RetrievalResult(
                rank=1,
                score=0.85,
                chunk_id="RTI_p6_c1",
                document="RTI_Act_2005.pdf",
                category="RTI",
                page=6,
                legal_reference="Section 6",
                title="The Right to Information Act, 2005",
                source="India Code",
                source_url="https://cic.gov.in",
                text="A person who desires to obtain any information shall make a request in writing."
            ),
            RetrievalResult(
                rank=2,
                score=0.78,
                chunk_id="RTI_Rules_p2_c1",
                document="RTI_Rules_2012.pdf",
                category="RTI",
                page=2,
                legal_reference="Rule 3",
                title="The Right to Information Rules, 2012",
                source="Government of India",
                source_url="https://nationalarchives.nic.in",
                text="Application Fee: An application for obtaining information shall be accompanied by a fee of rupees ten."
            )
        ]

    def test_context_builder_formatting(self):
        context_str = ContextBuilder.build_context_string(self.mock_results)
        self.assertIn("[SOURCE 1]", context_str)
        self.assertIn("Document: RTI_Act_2005.pdf", context_str)
        self.assertIn("Page: 6", context_str)
        self.assertIn("Legal Reference: Section 6", context_str)
        self.assertIn("[SOURCE 2]", context_str)
        self.assertIn("Rule 3", context_str)
        self.assertIn("rupees ten", context_str)

    def test_programmatic_citations(self):
        citations = ContextBuilder.extract_programmatic_citations(self.mock_results)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0].document, "RTI_Act_2005.pdf")
        self.assertEqual(citations[0].page, 6)
        self.assertEqual(citations[0].legal_reference, "Section 6")
        self.assertEqual(citations[1].legal_reference, "Rule 3")

    def test_citation_text_formatting(self):
        citations = ContextBuilder.extract_programmatic_citations(self.mock_results)
        citation_text = ContextBuilder.format_citation_text(citations)
        self.assertIn("Sources:", citation_text)
        self.assertIn("The Right to Information Act, 2005", citation_text)
        self.assertIn("Section 6", citation_text)
        self.assertIn("Page 6", citation_text)

    def test_system_prompt_rules_presence(self):
        self.assertIn("Do not invent legal facts", SYSTEM_PROMPT)
        self.assertIn("Do not invent sections or rules", SYSTEM_PROMPT)
        self.assertIn("Do not claim to be a lawyer", SYSTEM_PROMPT)
        self.assertIn("insufficient", SYSTEM_PROMPT.lower())

    def test_rag_pipeline_with_mock_llm(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = self.mock_results

        mock_llm = MagicMock()
        mock_llm.model = "google/gemini-2.0-flash-001"
        mock_llm.generate.return_value = "To file an RTI application, submit a written request with Rs 10 fee."

        pipeline = NyayaRAGPipeline(
            retriever=mock_retriever,
            llm_client=mock_llm,
            min_relevance_threshold=0.50
        )

        response = pipeline.ask("How to file an RTI application?")
        self.assertIsInstance(response, RAGResponse)
        self.assertFalse(response.is_abstention)
        self.assertIn("To file an RTI application", response.answer)
        self.assertIn("Sources:", response.answer)
        self.assertEqual(len(response.sources), 2)
        self.assertEqual(response.top_score, 0.85)

    def test_rag_pipeline_abstention_on_low_score(self):
        low_score_results = [
            RetrievalResult(
                rank=1,
                score=0.42,  # Below 0.50 threshold
                chunk_id="CPA_p1",
                document="Consumer_Protection_Act_2019.pdf",
                category="CONSUMER",
                page=1,
                legal_reference=None,
                title="Consumer Protection Act, 2019",
                source="Gov of India",
                source_url=None,
                text="Arbitrary text segment."
            )
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = low_score_results

        mock_llm = MagicMock()

        pipeline = NyayaRAGPipeline(
            retriever=mock_retriever,
            llm_client=mock_llm,
            min_relevance_threshold=0.50
        )

        response = pipeline.ask("What is the weather tomorrow?")
        self.assertTrue(response.is_abstention)
        self.assertEqual(response.answer, ABSTENTION_MESSAGE)
        self.assertEqual(len(response.sources), 0)
        self.assertEqual(mock_llm.generate.call_count, 0)  # LLM should not be called


if __name__ == "__main__":
    unittest.main(verbosity=2)
