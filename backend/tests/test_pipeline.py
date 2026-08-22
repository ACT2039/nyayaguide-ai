"""
Unit and Integration Tests for NyayaGuide AI Document Ingestion Pipeline.
Tests:
1. Metadata registry & document mapping
2. Text cleaner (whitespace, unicode, legal preservation)
3. Legal reference extraction (Sections, Rules, Chapters, etc.)
4. PDF page-aware parser & page boundaries
5. Document chunker & legal_reference metadata
"""
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import TARGET_DOCUMENTS, LOCAL_DOC_DIRS
from app.ingestion.metadata import MetadataRegistry
from app.ingestion.cleaner import TextCleaner
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import LegalDocumentChunker, extract_legal_reference
from app.models.document import PageContent, ParsedDocument, DocumentChunk


class TestMetadataRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = MetadataRegistry()

    def test_all_target_documents_registered(self):
        for doc in TARGET_DOCUMENTS:
            meta = self.registry.get_metadata(doc)
            self.assertIsNotNone(meta)
            self.assertIn("category", meta)
            self.assertIn(meta["category"], ["RTI", "CONSUMER"])
            self.assertIn("title", meta)
            self.assertIn("source", meta)

    def test_category_mapping(self):
        self.assertEqual(self.registry.get_category("RTI_Act_2005.pdf"), "RTI")
        self.assertEqual(self.registry.get_category("RTI_Rules_2012.pdf"), "RTI")
        self.assertEqual(self.registry.get_category("Consumer_Protection_Act_2019.pdf"), "CONSUMER")
        self.assertEqual(self.registry.get_category("Consumer_Commission_and_General_Rules_2020.pdf"), "CONSUMER")


class TestTextCleaner(unittest.TestCase):
    def test_whitespace_and_unicode_cleaning(self):
        raw = "Section   6.   Request  for\u00a0obtaining\u200b information.\n\n\n\n(1) A person..."
        cleaned = TextCleaner.clean_text(raw)
        self.assertNotIn("\u00a0", cleaned)
        self.assertNotIn("\u200b", cleaned)
        self.assertIn("Section 6. Request for obtaining information.", cleaned)
        self.assertIn("(1) A person...", cleaned)

    def test_legal_verbatim_preservation(self):
        legal_text = (
            "Section 4(1)(b) Every public authority shall publish within one hundred "
            "and twenty days from the enactment of this Act..."
        )
        cleaned = TextCleaner.clean_text(legal_text)
        self.assertEqual(cleaned, legal_text)


class TestLegalReferenceDetection(unittest.TestCase):
    def test_section_references(self):
        # Test Section 35
        text_35 = "Under Section 35 of the Consumer Protection Act, a complaint, in relation to any goods sold..."
        self.assertEqual(extract_legal_reference(text_35), "Section 35")

        # Test Section 6
        text_6 = "Under Section 6(1) of the Act, a person who desires to obtain any information..."
        self.assertEqual(extract_legal_reference(text_6), "Section 6(1)")

    def test_rule_references(self):
        # Test Rule 3(i)
        text_rule_i = "In accordance with Rule 3(i) of the Consumer Protection Rules, the commission shall..."
        self.assertEqual(extract_legal_reference(text_rule_i), "Rule 3(i)")

        # Test Rule 3
        text_rule = "Rule 3. Application Fee.— An application for obtaining information under sub-section (1)..."
        self.assertEqual(extract_legal_reference(text_rule), "Rule 3")

    def test_chapter_references(self):
        # Test Chapter headings
        text_chap = "CHAPTER IV\nCONSUMER DISPUTES REDRESSAL COMMISSION\nEstablishment of District Consumer Disputes Redressal Commission."
        ref = extract_legal_reference(text_chap)
        self.assertIsNotNone(ref)
        self.assertTrue("CHAPTER IV" in ref)

    def test_bare_act_heading_detection(self):
        text = "6. Request for obtaining information.— (1) A person who desires..."
        ref = extract_legal_reference(text)
        self.assertIsNotNone(ref)
        self.assertTrue("Section 6" in ref)

    def test_no_false_positive_legal_reference(self):
        text = "This is a general paragraph discussing administrative principles without any statute numbers."
        ref = extract_legal_reference(text)
        self.assertIsNone(ref)


class TestPDFParserAndChunker(unittest.TestCase):
    def setUp(self):
        self.parser = PDFParser()
        self.chunker = LegalDocumentChunker()

    def test_parser_preserves_pages(self):
        # Locate RTI_Act_2005.pdf
        pdf_path = None
        for folder in LOCAL_DOC_DIRS:
            candidate = folder / "RTI_Act_2005.pdf"
            if candidate.exists():
                pdf_path = candidate
                break

        if pdf_path is None:
            self.skipTest("RTI_Act_2005.pdf not found in local directories for testing.")

        parsed = self.parser.parse_pdf(pdf_path, filename="RTI_Act_2005.pdf")
        self.assertGreater(parsed.total_pages, 0)
        self.assertEqual(len(parsed.pages), parsed.total_pages)
        self.assertEqual(parsed.category, "RTI")
        self.assertGreater(parsed.total_chars, 1000)

        # Check page numbers are 1-indexed and contiguous
        for idx, page in enumerate(parsed.pages, 1):
            self.assertEqual(page.page_number, idx)

    def test_chunking_creates_valid_legal_reference_metadata(self):
        mock_pages = [
            PageContent(
                page_number=1,
                text="CHAPTER I\nPRELIMINARY\n\nSection 1. Short title, extent and commencement.\n(1) This Act may be called the Right to Information Act, 2005.",
                char_count=150,
                is_empty=False
            ),
            PageContent(
                page_number=2,
                text="Section 35. Manner in which complaint shall be made.— A complaint may be filed with a District Commission by...",
                char_count=120,
                is_empty=False
            ),
            PageContent(
                page_number=3,
                text="Rule 3(i). Manner of payment of fee.— The fee may be paid in cash against proper receipt or by demand draft...",
                char_count=110,
                is_empty=False
            )
        ]
        doc = ParsedDocument(
            filename="Mock_Act.pdf",
            category="CONSUMER",
            title="Mock Legal Document",
            source="Department of Consumer Affairs",
            source_url="https://egazette.gov.in",
            total_pages=3,
            pages=mock_pages,
            total_chars=380
        )

        chunks = self.chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 3)
        
        # Check chunk 1 legal reference
        self.assertIn("Section 1", chunks[0].legal_reference or chunks[0].text)
        self.assertEqual(chunks[0].category, "CONSUMER")

        # Check chunk 2: Section 35
        self.assertEqual(chunks[1].legal_reference, "Section 35")
        self.assertEqual(chunks[1].page, 2)

        # Check chunk 3: Rule 3(i)
        self.assertEqual(chunks[2].legal_reference, "Rule 3(i)")
        self.assertEqual(chunks[2].page, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
