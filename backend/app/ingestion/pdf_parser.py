from pathlib import Path
from typing import List, Optional
from ..models.document import PageContent, ParsedDocument
from .cleaner import TextCleaner
from .metadata import MetadataRegistry


class PDFParser:
    """
    Extracts text from PDF documents preserving page boundaries, page numbers,
    and metadata.
    """

    def __init__(self, metadata_registry: Optional[MetadataRegistry] = None):
        self.metadata_registry = metadata_registry or MetadataRegistry()
        self.cleaner = TextCleaner()

    def parse_pdf(self, file_path: Path, filename: Optional[str] = None) -> ParsedDocument:
        """
        Parse a PDF file page-by-page into a ParsedDocument model.
        Preserves individual page boundaries and numbers.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        doc_name = filename or file_path.name
        doc_meta = self.metadata_registry.get_metadata(doc_name)

        pages: List[PageContent] = []
        total_chars = 0

        # Primary extraction using PyMuPDF (fitz)
        try:
            import fitz
            doc = fitz.open(file_path)
            total_pages = len(doc)

            for page_idx in range(total_pages):
                page_num = page_idx + 1
                page = doc[page_idx]
                raw_text = page.get_text("text")
                cleaned_text = self.cleaner.clean_text(raw_text)
                
                is_empty = len(cleaned_text.strip()) == 0
                char_count = len(cleaned_text)
                total_chars += char_count

                pages.append(PageContent(
                    page_number=page_num,
                    text=cleaned_text,
                    char_count=char_count,
                    is_empty=is_empty
                ))

            doc.close()

        except ImportError:
            # Fallback extraction using pypdf
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)

            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                raw_text = page.extract_text() or ""
                cleaned_text = self.cleaner.clean_text(raw_text)

                is_empty = len(cleaned_text.strip()) == 0
                char_count = len(cleaned_text)
                total_chars += char_count

                pages.append(PageContent(
                    page_number=page_num,
                    text=cleaned_text,
                    char_count=char_count,
                    is_empty=is_empty
                ))

        # Check if entire document was image-only / empty (which would require OCR)
        empty_pages_count = sum(1 for p in pages if p.is_empty)
        if empty_pages_count == total_pages and total_pages > 0:
            print(f"WARNING: '{doc_name}' has 0 extractable text across all {total_pages} pages. Scanned/OCR may be required.")

        return ParsedDocument(
            filename=doc_name,
            category=doc_meta.get("category", "RTI"),
            title=doc_meta.get("title", doc_name),
            source=doc_meta.get("source", "Government of India"),
            source_url=doc_meta.get("source_url"),
            total_pages=total_pages,
            pages=pages,
            total_chars=total_chars
        )
