import re
from typing import List, Optional
from ..models.document import ParsedDocument, DocumentChunk, PageContent
from ..config import TARGET_CHUNK_SIZE_WORDS, MAX_CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS


def extract_legal_reference(text: str) -> Optional[str]:
    """
    Detects if a text block contains a clear Section, Rule, Chapter, or other legal reference.
    Supports:
      - Sections: "Section 35", "Section 6", "Section 4(1)(b)", "Sec. 12"
      - Rules: "Rule 3", "Rule 3(i)", "Rule 4(1)", "Rule 11"
      - Chapters: "CHAPTER II", "CHAPTER IV"
      - Bare Act headings: "6. Request for obtaining information.—"
      - Orders, Schedules, Articles: "Order II", "Schedule I", "Article 21"
    Returns the detected legal reference string or None if not determinable.
    DO NOT invent legal references.
    """
    if not text:
        return None

    # Check for Section pattern: e.g. Section 35, Section 6, Section 4(1)(b), Sec. 2(d)
    match = re.search(r'\b(?:Section|Sec\.)\s+(\d+[A-Za-z]?(?:\s*\([0-9a-zA-ZivxlcdmIVXLCDM]+\))*)', text, re.IGNORECASE)
    if match:
        sec_num = re.sub(r'\s+', '', match.group(1))
        return f"Section {sec_num}"

    # Check for Rule pattern: e.g. Rule 3, Rule 3(i), Rule 4(1)
    match = re.search(r'\bRule\s+(\d+[A-Za-z]?(?:\s*\([0-9a-zA-ZivxlcdmIVXLCDM]+\))*)', text, re.IGNORECASE)
    if match:
        rule_num = re.sub(r'\s+', '', match.group(1))
        return f"Rule {rule_num}"

    # Check for Bare Act line heading e.g. "6. Request for obtaining information.—" or "35. Manner in which complaint shall be made"
    match = re.search(r'(?:^|\n)\s*(\d+[A-Za-z]?)\.\s+([A-Z][A-Za-z\s]{3,40}?)(?:—|–|-|\.|\n)', text)
    if match:
        num = match.group(1)
        heading = match.group(2).strip()
        if len(heading) > 3 and not heading.startswith("The ") and not heading.lower().startswith("is "):
            return f"Section {num} ({heading})"
        return f"Section {num}"

    # Check for Chapter marker: e.g. "CHAPTER II", "CHAPTER IV"
    match = re.search(r'\b(CHAPTER\s+[IVXLCDM\d]+(?:\s*[-–—:]\s*[A-Z\s]{3,40})?)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Check for Order / Schedule / Article
    match = re.search(r'\b(Order\s+[IVXLCDM\d]+|Schedule\s+[IVXLCDM\d]+|Article\s+\d+[A-Za-z]?)', text, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


# Backward-compatible alias
extract_legal_section_reference = extract_legal_reference


class LegalDocumentChunker:
    """
    Splits page-aware parsed legal documents into chunks suitable for RAG retrieval.
    Preserves page boundaries, legal references (sections, rules, chapters), and metadata.
    """

    def __init__(
        self,
        target_size: int = TARGET_CHUNK_SIZE_WORDS,
        max_size: int = MAX_CHUNK_SIZE_WORDS,
        overlap: int = CHUNK_OVERLAP_WORDS
    ):
        self.target_size = target_size
        self.max_size = max_size
        self.overlap = overlap

    def chunk_document(self, parsed_doc: ParsedDocument) -> List[DocumentChunk]:
        """
        Produce structured DocumentChunk list from a ParsedDocument.
        Chunking is page-aware: chunks respect page boundaries where possible.
        """
        chunks: List[DocumentChunk] = []
        chunk_idx = 0
        doc_slug = parsed_doc.filename.replace(".pdf", "").replace(" ", "_")

        for page in parsed_doc.pages:
            if page.is_empty or not page.text.strip():
                continue

            page_chunks = self._chunk_page(page, parsed_doc, doc_slug, chunk_idx)
            chunks.extend(page_chunks)
            chunk_idx += len(page_chunks)

        return chunks

    def _chunk_page(
        self,
        page: PageContent,
        doc: ParsedDocument,
        doc_slug: str,
        start_index: int
    ) -> List[DocumentChunk]:
        """
        Chunk text from a single page into overlapping windows when larger than max_size.
        """
        text = page.text.strip()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            paragraphs = [text]

        # Assemble paragraphs into chunks of target size
        current_paragraphs: List[str] = []
        current_word_count = 0
        page_chunks: List[DocumentChunk] = []
        page_chunk_num = 1

        for p in paragraphs:
            p_words = p.split()
            p_len = len(p_words)

            if current_word_count + p_len > self.max_size and current_paragraphs:
                # Flush current chunk
                chunk_text = "\n\n".join(current_paragraphs).strip()
                legal_ref = extract_legal_reference(chunk_text)
                
                chunk_id = f"{doc_slug}_p{page.page_number}_c{page_chunk_num}"
                page_chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    document=doc.filename,
                    category=doc.category,
                    page=page.page_number,
                    legal_reference=legal_ref,
                    title=doc.title,
                    source=doc.source,
                    source_url=doc.source_url,
                    text=chunk_text,
                    word_count=len(chunk_text.split()),
                    chunk_index=start_index + len(page_chunks)
                ))
                page_chunk_num += 1

                # Carry over overlap if possible
                overlap_words: List[str] = []
                for prev_p in reversed(current_paragraphs):
                    prev_words = prev_p.split()
                    if len(overlap_words) + len(prev_words) <= self.overlap:
                        overlap_words = prev_words + overlap_words
                    else:
                        break

                current_paragraphs = [" ".join(overlap_words)] if overlap_words else []
                current_word_count = len(overlap_words)

            current_paragraphs.append(p)
            current_word_count += p_len

        # Flush final remaining paragraphs for the page
        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs).strip()
            if chunk_text:
                legal_ref = extract_legal_reference(chunk_text)
                chunk_id = f"{doc_slug}_p{page.page_number}_c{page_chunk_num}"
                page_chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    document=doc.filename,
                    category=doc.category,
                    page=page.page_number,
                    legal_reference=legal_ref,
                    title=doc.title,
                    source=doc.source,
                    source_url=doc.source_url,
                    text=chunk_text,
                    word_count=len(chunk_text.split()),
                    chunk_index=start_index + len(page_chunks)
                ))

        return page_chunks
