"""
NyayaGuide AI — RAG Context Builder & Source Citation Generator
Formats retrieved legal chunks and generates programmatic citations without LLM hallucination.
"""
from typing import List, Tuple
from ..models.document import RetrievalResult, SourceCitation


class ContextBuilder:
    """
    Assembles retrieved legal document chunks into structured grounding context
    and generates verifiable source citations.
    """

    @staticmethod
    def build_context_string(results: List[RetrievalResult]) -> str:
        """
        Formats retrieved legal chunks into standard grounding context.
        """
        if not results:
            return "No relevant source material retrieved."

        context_blocks = []
        for idx, r in enumerate(results, 1):
            legal_ref = r.legal_reference if r.legal_reference else "N/A"
            block = (
                f"[SOURCE {idx}]\n"
                f"Document: {r.document}\n"
                f"Category: {r.category}\n"
                f"Page: {r.page}\n"
                f"Legal Reference: {legal_ref}\n"
                f"Title: {r.title}\n"
                f"Source: {r.source}\n\n"
                f"Text:\n{r.text}"
            )
            context_blocks.append(block)

        return "\n\n" + ("=" * 50) + "\n\n".join([""] + context_blocks) + "\n\n" + ("=" * 50)

    @staticmethod
    def extract_programmatic_citations(results: List[RetrievalResult]) -> List[SourceCitation]:
        """
        Extracts clean, deduplicated structured source citations directly from retrieved metadata.
        Guarantees that document names, pages, and legal references are not invented.
        """
        citations: List[SourceCitation] = []
        seen_keys = set()

        for r in results:
            # Deduplicate by document, page, and legal reference
            key = (r.document, r.page, r.legal_reference)
            if key not in seen_keys:
                seen_keys.add(key)
                citations.append(SourceCitation(
                    document=r.document,
                    category=r.category,
                    page=r.page,
                    legal_reference=r.legal_reference,
                    title=r.title,
                    source=r.source,
                    source_url=r.source_url,
                    chunk_id=r.chunk_id
                ))

        return citations

    @staticmethod
    def format_citation_text(citations: List[SourceCitation]) -> str:
        """
        Formats a clean, standardized human-readable Sources block.
        """
        if not citations:
            return ""

        lines = ["\n\nSources:\n"]
        for idx, c in enumerate(citations, 1):
            ref_str = f"   {c.legal_reference}\n" if c.legal_reference else ""
            lines.append(
                f"{idx}. {c.title}\n"
                f"{ref_str}"
                f"   Document: {c.document} (Page {c.page})\n"
                f"   Source: {c.source}\n"
            )

        return "\n".join(lines).strip()
