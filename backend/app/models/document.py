from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class PageContent(BaseModel):
    """Represents the extracted text of a single PDF page."""
    page_number: int = Field(..., description="1-indexed page number in the source PDF")
    text: str = Field(..., description="Cleaned extracted text of the page")
    char_count: int = Field(default=0, description="Number of characters in the page text")
    is_empty: bool = Field(default=False, description="True if no extractable text was found")


class ParsedDocument(BaseModel):
    """Represents a fully parsed document containing page-by-page text and metadata."""
    filename: str = Field(..., description="PDF filename, e.g. RTI_Act_2005.pdf")
    category: str = Field(..., description="RTI or CONSUMER")
    title: str = Field(..., description="Official title of the Act/Rules")
    source: str = Field(..., description="Official issuing authority / source")
    source_url: Optional[str] = Field(default=None, description="Official source URL")
    total_pages: int = Field(default=0, description="Total number of pages in the PDF")
    pages: List[PageContent] = Field(default_factory=list, description="List of page contents")
    total_chars: int = Field(default=0, description="Total characters extracted across all pages")


class DocumentChunk(BaseModel):
    """Represents a single chunk ready for RAG embeddings and retrieval."""
    chunk_id: str = Field(..., description="Deterministic unique ID, e.g. RTI_Act_2005_p6_c1")
    document: str = Field(..., description="Source PDF filename")
    category: str = Field(..., description="RTI or CONSUMER")
    page: int = Field(..., description="Page number where the chunk content starts")
    legal_reference: Optional[str] = Field(default=None, description="Detected Section / Rule / Chapter or other legal reference")
    title: str = Field(..., description="Title of the source legal document")
    source: str = Field(..., description="Issuing authority / publisher")
    source_url: Optional[str] = Field(default=None, description="Official URL")
    text: str = Field(..., description="Original verbatim text segment for retrieval")
    word_count: int = Field(default=0, description="Approximate word count of the chunk")
    chunk_index: int = Field(default=0, description="Sequential index of this chunk within the document")


class RetrievalResult(BaseModel):
    """Represents a retrieved document chunk with similarity ranking and score."""
    rank: int = Field(..., description="1-indexed rank of this search result")
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    chunk_id: str = Field(..., description="ID of the matching chunk")
    document: str = Field(..., description="Source PDF filename")
    category: str = Field(..., description="RTI or CONSUMER")
    page: int = Field(..., description="Page number in the original PDF")
    legal_reference: Optional[str] = Field(default=None, description="Legal section, rule, or chapter")
    title: str = Field(..., description="Official document title")
    source: str = Field(..., description="Issuing authority")
    source_url: Optional[str] = Field(default=None, description="Official source URL")
    text: str = Field(..., description="Original verbatim text of the chunk")


class SourceCitation(BaseModel):
    """Represents a structured source citation generated programmatically from retrieved metadata."""
    document: str = Field(..., description="Source PDF filename")
    category: str = Field(..., description="RTI or CONSUMER")
    page: int = Field(..., description="Page number in the source PDF")
    legal_reference: Optional[str] = Field(default=None, description="Section, Rule, or Chapter reference")
    title: str = Field(..., description="Official document title")
    source: str = Field(..., description="Issuing authority")
    source_url: Optional[str] = Field(default=None, description="Official source URL")
    chunk_id: str = Field(..., description="ID of the chunk cited")


class RAGResponse(BaseModel):
    """Structured response from the NyayaGuide AI RAG pipeline."""
    question: str = Field(..., description="User's original query")
    answer: str = Field(..., description="Synthesized grounded answer with source citations")
    sources: List[SourceCitation] = Field(default_factory=list, description="Programmatic citations from retrieved chunks")
    retrieval_results: List[RetrievalResult] = Field(default_factory=list, description="Full top-k retrieved chunks from FAISS")
    is_abstention: bool = Field(default=False, description="True if query was abstained due to insufficient relevance")
    model_used: Optional[str] = Field(default=None, description="LLM model identifier used for generation")
    top_score: float = Field(default=0.0, description="Highest similarity score among retrieved chunks")
    follow_up_questions: List[str] = Field(default_factory=list, description="3-4 suggested follow-up questions grounded in the context")


class IngestionSummary(BaseModel):
    """Summary statistics of the document ingestion pipeline."""
    documents_processed: int = 0
    total_pages: int = 0
    total_chunks: int = 0
    document_stats: List[Dict[str, Any]] = Field(default_factory=list)
