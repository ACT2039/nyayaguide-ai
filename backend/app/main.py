import sys
from pathlib import Path
from typing import List, Tuple

# Reconfigure stdout for UTF-8 compatibility on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure backend package is on Python path
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import (
    TARGET_DOCUMENTS,
    is_hf_token_configured,
    get_safe_hf_token_status,
    HF_DATASET_REPO
)
from app.ingestion.hf_loader import HuggingFaceDatasetLoader
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import LegalDocumentChunker
from app.ingestion.metadata import MetadataRegistry
from app.models.document import ParsedDocument, DocumentChunk, IngestionSummary


def run_ingestion(use_hf: bool = True, allow_local_fallback: bool = True) -> Tuple[List[DocumentChunk], IngestionSummary]:
    """
    Executes the full document ingestion pipeline for NyayaGuide AI.
    1. Hugging Face authentication / PDF retrieval
    2. Page-aware text extraction
    3. Legal text cleaning
    4. Structural chunking & legal reference metadata attachment
    """
    print("=" * 65)
    print("NyayaGuide AI -- Document Ingestion Pipeline")
    print("=" * 65)
    print(f"Target Documents     : {len(TARGET_DOCUMENTS)} documents")
    print(f"Hugging Face Auth    : {get_safe_hf_token_status()}")
    print(f"Dataset Repository   : {HF_DATASET_REPO}")
    print("-" * 65)

    metadata_registry = MetadataRegistry()
    hf_loader = HuggingFaceDatasetLoader()
    pdf_parser = PDFParser(metadata_registry=metadata_registry)
    chunker = LegalDocumentChunker()

    all_chunks: List[DocumentChunk] = []
    doc_stats = []
    total_pages_all = 0

    # Step 1: Check Hugging Face connectivity if token configured
    if is_hf_token_configured():
        print("\n[1/4] Authenticating with Hugging Face...")
        success, message, files = hf_loader.verify_connection()
        if success:
            print(f"  [OK] {message}")
        else:
            print(f"  [NOTICE] Authentication: {message}")
            if not allow_local_fallback:
                raise PermissionError(f"Hugging Face connection failed and fallback disabled: {message}")
            print("  [INFO] Proceeding with verified local knowledge base fallback.")
    else:
        print("\n[1/4] HF_TOKEN not set -- using local knowledge base source.")

    # Step 2: Ingest each document
    print("\n[2/4] Processing source documents...")
    for idx, doc_name in enumerate(TARGET_DOCUMENTS, 1):
        print(f"\n--- Document {idx}/{len(TARGET_DOCUMENTS)}: {doc_name} ---")
        
        # Retrieve PDF
        try:
            doc_path = hf_loader.get_document_path(doc_name, allow_local_fallback=allow_local_fallback)
            print(f"  Source Path: {doc_path} ({doc_path.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"  [ERROR] Failed to obtain {doc_name}: {e}")
            raise

        # Extract page-aware text
        parsed_doc = pdf_parser.parse_pdf(doc_path, filename=doc_name)
        total_pages_all += parsed_doc.total_pages
        print(f"  Pages Extracted     : {parsed_doc.total_pages}")
        print(f"  Characters Cleaned  : {parsed_doc.total_chars:,}")
        
        # Check empty pages
        empty_count = sum(1 for p in parsed_doc.pages if p.is_empty)
        if empty_count > 0:
            print(f"  Empty Pages Notice  : {empty_count} pages had no extractable text")

        # Chunk document
        chunks = chunker.chunk_document(parsed_doc)
        print(f"  Chunks Created      : {len(chunks)}")
        
        # Legal reference detection stats
        refs_detected = sum(1 for c in chunks if c.legal_reference is not None)
        print(f"  Legal References    : {refs_detected} chunks tagged with legal references")

        all_chunks.extend(chunks)
        doc_stats.append({
            "document": doc_name,
            "category": parsed_doc.category,
            "title": parsed_doc.title,
            "pages": parsed_doc.total_pages,
            "characters": parsed_doc.total_chars,
            "chunks": len(chunks),
            "legal_references_detected": refs_detected
        })

    summary = IngestionSummary(
        documents_processed=len(TARGET_DOCUMENTS),
        total_pages=total_pages_all,
        total_chunks=len(all_chunks),
        document_stats=doc_stats
    )

    # Step 3: Print Safe Ingestion Summary
    print("\n" + "=" * 65)
    print("INGESTION PIPELINE SUMMARY")
    print("=" * 65)
    print(f"Documents processed: {summary.documents_processed}")
    print(f"Total Pages        : {summary.total_pages}")
    print(f"Total Chunks       : {summary.total_chunks}")
    print("-" * 65)

    for ds in summary.document_stats:
        print(f"Document           : {ds['document']}")
        print(f"Category           : {ds['category']}")
        print(f"Pages              : {ds['pages']}")
        print(f"Characters extracted: {ds['characters']:,}")
        print(f"Number of chunks   : {ds['chunks']}")
        print(f"Legal References   : {ds['legal_references_detected']} tagged")
        print("-" * 65)

    # Step 4: Display 3 Sample Chunks
    print("\nSAMPLE CHUNKS (3 Examples)")
    print("=" * 65)
    
    sample_indices = [0, len(all_chunks) // 2, len(all_chunks) - 1]
    for i, s_idx in enumerate(sample_indices, 1):
        if s_idx < len(all_chunks):
            chunk = all_chunks[s_idx]
            preview = chunk.text[:220].replace("\n", " ") + "..."
            print(f"\n[Sample Chunk #{i}]")
            print(f"  Chunk ID        : {chunk.chunk_id}")
            print(f"  Document        : {chunk.document}")
            print(f"  Category        : {chunk.category}")
            print(f"  Page            : {chunk.page}")
            print(f"  Legal Reference : {chunk.legal_reference or 'General Context'}")
            print(f"  Source          : {chunk.source}")
            print(f"  Word Count      : {chunk.word_count}")
            print(f"  Text Preview    : \"{preview}\"")

    print("\n" + "=" * 65)
    print("[SUCCESS] Ingestion pipeline completed successfully and verified.")
    print("=" * 65)

    return all_chunks, summary


if __name__ == "__main__":
    run_ingestion()
