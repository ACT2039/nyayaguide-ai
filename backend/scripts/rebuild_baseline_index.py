"""
Script to rebuild the clean baseline FAISS index from only the 4 official documents.
Run from the project root: python backend/scripts/rebuild_baseline_index.py
"""
import sys
from pathlib import Path

# Ensure backend is on path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.main import run_ingestion
from app.retrieval.retriever import NyayaRetriever
from app.config import TARGET_DOCUMENTS, FAISS_INDEX_PATH, METADATA_STORE_PATH
import json

print("=" * 65)
print("REBUILDING CLEAN BASELINE FAISS INDEX")
print("Source: 4 official Government of India documents only")
print("=" * 65)
print(f"Target documents: {TARGET_DOCUMENTS}")
print()

# Step 1: Ingest the 4 official documents
chunks, summary = run_ingestion(use_hf=True, allow_local_fallback=True)

# Verify ONLY the 4 official documents are in the chunks
doc_names = set(c.document for c in chunks)
print()
print("=== CHUNK DOCUMENT COMPOSITION ===")
from collections import Counter
doc_counts = Counter(c.document for c in chunks)
cat_counts = Counter(c.category for c in chunks)
for doc, count in sorted(doc_counts.items(), key=lambda x: -x[1]):
    cat = next((c.category for c in chunks if c.document == doc), "?")
    print(f"  {count:4d} chunks  [{cat}]  {doc}")
print()
print(f"Categories: {dict(cat_counts)}")

# Verify no test documents crept in
unexpected = doc_names - set(TARGET_DOCUMENTS)
if unexpected:
    print(f"\nERROR: Unexpected documents found: {unexpected}")
    sys.exit(1)
else:
    print(f"\nOK: Only official baseline documents present.")

# Step 2: Build FAISS index from clean chunks
print()
print("=== BUILDING FAISS VECTOR INDEX ===")
retriever = NyayaRetriever()
stats = retriever.build_index(chunks=chunks, save_to_disk=True)

# Step 3: Verify index
print()
print("=== VERIFICATION ===")
print(f"FAISS index path:     {FAISS_INDEX_PATH}")
print(f"Metadata path:        {METADATA_STORE_PATH}")
print(f"FAISS vector count:   {stats['faiss_vectors']}")
print(f"Embedding dimension:  {stats['embedding_dimension']}")
print(f"Chunks embedded:      {stats['embeddings_generated']}")
print(f"Unique documents:     {stats['documents']}")
print(f"Total chunks:         {stats['chunks']}")

# Step 4: Verify dimension=384 and IndexFlatIP
import faiss
index = faiss.read_index(str(FAISS_INDEX_PATH))
print(f"Index type:           {type(index).__name__}")
print(f"Index dimension:      {index.d}")
assert index.d == 384, f"ERROR: Expected dimension 384, got {index.d}"
assert isinstance(index, faiss.IndexFlatIP), f"ERROR: Expected IndexFlatIP, got {type(index)}"
print("OK: Index type is IndexFlatIP with dimension=384")

# Step 5: Verify metadata
with open(METADATA_STORE_PATH, "r", encoding="utf-8") as f:
    meta_data = json.load(f)
meta_records = meta_data.get("metadata", [])
meta_docs = set(m.get("document", "") for m in meta_records)
meta_cats = Counter(m.get("category", "") for m in meta_records)
print()
print("=== METADATA VERIFICATION ===")
print(f"Metadata records:     {len(meta_records)}")
print(f"Documents in metadata: {sorted(meta_docs)}")
print(f"Category distribution: {dict(meta_cats)}")

# Final check: no test docs in metadata
meta_unexpected = meta_docs - set(TARGET_DOCUMENTS)
if meta_unexpected:
    print(f"\nERROR: Unexpected documents in metadata: {meta_unexpected}")
    sys.exit(1)
else:
    print("\nOK: Metadata contains only official baseline documents.")

print()
print("=" * 65)
print("CLEAN BASELINE INDEX REBUILD COMPLETE")
print("=" * 65)
