"""
NyayaGuide AI — Persistent SQLite Document Registry
Maintains persistent records for all documents in the knowledge base, seeds baseline Government of India documents,
and enforces SHA-256 duplicate detection across restarts.
"""
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..config import (
    REGISTRY_DB_PATH,
    TARGET_DOCUMENTS,
    METADATA_FILE,
    DOCUMENTS_DIR
)
from ..models.registry import DocumentRecord, DocumentStatus, KnowledgeBaseStats


# Baseline Document Metadata mapping for seeding with accurate SHA-256 hashes
BASELINE_RECORDS_DATA = [
    {
        "filename": "RTI_Act_2005.pdf",
        "category": "RTI",
        "title": "The Right to Information Act, 2005",
        "source": "India Code",
        "authority": "Ministry of Law and Justice, Government of India",
        "source_url": "https://cic.gov.in/sites/default/files/RTI-Act_English.pdf",
        "pages": 27,
        "chunks": 27,
        "size": 850124,
        "content_hash": "e0b70162cb414d26455b86ee566db5f918bcd2bbcdc1b24d331e1a5a3933e577"
    },
    {
        "filename": "RTI_Rules_2012.pdf",
        "category": "RTI",
        "title": "The Right to Information Rules, 2012",
        "source": "Government of India / Central Information Commission",
        "authority": "Department of Personnel and Training, Government of India",
        "source_url": "https://nationalarchives.nic.in//sites/default/files/2023-08/RTIRules2012.pdf",
        "pages": 4,
        "chunks": 4,
        "size": 138490,
        "content_hash": "8d73989088febe131299d4b9a2f17b83002a576fab15084882c8f30d3424bc20"
    },
    {
        "filename": "Consumer_Protection_Act_2019.pdf",
        "category": "CONSUMER",
        "title": "The Consumer Protection Act, 2019",
        "source": "Department of Consumer Affairs, Government of India",
        "authority": "Department of Consumer Affairs, Government of India",
        "source_url": "https://egazette.gov.in/WriteReadData/2019/210422.pdf",
        "pages": 40,
        "chunks": 40,
        "size": 1258291,
        "content_hash": "0ae7f82ab5b77d50087cd4cd2dd2dbc12755c0bdb31371fdefd042b020d50756"
    },
    {
        "filename": "Consumer_Commission_and_General_Rules_2020.pdf",
        "category": "CONSUMER",
        "title": "Consumer Protection (Consumer Disputes Redressal Commissions) Rules, 2020 and Consumer Protection (General) Rules, 2020",
        "source": "Department of Consumer Affairs, Government of India",
        "authority": "Department of Consumer Affairs, Government of India",
        "source_url": "https://egazette.gov.in/WriteReadData/2020/220555.pdf",
        "pages": 18,
        "chunks": 25,
        "size": 786432,
        "content_hash": "65b360f6a62bb219ad35742adc48ee669ecda1dd3e28de7cd05f309f9f6f39da"
    }
]


class SQLiteDocumentRegistry:
    """
    SQLite-backed Document Registry for NyayaGuide AI.
    Guarantees persistence across backend restarts, transactional updates, duplicate protection,
    and safe document deletion.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or REGISTRY_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._seed_baseline_documents()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a sqlite3 connection with Row factory enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=20.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the documents table if it does not already exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    original_file_name TEXT NOT NULL,
                    stored_file_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    authority TEXT NOT NULL,
                    source_url TEXT,
                    file_size_bytes INTEGER NOT NULL DEFAULT 0,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    indexed_at TEXT,
                    content_hash TEXT NOT NULL UNIQUE,
                    version INTEGER NOT NULL DEFAULT 1,
                    error_message TEXT,
                    is_baseline INTEGER NOT NULL DEFAULT 0
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON documents(content_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON documents(category);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON documents(status);")
            conn.commit()

    def _seed_baseline_documents(self) -> None:
        """Seed the 4 baseline Government of India documents if not already registered."""
        with self._get_connection() as conn:
            for idx, item in enumerate(BASELINE_RECORDS_DATA, 1):
                fn = item["filename"]
                doc_id = f"DOC-BASELINE-{idx:04d}"
                content_hash = item["content_hash"]
                
                # Check if already present by document_id or original_file_name or content_hash
                cursor = conn.execute(
                    "SELECT document_id FROM documents WHERE original_file_name = ? OR document_id = ? OR content_hash = ?",
                    (fn, doc_id, content_hash)
                )
                existing = cursor.fetchone()
                if not existing:
                    conn.execute("""
                        INSERT INTO documents (
                            document_id, original_file_name, stored_file_name, category,
                            title, source, authority, source_url, file_size_bytes,
                            page_count, chunk_count, status, uploaded_at, indexed_at,
                            content_hash, version, error_message, is_baseline
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    """, (
                        doc_id,
                        fn,
                        fn,
                        item["category"],
                        item["title"],
                        item["source"],
                        item["authority"],
                        item["source_url"],
                        item["size"],
                        item["pages"],
                        item["chunks"],
                        DocumentStatus.INDEXED.value,
                        "Baseline",
                        "Baseline",
                        content_hash,
                        1,
                        None,
                        1
                    ))
            conn.commit()

    def generate_document_id(self) -> str:
        """Generates a standard sequential document ID, e.g. DOC-20260823-0005."""
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        prefix = f"DOC-{today_str}-"
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT document_id FROM documents WHERE document_id LIKE ? ORDER BY document_id DESC LIMIT 1",
                (f"{prefix}%",)
            )
            row = cursor.fetchone()
            if row and row["document_id"]:
                try:
                    last_num = int(row["document_id"].split("-")[-1])
                    new_num = last_num + 1
                except ValueError:
                    new_num = 1
            else:
                new_num = 1
            return f"{prefix}{new_num:04d}"

    def get_by_id(self, document_id: str) -> Optional[DocumentRecord]:
        """Fetch a document record by its unique document ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_record(row)
        return None

    def get_by_hash(self, content_hash: str) -> Optional[DocumentRecord]:
        """Fetch a document record by its SHA-256 hash to detect duplicates."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM documents WHERE content_hash = ?", (content_hash,))
            row = cursor.fetchone()
            if row:
                return self._row_to_record(row)
        return None

    def get_by_filename(self, filename: str) -> Optional[DocumentRecord]:
        """Fetch a document record by its original or stored filename (case-insensitive)."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE LOWER(original_file_name) = LOWER(?) OR LOWER(stored_file_name) = LOWER(?)",
                (filename, filename)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_record(row)
        return None

    def list_documents(
        self,
        category: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[DocumentRecord]:
        """Returns all documents in the registry with optional filtering."""
        query = "SELECT * FROM documents WHERE 1=1"
        params = []
        if category:
            query += " AND UPPER(category) = UPPER(?)"
            params.append(category)
        if status:
            query += " AND UPPER(status) = UPPER(?)"
            params.append(status)
        
        query += " ORDER BY is_baseline DESC, rowid ASC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def create_document(self, record: DocumentRecord) -> DocumentRecord:
        """Insert a new document record in the SQLite database."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO documents (
                    document_id, original_file_name, stored_file_name, category,
                    title, source, authority, source_url, file_size_bytes,
                    page_count, chunk_count, status, uploaded_at, indexed_at,
                    content_hash, version, error_message, is_baseline
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                record.document_id,
                record.original_file_name,
                record.stored_file_name,
                record.category,
                record.title,
                record.source,
                record.authority,
                record.source_url,
                record.file_size_bytes,
                record.page_count,
                record.chunk_count,
                record.status.value,
                record.uploaded_at,
                record.indexed_at,
                record.content_hash,
                record.version,
                record.error_message,
                1 if record.is_baseline else 0
            ))
            conn.commit()
        return record

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        page_count: Optional[int] = None,
        chunk_count: Optional[int] = None,
        error_message: Optional[str] = None,
        indexed_at: Optional[str] = None
    ) -> bool:
        """Update status, counts, and timestamps for an ongoing ingestion."""
        updates = ["status = ?"]
        params: List[Any] = [status.value]

        if page_count is not None:
            updates.append("page_count = ?")
            params.append(page_count)
        if chunk_count is not None:
            updates.append("chunk_count = ?")
            params.append(chunk_count)
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        if indexed_at is not None:
            updates.append("indexed_at = ?")
            params.append(indexed_at)

        params.append(document_id)
        query = f"UPDATE documents SET {', '.join(updates)} WHERE document_id = ?"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0

    def delete_document(self, document_id: str) -> bool:
        """Deletes a document record from SQLite by document_id."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_stats(self) -> KnowledgeBaseStats:
        """Computes current aggregated statistics across all indexed documents."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_docs,
                    SUM(page_count) as total_pages,
                    SUM(chunk_count) as total_chunks,
                    SUM(file_size_bytes) as total_size,
                    MAX(indexed_at) as last_indexed
                FROM documents 
                WHERE status = 'INDEXED';
            """)
            row = cursor.fetchone()
            
            total_docs = row["total_docs"] or 0
            total_pages = row["total_pages"] or 0
            total_chunks = row["total_chunks"] or 0
            total_size = row["total_size"] or 0
            last_updated = row["last_indexed"] or datetime.now(timezone.utc).isoformat()
            if last_updated == "Baseline":
                last_updated = datetime.now(timezone.utc).strftime("%d %b %Y")

            return KnowledgeBaseStats(
                total_documents=total_docs,
                total_pages=total_pages,
                total_chunks=total_chunks,
                total_file_size_bytes=total_size,
                last_updated=str(last_updated)
            )

    def _row_to_record(self, row: sqlite3.Row) -> DocumentRecord:
        """Converts an SQLite row to a Pydantic DocumentRecord."""
        return DocumentRecord(
            document_id=row["document_id"],
            original_file_name=row["original_file_name"],
            stored_file_name=row["stored_file_name"],
            category=row["category"],
            title=row["title"],
            source=row["source"],
            authority=row["authority"],
            source_url=row["source_url"],
            file_size_bytes=row["file_size_bytes"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            status=DocumentStatus(row["status"]),
            uploaded_at=row["uploaded_at"],
            indexed_at=row["indexed_at"],
            content_hash=row["content_hash"],
            version=row["version"],
            error_message=row["error_message"],
            is_baseline=bool(row["is_baseline"])
        )
