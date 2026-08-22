export interface SourceCitation {
  document: string;
  category: string;
  page: number;
  legal_reference?: string | null;
  title: string;
  source: string;
  source_url?: string | null;
  chunk_id: string;
}

export interface AskResponse {
  question: string;
  answer: string;
  sources: SourceCitation[];
  is_abstention: boolean;
  model_used?: string | null;
  top_score: number;
  follow_up_questions?: string[];
}

export interface AskRequest {
  question: string;
}

export interface ConversationTurn {
  id: string;
  question: string;
  timestamp: string;
  response?: AskResponse;
  loading?: boolean;
  error?: string | null;
}

// ──────────────────────────────────────────────
// Phase 7: Knowledge Base Management Types
// ──────────────────────────────────────────────

export type DocumentStatus =
  | 'UPLOADING'
  | 'PROCESSING'
  | 'EMBEDDING'
  | 'INDEXING'
  | 'INDEXED'
  | 'FAILED';

export interface DocumentItem {
  document_id: string;
  original_file_name: string;
  stored_file_name: string;
  category: string;
  title: string;
  source: string;
  authority: string;
  source_url?: string | null;
  file_size_bytes: number;
  page_count: number;
  chunk_count: number;
  status: DocumentStatus;
  uploaded_at: string;
  indexed_at?: string | null;
  content_hash: string;
  version: number;
  error_message?: string | null;
  is_baseline: boolean;
}

export interface KnowledgeBaseStats {
  total_documents: number;
  total_pages: number;
  total_chunks: number;
  total_file_size_bytes: number;
  last_updated: string;
}

export interface DocumentUploadResponse {
  document_id: string;
  original_file_name: string;
  status: DocumentStatus;
  uploaded_at: string;
  message: string;
  page_count?: number | null;
  chunk_count?: number | null;
}

export interface DocumentStatusResponse {
  document_id: string;
  original_file_name: string;
  status: DocumentStatus;
  progress_stage: string;
  page_count: number;
  chunk_count: number;
  error_message?: string | null;
  indexed_at?: string | null;
}

export interface DocumentDeleteResponse {
  document_id: string;
  original_file_name: string;
  removed_chunks: number;
  message: string;
}

