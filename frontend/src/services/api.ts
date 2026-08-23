import {
  AskResponse,
  AskRequest,
  DocumentItem,
  KnowledgeBaseStats,
  DocumentUploadResponse,
  DocumentStatusResponse,
  DocumentDeleteResponse
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

// Admin key storage in sessionStorage for admin session management
const ADMIN_KEY_STORAGE_KEY = 'nyayaguide_admin_key';

export function getStoredAdminKey(): string {
  return sessionStorage.getItem(ADMIN_KEY_STORAGE_KEY) || 'nyayaguide_admin_secret_2026';
}

export function setStoredAdminKey(key: string): void {
  const trimmed = key.trim();
  if (trimmed) {
    sessionStorage.setItem(ADMIN_KEY_STORAGE_KEY, trimmed);
  } else {
    sessionStorage.removeItem(ADMIN_KEY_STORAGE_KEY);
  }
}

export function hasCustomAdminKey(): boolean {
  return !!sessionStorage.getItem(ADMIN_KEY_STORAGE_KEY);
}

export function clearStoredAdminKey(): void {
  sessionStorage.removeItem(ADMIN_KEY_STORAGE_KEY);
}

export async function askQuestion(question: string): Promise<AskResponse> {
  const url = `${API_BASE_URL}/api/ask`;
  const payload: AskRequest = { question: question.trim() };

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const detail = errorData.detail || `Server returned error status ${response.status}`;
      throw new Error(detail);
    }

    const data: AskResponse = await response.json();
    return data;
  } catch (err: any) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new Error('NyayaGuide is temporarily unable to connect to the server. Please verify the backend is running.');
    }
    throw err;
  }
}

export async function checkHealth(): Promise<{ status: string }> {
  const url = `${API_BASE_URL}/health`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error('Health check failed');
  }
  return response.json();
}

// ──────────────────────────────────────────────
// Phase 7: Document Management APIs
// ──────────────────────────────────────────────

export async function fetchDocuments(
  category?: string,
  status?: string,
  adminKey?: string
): Promise<DocumentItem[]> {
  const params = new URLSearchParams();
  if (category && category !== 'ALL') params.append('category', category);
  if (status && status !== 'ALL') params.append('doc_status', status);

  const queryString = params.toString() ? `?${params.toString()}` : '';
  const url = `${API_BASE_URL}/api/documents${queryString}`;

  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
      'X-Admin-Key': adminKey || getStoredAdminKey()
    }
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch documents (${response.status})`);
  }

  return response.json();
}

export async function fetchStats(adminKey?: string): Promise<KnowledgeBaseStats> {
  const url = `${API_BASE_URL}/api/documents/stats`;
  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
      'X-Admin-Key': adminKey || getStoredAdminKey()
    }
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch statistics (${response.status})`);
  }

  return response.json();
}

export async function fetchDocumentDetails(
  documentId: string,
  adminKey?: string
): Promise<DocumentItem> {
  const url = `${API_BASE_URL}/api/documents/${encodeURIComponent(documentId)}`;
  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
      'X-Admin-Key': adminKey || getStoredAdminKey()
    }
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch document details (${response.status})`);
  }

  return response.json();
}

export async function fetchDocumentStatus(
  documentId: string,
  adminKey?: string
): Promise<DocumentStatusResponse> {
  const url = `${API_BASE_URL}/api/documents/${encodeURIComponent(documentId)}/status`;
  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
      'X-Admin-Key': adminKey || getStoredAdminKey()
    }
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch document status (${response.status})`);
  }

  return response.json();
}

export interface UploadMetadata {
  category: string;
  title?: string;
  source?: string;
  authority?: string;
  source_url?: string;
}

export async function uploadDocument(
  file: File,
  metadata: UploadMetadata,
  adminKey?: string
): Promise<DocumentUploadResponse> {
  const url = `${API_BASE_URL}/api/documents/upload`;
  const formData = new FormData();
  formData.append('file', file);
  formData.append('category', metadata.category);
  if (metadata.title) formData.append('title', metadata.title);
  if (metadata.source) formData.append('source', metadata.source);
  if (metadata.authority) formData.append('authority', metadata.authority);
  if (metadata.source_url) formData.append('source_url', metadata.source_url);

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'X-Admin-Key': adminKey || getStoredAdminKey()
    },
    body: formData
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Upload failed with status ${response.status}`);
  }

  return response.json();
}

export async function deleteDocument(
  documentId: string,
  adminKey?: string
): Promise<DocumentDeleteResponse> {
  const url = `${API_BASE_URL}/api/documents/${encodeURIComponent(documentId)}`;
  const response = await fetch(url, {
    method: 'DELETE',
    headers: {
      'Accept': 'application/json',
      'X-Admin-Key': adminKey || getStoredAdminKey()
    }
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to delete document (${response.status})`);
  }

  return response.json();
}

