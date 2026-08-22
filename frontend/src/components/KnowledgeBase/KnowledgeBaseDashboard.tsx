import React, { useState, useEffect } from 'react';
import {
  BookOpen,
  FileText,
  Layers,
  RefreshCw,
  Plus,
  ShieldCheck,
  Calendar,
  CheckCircle2
} from 'lucide-react';
import { DocumentItem, KnowledgeBaseStats } from '../../types/api';
import { fetchDocuments, fetchStats, deleteDocument } from '../../services/api';
import { DocumentTable } from './DocumentTable';
import { DocumentDetailsModal } from './DocumentDetailsModal';
import { UploadDocumentModal } from './UploadDocumentModal';
import { DeleteConfirmModal } from './DeleteConfirmModal';

export const KnowledgeBaseDashboard: React.FC = () => {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [stats, setStats] = useState<KnowledgeBaseStats | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [successToast, setSuccessToast] = useState<string | null>(null);

  const [selectedDoc, setSelectedDoc] = useState<DocumentItem | null>(null);
  const [deleteTargetDoc, setDeleteTargetDoc] = useState<DocumentItem | null>(null);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState<boolean>(false);
  const [isDeleting, setIsDeleting] = useState<boolean>(false);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [docsData, statsData] = await Promise.all([
        fetchDocuments(),
        fetchStats()
      ]);
      setDocuments(docsData);
      setStats(statsData);
    } catch (err: any) {
      setError(err.message || 'Failed to load Knowledge Base data.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleDeleteConfirm = async (doc: DocumentItem) => {
    setIsDeleting(true);
    setError(null);
    try {
      const res = await deleteDocument(doc.document_id);
      setDeleteTargetDoc(null);
      setSuccessToast(res.message || `Document '${doc.original_file_name}' deleted successfully.`);
      
      // Auto-hide toast after 4 seconds
      setTimeout(() => {
        setSuccessToast(null);
      }, 4000);

      // Refresh data
      await loadData();
    } catch (err: any) {
      setError(err.message || 'Failed to delete document.');
    } finally {
      setIsDeleting(false);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="kb-dashboard-layout">
      {/* Top Header Row */}
      <div className="kb-header-row">
        <div>
          <h2 className="kb-section-title">Knowledge Base Management</h2>
          <p className="kb-section-desc">
            Repository of official Government of India bare acts, statutory rules, and gazettes indexed for RAG retrieval.
          </p>
        </div>
        <div className="kb-header-actions">
          <button
            className="btn btn-secondary btn-icon-only"
            onClick={loadData}
            title="Refresh Knowledge Base"
            disabled={loading}
          >
            <RefreshCw size={16} className={loading ? 'icon-spin' : ''} />
          </button>
          <button
            className="btn btn-primary"
            onClick={() => setIsUploadModalOpen(true)}
          >
            <Plus size={16} />
            <span>Add Government Document</span>
          </button>
        </div>
      </div>

      {successToast && (
        <div className="kb-toast-banner" role="status">
          <CheckCircle2 size={18} className="toast-icon" />
          <span>{successToast}</span>
        </div>
      )}

      {error && (
        <div className="kb-error-banner" role="alert">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Summary Statistics Cards */}
      <div className="kb-stats-grid">
        <div className="kb-stat-card">
          <div className="stat-icon-wrap stat-icon-primary">
            <BookOpen size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">Total Documents</span>
            <span className="stat-value">{stats ? stats.total_documents : '—'}</span>
            <span className="stat-sub">Indexed Acts & Gazettes</span>
          </div>
        </div>

        <div className="kb-stat-card">
          <div className="stat-icon-wrap stat-icon-info">
            <FileText size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">Extracted Pages</span>
            <span className="stat-value">{stats ? stats.total_pages : '—'}</span>
            <span className="stat-sub">Clean Page Boundaries</span>
          </div>
        </div>

        <div className="kb-stat-card">
          <div className="stat-icon-wrap stat-icon-accent">
            <Layers size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">FAISS Chunks</span>
            <span className="stat-value">{stats ? stats.total_chunks : '—'}</span>
            <span className="stat-sub">BGE 384-d Vectors</span>
          </div>
        </div>

        <div className="kb-stat-card">
          <div className="stat-icon-wrap stat-icon-neutral">
            <Calendar size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">Last Updated</span>
            <span className="stat-value text-sm">
              {stats ? stats.last_updated : '—'}
            </span>
            <span className="stat-sub">
              {stats ? formatBytes(stats.total_file_size_bytes) : ''} Storage
            </span>
          </div>
        </div>
      </div>

      {/* Document Registry Table */}
      <div className="kb-table-section">
        <div className="kb-table-header">
          <div className="table-header-left">
            <ShieldCheck size={18} className="shield-icon" />
            <h3 className="table-heading">Verified Legal Document Registry</h3>
            <span className="table-count-badge">{documents.length} registered</span>
          </div>
        </div>

        <DocumentTable
          documents={documents}
          loading={loading}
          onSelectDocument={(doc) => setSelectedDoc(doc)}
          onDeleteDocument={(doc) => setDeleteTargetDoc(doc)}
        />
      </div>

      {/* Modals */}
      <DocumentDetailsModal
        document={selectedDoc}
        onClose={() => setSelectedDoc(null)}
      />

      <UploadDocumentModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onSuccess={() => {
          loadData();
        }}
      />

      <DeleteConfirmModal
        document={deleteTargetDoc}
        isOpen={deleteTargetDoc !== null}
        isDeleting={isDeleting}
        onClose={() => setDeleteTargetDoc(null)}
        onConfirm={handleDeleteConfirm}
      />
    </div>
  );
};
