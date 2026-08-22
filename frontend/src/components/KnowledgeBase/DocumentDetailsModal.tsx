import React from 'react';
import {
  FileText,
  Calendar,
  X,
  ExternalLink,
  ShieldCheck,
  Hash,
  Bookmark,
  Layers,
  HardDrive
} from 'lucide-react';
import { DocumentItem } from '../../types/api';

interface DocumentDetailsModalProps {
  document: DocumentItem | null;
  onClose: () => void;
}

export const DocumentDetailsModal: React.FC<DocumentDetailsModalProps> = ({
  document,
  onClose
}) => {
  if (!document) return null;

  const isRTI = document.category === 'RTI';
  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-labelledby="doc-details-title"
      onClick={onClose}
    >
      <div
        className="modal-content doc-details-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div className="modal-title-group">
            <div className={`category-avatar ${isRTI ? 'avatar-rti' : 'avatar-consumer'}`}>
              <FileText size={20} />
            </div>
            <div>
              <h3 id="doc-details-title" className="modal-title">
                {document.title}
              </h3>
              <span className="modal-subtitle doc-id-badge">
                ID: {document.document_id} {document.is_baseline && '• Baseline Gazette'}
              </span>
            </div>
          </div>
          <button
            className="modal-close-btn"
            onClick={onClose}
            aria-label="Close modal"
          >
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          <div className="details-grid">
            <div className="detail-card">
              <span className="detail-label">
                <FileText size={14} className="detail-icon" /> Original File Name
              </span>
              <span className="detail-value mono-val" title={document.original_file_name}>
                {document.original_file_name}
              </span>
            </div>

            <div className="detail-card">
              <span className="detail-label">
                <Bookmark size={14} className="detail-icon" /> Legal Category
              </span>
              <span className={`status-pill pill-${document.category.toLowerCase()}`}>
                {document.category}
              </span>
            </div>

            <div className="detail-card">
              <span className="detail-label">
                <Layers size={14} className="detail-icon" /> Extracted Pages
              </span>
              <span className="detail-value">{document.page_count} Pages</span>
            </div>

            <div className="detail-card">
              <span className="detail-label">
                <Hash size={14} className="detail-icon" /> FAISS RAG Chunks
              </span>
              <span className="detail-value highlight-val">
                {document.chunk_count} Chunks
              </span>
            </div>

            <div className="detail-card">
              <span className="detail-label">
                <HardDrive size={14} className="detail-icon" /> File Size
              </span>
              <span className="detail-value">{formatBytes(document.file_size_bytes)}</span>
            </div>

            <div className="detail-card">
              <span className="detail-label">
                <ShieldCheck size={14} className="detail-icon" /> Ingestion Status
              </span>
              <span className={`status-pill pill-${document.status.toLowerCase()}`}>
                {document.status}
              </span>
            </div>

            <div className="detail-card full-span">
              <span className="detail-label">Issuing Authority / Source</span>
              <span className="detail-value">
                {document.authority || document.source || 'Government of India'}
              </span>
            </div>

            <div className="detail-card full-span">
              <span className="detail-label">SHA-256 Content Hash (Deduplication Signature)</span>
              <span className="detail-value mono-val hash-val" title={document.content_hash}>
                {document.content_hash}
              </span>
            </div>

            <div className="detail-card">
              <span className="detail-label">
                <Calendar size={14} className="detail-icon" /> Uploaded On
              </span>
              <span className="detail-value">
                {document.uploaded_at === 'Baseline'
                  ? 'Baseline Government Document'
                  : new Date(document.uploaded_at).toLocaleString()}
              </span>
            </div>

            <div className="detail-card">
              <span className="detail-label">
                <Calendar size={14} className="detail-icon" /> Indexed On
              </span>
              <span className="detail-value">
                {document.indexed_at === 'Baseline'
                  ? 'Baseline Verification'
                  : document.indexed_at
                  ? new Date(document.indexed_at).toLocaleString()
                  : 'Pending'}
              </span>
            </div>
          </div>

          {document.error_message && (
            <div className="modal-error-box">
              <strong>Processing Error:</strong>
              <p>{document.error_message}</p>
            </div>
          )}

          {document.source_url && (
            <div className="modal-link-box">
              <a
                href={document.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="btn-link"
              >
                <span>View Official Gazette / Publication Source</span>
                <ExternalLink size={14} />
              </a>
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
