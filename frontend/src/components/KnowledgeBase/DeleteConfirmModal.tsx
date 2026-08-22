import React from 'react';
import { AlertTriangle, Trash2, X, FileText, Bookmark, Layers, Hash } from 'lucide-react';
import { DocumentItem } from '../../types/api';

interface DeleteConfirmModalProps {
  document: DocumentItem | null;
  isOpen: boolean;
  isDeleting: boolean;
  onClose: () => void;
  onConfirm: (document: DocumentItem) => void;
}

export const DeleteConfirmModal: React.FC<DeleteConfirmModalProps> = ({
  document,
  isOpen,
  isDeleting,
  onClose,
  onConfirm
}) => {
  if (!isOpen || !document) return null;

  const isBaseline = document.is_baseline;

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-labelledby="delete-dialog-title"
      onClick={() => {
        if (!isDeleting) onClose();
      }}
    >
      <div
        className="modal-content delete-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div className="modal-title-group">
            <div className={`category-avatar ${isBaseline ? 'avatar-baseline-locked' : 'avatar-delete'}`}>
              <AlertTriangle size={20} />
            </div>
            <div>
              <h3 id="delete-dialog-title" className="modal-title delete-title">
                {isBaseline ? 'Protected Baseline Document' : 'Delete Document from Knowledge Base?'}
              </h3>
              <p className="modal-subtitle">
                ID: {document.document_id}
              </p>
            </div>
          </div>
          <button
            className="modal-close-btn"
            onClick={onClose}
            disabled={isDeleting}
            aria-label="Close modal"
          >
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          {isBaseline ? (
            <div className="baseline-protection-box">
              <p className="protection-message">
                <strong>Deletion Prohibited:</strong> <code>{document.original_file_name}</code> is a core Government of India baseline statutory document (The Right to Information Act or The Consumer Protection Act). Core baseline documents are permanent and cannot be deleted.
              </p>
            </div>
          ) : (
            <>
              <div className="delete-summary-card">
                <div className="summary-item">
                  <span className="summary-label"><FileText size={13} /> Title:</span>
                  <span className="summary-val font-bold">{document.title}</span>
                </div>
                <div className="summary-item">
                  <span className="summary-label"><Bookmark size={13} /> Category:</span>
                  <span className={`status-pill pill-${document.category.toLowerCase()}`}>{document.category}</span>
                </div>
                <div className="summary-item">
                  <span className="summary-label"><Layers size={13} /> Pages:</span>
                  <span className="summary-val">{document.page_count}</span>
                </div>
                <div className="summary-item">
                  <span className="summary-label"><Hash size={13} /> FAISS Chunks:</span>
                  <span className="summary-val font-bold">{document.chunk_count}</span>
                </div>
              </div>

              <div className="delete-warning-banner" role="alert">
                <p>
                  <strong>Warning:</strong> This action will permanently remove this document and its <strong>{document.chunk_count} vector chunks</strong> from the FAISS index and local storage. It will immediately cease to be retrievable by the NyayaGuide AI Citizen Assistant.
                </p>
              </div>
            </>
          )}
        </div>

        <div className="modal-footer">
          {isBaseline ? (
            <button className="btn btn-secondary" onClick={onClose}>
              Dismiss
            </button>
          ) : (
            <>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={onClose}
                disabled={isDeleting}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-danger"
                onClick={() => onConfirm(document)}
                disabled={isDeleting}
              >
                <Trash2 size={14} />
                <span>{isDeleting ? 'Removing from FAISS...' : 'Delete Document'}</span>
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
