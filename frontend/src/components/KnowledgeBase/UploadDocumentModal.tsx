import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileText,
  X,
  CheckCircle,
  AlertTriangle,
  Loader2,
  Bookmark
} from 'lucide-react';
import { uploadDocument, fetchDocumentStatus, UploadMetadata } from '../../services/api';

interface UploadDocumentModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

type StepKey = 'uploading' | 'extracting' | 'chunking' | 'embedding' | 'indexing' | 'completed';

const STAGES: { key: StepKey; label: string }[] = [
  { key: 'uploading', label: 'Uploading PDF' },
  { key: 'extracting', label: 'Extracting & Cleaning Text' },
  { key: 'chunking', label: 'Legal Chunking & Section Tagging' },
  { key: 'embedding', label: 'Generating BGE Embeddings' },
  { key: 'indexing', label: 'Incremental FAISS Indexing' },
  { key: 'completed', label: 'Completed & Indexed' }
];

export const UploadDocumentModal: React.FC<UploadDocumentModalProps> = ({
  isOpen,
  onClose,
  onSuccess
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [category, setCategory] = useState<string>('RTI');
  const [title, setTitle] = useState<string>('');
  const [source, setSource] = useState<string>('Government of India');
  const [authority, setAuthority] = useState<string>('');
  const [sourceUrl, setSourceUrl] = useState<string>('');
  const [dragOver, setDragOver] = useState<boolean>(false);

  const [loading, setLoading] = useState<boolean>(false);
  const [currentStageIndex, setCurrentStageIndex] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [successInfo, setSuccessInfo] = useState<{
    docId: string;
    pages?: number | null;
    chunks?: number | null;
  } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      if (!selected.name.toLowerCase().endsWith('.pdf')) {
        setError('Only PDF files (.pdf) are allowed.');
        return;
      }
      if (selected.size > 20 * 1024 * 1024) {
        setError('File exceeds maximum size limit of 20 MB.');
        return;
      }
      setFile(selected);
      setError(null);
      if (!title) {
        setTitle(selected.name.replace('.pdf', '').replace(/_/g, ' '));
      }
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const dropped = e.dataTransfer.files[0];
      if (!dropped.name.toLowerCase().endsWith('.pdf')) {
        setError('Only PDF files (.pdf) are allowed.');
        return;
      }
      if (dropped.size > 20 * 1024 * 1024) {
        setError('File exceeds maximum size limit of 20 MB.');
        return;
      }
      setFile(dropped);
      setError(null);
      if (!title) {
        setTitle(dropped.name.replace('.pdf', '').replace(/_/g, ' '));
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || loading) return;

    setLoading(true);
    setError(null);
    setSuccessInfo(null);
    setCurrentStageIndex(0); // Uploading

    const metadata: UploadMetadata = {
      category,
      title: title.trim() || undefined,
      source: source.trim() || undefined,
      authority: authority.trim() || undefined,
      source_url: sourceUrl.trim() || undefined
    };

    try {
      const resp = await uploadDocument(file, metadata);

      if (resp.status === 'INDEXED') {
        setCurrentStageIndex(5); // Completed
        setSuccessInfo({
          docId: resp.document_id,
          pages: resp.page_count,
          chunks: resp.chunk_count
        });
        onSuccess();
        setLoading(false);
        return;
      }

      // Status is PROCESSING -> Poll status until INDEXED or FAILED
      setCurrentStageIndex(1); // Extracting & Cleaning Text

      let attempts = 0;
      const maxAttempts = 120; // 6 minutes max

      const checkStatus = async () => {
        try {
          const statusResp = await fetchDocumentStatus(resp.document_id);

          if (statusResp.status === 'PROCESSING') {
            setCurrentStageIndex(1);
          } else if (statusResp.status === 'EMBEDDING') {
            setCurrentStageIndex(3);
          } else if (statusResp.status === 'INDEXING') {
            setCurrentStageIndex(4);
          } else if (statusResp.status === 'INDEXED') {
            setCurrentStageIndex(5);
            setSuccessInfo({
              docId: resp.document_id,
              pages: statusResp.page_count,
              chunks: statusResp.chunk_count
            });
            onSuccess();
            setLoading(false);
            return;
          } else if (statusResp.status === 'FAILED') {
            setError(statusResp.error_message || 'Background document processing failed.');
            setLoading(false);
            return;
          }

          attempts++;
          if (attempts < maxAttempts) {
            setTimeout(checkStatus, 3000);
          } else {
            setError('Ingestion is taking longer than expected. Please check dashboard status.');
            setLoading(false);
          }
        } catch (pollErr: any) {
          const errMsg = String(pollErr?.message || pollErr || '');
          if (errMsg.includes('404') || errMsg.includes('not found')) {
            setError('Document record not found or processing was interrupted. Please try uploading again.');
            setLoading(false);
            return;
          }
          attempts++;
          if (attempts < maxAttempts) {
            setTimeout(checkStatus, 3000);
          } else {
            setError(pollErr.message || 'Error checking processing status.');
            setLoading(false);
          }
        }
      };

      setTimeout(checkStatus, 2000);
    } catch (err: any) {
      setError(err.message || 'Failed to upload and index document.');
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setTitle('');
    setSource('Government of India');
    setAuthority('');
    setSourceUrl('');
    setError(null);
    setSuccessInfo(null);
    setCurrentStageIndex(0);
  };

  const handleModalClose = () => {
    if (loading) return;
    handleReset();
    onClose();
  };

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-labelledby="upload-modal-title"
      onClick={handleModalClose}
    >
      <div
        className="modal-content upload-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div className="modal-title-group">
            <div className="category-avatar avatar-primary">
              <UploadCloud size={20} />
            </div>
            <div>
              <h3 id="upload-modal-title" className="modal-title">
                Add Government Legal Document
              </h3>
              <p className="modal-subtitle">
                Automatic PDF parsing, chunking, BGE embedding, and incremental FAISS indexing
              </p>
            </div>
          </div>
          <button
            className="modal-close-btn"
            onClick={handleModalClose}
            disabled={loading}
            aria-label="Close modal"
          >
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          {error && (
            <div className="modal-error-banner" role="alert">
              <AlertTriangle size={18} className="error-icon" />
              <div>
                <strong>Upload Rejected</strong>
                <p>{error}</p>
              </div>
            </div>
          )}

          {successInfo && (
            <div className="modal-success-banner" role="status">
              <CheckCircle size={22} className="success-icon" />
              <div>
                <strong>Document Successfully Indexed!</strong>
                <p>
                  Document ID: <code>{successInfo.docId}</code>
                  {successInfo.pages !== undefined && ` • ${successInfo.pages} Pages`}
                  {successInfo.chunks !== undefined && ` • ${successInfo.chunks} Chunks Added`}
                </p>
                <span className="success-note">
                  Immediately searchable by NyayaGuide AI without server restart.
                </span>
              </div>
            </div>
          )}

          {loading ? (
            <div className="upload-progress-container">
              <h4 className="progress-title">Ingesting & Indexing Document...</h4>
              <div className="stages-list">
                {STAGES.map((stg, sIdx) => {
                  const isDone = sIdx < currentStageIndex;
                  const isCurrent = sIdx === currentStageIndex;
                  return (
                    <div
                      key={stg.key}
                      className={`stage-item ${isDone ? 'stage-done' : ''} ${
                        isCurrent ? 'stage-current' : ''
                      }`}
                    >
                      <div className="stage-icon-wrap">
                        {isDone ? (
                          <CheckCircle size={16} className="stage-icon-done" />
                        ) : isCurrent ? (
                          <Loader2 size={16} className="stage-icon-spinner" />
                        ) : (
                          <div className="stage-bullet" />
                        )}
                      </div>
                      <span className="stage-label">{stg.label}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <form id="upload-form" onSubmit={handleSubmit}>
              {/* Dropzone */}
              <div
                className={`dropzone ${dragOver ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="application/pdf"
                  style={{ display: 'none' }}
                  onChange={handleFileChange}
                />
                <UploadCloud size={36} className="dropzone-icon" />
                {file ? (
                  <div className="file-preview">
                    <FileText size={20} className="file-icon" />
                    <div className="file-info">
                      <span className="file-name" title={file.name}>{file.name}</span>
                      <span className="file-size">
                        {(file.size / (1024 * 1024)).toFixed(2)} MB
                      </span>
                    </div>
                    <button
                      type="button"
                      className="btn-change-file"
                      onClick={(e) => {
                        e.stopPropagation();
                        setFile(null);
                      }}
                    >
                      Change
                    </button>
                  </div>
                ) : (
                  <div className="dropzone-text">
                    <p className="dropzone-primary">
                      Drag & Drop Government PDF here, or <span>Browse files</span>
                    </p>
                    <p className="dropzone-secondary">
                      Max file size: 20 MB • Only official bare acts & gazettes (.pdf)
                    </p>
                  </div>
                )}
              </div>

              {/* Metadata inputs */}
              <div className="form-grid">
                <div className="form-group">
                  <label htmlFor="doc-category">
                    <Bookmark size={14} /> Legal Category *
                  </label>
                  <select
                    id="doc-category"
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                    required
                  >
                    <option value="RTI">Right to Information (RTI)</option>
                    <option value="CONSUMER">Consumer Protection</option>
                    <option value="CIVIC">Civic & Public Services</option>
                    <option value="EDUCATION">Education & Students</option>
                    <option value="TRANSPORT">Transport & Motor Vehicles</option>
                    <option value="ENVIRONMENT">Environment & Forests</option>
                    <option value="OTHER">Other Statutory Rules</option>
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="doc-title">Official Document Title</label>
                  <input
                    id="doc-title"
                    type="text"
                    placeholder="e.g. Consumer Protection (E-Commerce) Rules, 2020"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="doc-source">Issuing Authority / Ministry</label>
                  <input
                    id="doc-source"
                    type="text"
                    placeholder="e.g. Ministry of Consumer Affairs, Food & Public Distribution"
                    value={source}
                    onChange={(e) => setSource(e.target.value)}
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="doc-url">Official Gazette URL (Optional)</label>
                  <input
                    id="doc-url"
                    type="url"
                    placeholder="https://egazette.gov.in/..."
                    value={sourceUrl}
                    onChange={(e) => setSourceUrl(e.target.value)}
                  />
                </div>
              </div>
            </form>
          )}
        </div>

        <div className="modal-footer">
          {successInfo ? (
            <button className="btn btn-primary" onClick={handleModalClose}>
              Done
            </button>
          ) : (
            <>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleModalClose}
                disabled={loading}
              >
                Cancel
              </button>
              <button
                type="submit"
                form="upload-form"
                className="btn btn-primary"
                disabled={!file || loading}
              >
                {loading ? 'Processing...' : 'Upload & Index into FAISS'}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
