import React, { useState, useMemo } from 'react';
import {
  FileText,
  Search,
  Filter,
  ArrowUpDown,
  Eye,
  Trash2,
  CheckCircle,
  Clock,
  AlertCircle,
  Lock
} from 'lucide-react';
import { DocumentItem } from '../../types/api';

interface DocumentTableProps {
  documents: DocumentItem[];
  loading: boolean;
  onSelectDocument: (doc: DocumentItem) => void;
  onDeleteDocument: (doc: DocumentItem) => void;
}

type SortField = 'title' | 'pages' | 'chunks' | 'size' | 'uploaded' | 'status';
type SortOrder = 'asc' | 'desc';

export const DocumentTable: React.FC<DocumentTableProps> = ({
  documents,
  loading,
  onSelectDocument,
  onDeleteDocument
}) => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');
  const [selectedStatus, setSelectedStatus] = useState<string>('ALL');
  const [sortField, setSortField] = useState<SortField>('uploaded');
  const [sortOrder, setSortOrder] = useState<SortOrder>('asc');

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortOrder('asc');
    }
  };

  const filteredAndSortedDocs = useMemo(() => {
    return documents
      .filter((doc) => {
        const matchesSearch =
          doc.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
          doc.original_file_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          doc.document_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
          doc.source.toLowerCase().includes(searchTerm.toLowerCase());

        const matchesCategory =
          selectedCategory === 'ALL' ||
          doc.category.toUpperCase() === selectedCategory.toUpperCase();

        const matchesStatus =
          selectedStatus === 'ALL' ||
          doc.status.toUpperCase() === selectedStatus.toUpperCase();

        return matchesSearch && matchesCategory && matchesStatus;
      })
      .sort((a, b) => {
        let comp = 0;
        switch (sortField) {
          case 'title':
            comp = a.title.localeCompare(b.title);
            break;
          case 'pages':
            comp = a.page_count - b.page_count;
            break;
          case 'chunks':
            comp = a.chunk_count - b.chunk_count;
            break;
          case 'size':
            comp = a.file_size_bytes - b.file_size_bytes;
            break;
          case 'uploaded':
            if (a.is_baseline && !b.is_baseline) return -1;
            if (!a.is_baseline && b.is_baseline) return 1;
            comp = a.uploaded_at.localeCompare(b.uploaded_at);
            break;
          case 'status':
            comp = a.status.localeCompare(b.status);
            break;
        }
        return sortOrder === 'asc' ? comp : -comp;
      });
  }, [documents, searchTerm, selectedCategory, selectedStatus, sortField, sortOrder]);

  const categories = useMemo(() => {
    const cats = new Set(documents.map((d) => d.category));
    return ['ALL', ...Array.from(cats)];
  }, [documents]);

  const renderStatusBadge = (status: string) => {
    switch (status) {
      case 'INDEXED':
        return (
          <span className="table-status-pill status-indexed">
            <CheckCircle size={12} /> Indexed
          </span>
        );
      case 'PROCESSING':
      case 'EMBEDDING':
      case 'INDEXING':
      case 'UPLOADING':
        return (
          <span className="table-status-pill status-processing">
            <Clock size={12} /> {status}
          </span>
        );
      case 'FAILED':
        return (
          <span className="table-status-pill status-failed">
            <AlertCircle size={12} /> Failed
          </span>
        );
      default:
        return <span className="table-status-pill">{status}</span>;
    }
  };

  return (
    <div className="registry-table-container">
      {/* Search and Filters Bar */}
      <div className="table-controls-bar">
        <div className="search-input-wrap">
          <Search size={16} className="search-icon" />
          <input
            type="text"
            className="table-search-input"
            placeholder="Search documents, IDs, or ministries..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="filter-group">
          <div className="filter-item">
            <Filter size={14} className="filter-icon" />
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="table-filter-select"
            >
              {categories.map((c) => (
                <option key={c} value={c}>
                  {c === 'ALL' ? 'All Categories' : c}
                </option>
              ))}
            </select>
          </div>

          <div className="filter-item">
            <select
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value)}
              className="table-filter-select"
            >
              <option value="ALL">All Statuses</option>
              <option value="INDEXED">Indexed</option>
              <option value="PROCESSING">Processing</option>
              <option value="FAILED">Failed</option>
            </select>
          </div>
        </div>
      </div>

      {/* Responsive Table */}
      <div className="table-responsive-wrapper">
        <table className="registry-table">
          <thead>
            <tr>
              <th className="col-num">#</th>
              <th className="col-doc" onClick={() => handleSort('title')}>
                <div className="th-content">
                  <span>Document / Bare Act</span>
                  <ArrowUpDown size={12} />
                </div>
              </th>
              <th className="col-cat">Category</th>
              <th className="col-pages text-right" onClick={() => handleSort('pages')}>
                <div className="th-content text-right">
                  <span>Pages</span>
                  <ArrowUpDown size={12} />
                </div>
              </th>
              <th className="col-chunks text-right" onClick={() => handleSort('chunks')}>
                <div className="th-content text-right">
                  <span>Chunks</span>
                  <ArrowUpDown size={12} />
                </div>
              </th>
              <th className="col-size text-right" onClick={() => handleSort('size')}>
                <div className="th-content text-right">
                  <span>Size</span>
                  <ArrowUpDown size={12} />
                </div>
              </th>
              <th className="col-uploaded" onClick={() => handleSort('uploaded')}>
                <div className="th-content">
                  <span>Uploaded On</span>
                  <ArrowUpDown size={12} />
                </div>
              </th>
              <th className="col-status" onClick={() => handleSort('status')}>
                <div className="th-content">
                  <span>Status</span>
                  <ArrowUpDown size={12} />
                </div>
              </th>
              <th className="col-actions text-center">Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={9} className="table-loading-state">
                  <div className="pulse-loader-mini" />
                  <span>Loading Knowledge Base Registry...</span>
                </td>
              </tr>
            ) : filteredAndSortedDocs.length === 0 ? (
              <tr>
                <td colSpan={9} className="table-empty-state">
                  <FileText size={32} className="empty-icon" />
                  <p>No documents found matching your search or filters.</p>
                </td>
              </tr>
            ) : (
              filteredAndSortedDocs.map((doc, idx) => (
                <tr
                  key={doc.document_id}
                  className="table-row-item"
                  onClick={() => onSelectDocument(doc)}
                >
                  <td className="col-num">{idx + 1}</td>
                  <td className="col-doc">
                    <div className="doc-cell-content">
                      <FileText size={16} className="doc-type-icon" />
                      <div className="doc-name-group">
                        <span className="doc-title-text" title={doc.title}>
                          {doc.title}
                        </span>
                        <span className="doc-filename-sub" title={doc.original_file_name}>
                          {doc.original_file_name} {doc.is_baseline && '• Baseline'}
                        </span>
                      </div>
                    </div>
                  </td>
                  <td className="col-cat">
                    <span className={`table-cat-badge tag-${doc.category.toLowerCase()}`}>
                      {doc.category}
                    </span>
                  </td>
                  <td className="col-pages text-right">{doc.page_count}</td>
                  <td className="col-chunks text-right font-bold">{doc.chunk_count}</td>
                  <td className="col-size text-right">{formatBytes(doc.file_size_bytes)}</td>
                  <td className="col-uploaded">
                    <span className="uploaded-date-text">
                      {doc.uploaded_at === 'Baseline'
                        ? 'Baseline'
                        : new Date(doc.uploaded_at).toLocaleDateString()}
                    </span>
                  </td>
                  <td className="col-status">{renderStatusBadge(doc.status)}</td>
                  <td className="col-actions text-center" onClick={(e) => e.stopPropagation()}>
                    <div className="action-buttons-group">
                      <button
                        className="btn-action-sm btn-action-view"
                        onClick={() => onSelectDocument(doc)}
                        aria-label={`View details for ${doc.title}`}
                        title="View document metadata"
                      >
                        <Eye size={13} />
                        <span>View</span>
                      </button>

                      {doc.is_baseline ? (
                        <button
                          className="btn-action-sm btn-action-disabled"
                          onClick={() => onDeleteDocument(doc)}
                          aria-label="Baseline document cannot be deleted"
                          title="Baseline Government Document (Protected from deletion)"
                        >
                          <Lock size={12} />
                          <span>Protected</span>
                        </button>
                      ) : (
                        <button
                          className="btn-action-sm btn-action-delete"
                          onClick={() => onDeleteDocument(doc)}
                          aria-label={`Delete ${doc.title}`}
                          title="Delete document from knowledge base"
                        >
                          <Trash2 size={13} />
                          <span>Delete</span>
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
