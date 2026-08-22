import React from 'react';
import { FileText, ExternalLink, Bookmark, Building, Hash } from 'lucide-react';
import { SourceCitation } from '../types/api';

interface SourceCardProps {
  source: SourceCitation;
  index: number;
}

export const SourceCard: React.FC<SourceCardProps> = ({ source, index }) => {
  const isRTI = source.category === 'RTI';

  return (
    <div className={`source-card ${isRTI ? 'source-rti' : 'source-consumer'}`}>
      <div className="source-card-header">
        <div className="source-index-badge">
          <span>Source #{index + 1}</span>
        </div>
        <span className={`source-category-tag ${isRTI ? 'tag-rti' : 'tag-consumer'}`}>
          {source.category}
        </span>
      </div>

      <h4 className="source-title" title={source.title}>{source.title}</h4>

      <div className="source-meta-grid">
        <div className="meta-item full-width" title={source.document}>
          <FileText size={14} className="meta-icon" />
          <span className="meta-label">Document:</span>
          <span className="meta-value doc-filename">{source.document}</span>
        </div>

        <div className="meta-item">
          <Hash size={14} className="meta-icon" />
          <span className="meta-label">Page:</span>
          <span className="meta-value page-number">Page {source.page}</span>
        </div>

        {source.legal_reference && (
          <div className="meta-item full-width" title={source.legal_reference}>
            <Bookmark size={14} className="meta-icon" />
            <span className="meta-label">Legal Ref:</span>
            <span className="meta-value legal-ref">{source.legal_reference}</span>
          </div>
        )}

        <div className="meta-item full-width" title={source.source}>
          <Building size={14} className="meta-icon" />
          <span className="meta-label">Authority:</span>
          <span className="meta-value authority">{source.source}</span>
        </div>
      </div>

      {source.source_url && (
        <div className="source-link-row">
          <a
            href={source.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="official-source-link"
            aria-label={`View official source for ${source.title}`}
          >
            <span>View official gazette/document</span>
            <ExternalLink size={13} />
          </a>
        </div>
      )}
    </div>
  );
};
