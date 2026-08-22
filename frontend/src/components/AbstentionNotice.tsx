import React from 'react';
import { AlertCircle, HelpCircle } from 'lucide-react';

interface AbstentionNoticeProps {
  message: string;
  onSelectExample?: (q: string) => void;
}

export const AbstentionNotice: React.FC<AbstentionNoticeProps> = ({ message, onSelectExample }) => {
  return (
    <div className="abstention-container" role="alert">
      <div className="abstention-icon-box">
        <AlertCircle size={22} className="abstention-icon" />
      </div>
      <div className="abstention-content">
        <h4 className="abstention-heading">Out of Domain / Insufficient Legal Context</h4>
        <p className="abstention-message">{message}</p>
        <div className="abstention-guidance">
          <p className="guidance-text">
            NyayaGuide AI is strictly grounded in official Government of India documents for:
          </p>
          <ul className="guidance-list">
            <li><strong>Right to Information:</strong> RTI Act 2005, RTI Rules 2012</li>
            <li><strong>Consumer Protection:</strong> CPA 2019, Consumer Commission & General Rules 2020</li>
          </ul>
          {onSelectExample && (
            <div className="guidance-actions">
              <span className="try-label">Try asking:</span>
              <button
                type="button"
                className="guidance-btn"
                onClick={() => onSelectExample("How can I file an RTI application?")}
              >
                <HelpCircle size={13} /> How can I file an RTI application?
              </button>
              <button
                type="button"
                className="guidance-btn"
                onClick={() => onSelectExample("What are my rights as a consumer?")}
              >
                <HelpCircle size={13} /> What are my rights as a consumer?
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
