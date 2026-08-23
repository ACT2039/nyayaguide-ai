import React from 'react';
import { AlertCircle, HelpCircle } from 'lucide-react';

interface AbstentionNoticeProps {
  message: string;
  onSelectExample?: (q: string) => void;
  currentQuestion?: string;
}

// Normalise a string for duplicate detection:
// lowercase, trim, collapse whitespace, remove trailing punctuation
function normaliseQ(s: string): string {
  return s
    .toLowerCase()
    .trim()
    .replace(/\s+/g, ' ')
    .replace(/[?!.,;]+$/, '');
}

export const AbstentionNotice: React.FC<AbstentionNoticeProps> = ({ message, onSelectExample, currentQuestion }) => {
  const normCurrent = currentQuestion ? normaliseQ(currentQuestion) : '';

  // Hardcoded suggestions — filtered if they match the current question
  const suggestions: Array<{ label: string; question: string }> = [
    { label: 'How can I file an RTI application?', question: 'How can I file an RTI application?' },
    { label: 'What are my rights as a consumer?', question: 'What are my rights as a consumer?' },
  ];

  const visibleSuggestions = suggestions.filter(
    (s) => normaliseQ(s.question) !== normCurrent
  );

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
            <li><strong>Consumer Protection:</strong> CPA 2019, Consumer Commission &amp; General Rules 2020</li>
          </ul>
          {onSelectExample && visibleSuggestions.length > 0 && (
            <div className="guidance-actions">
              <span className="try-label">Try asking:</span>
              {visibleSuggestions.map((s) => (
                <button
                  key={s.question}
                  type="button"
                  className="guidance-btn"
                  onClick={() => onSelectExample(s.question)}
                >
                  <HelpCircle size={13} /> {s.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
