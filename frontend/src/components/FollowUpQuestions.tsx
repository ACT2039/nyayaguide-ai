import React from 'react';
import { HelpCircle, ArrowRight } from 'lucide-react';

interface FollowUpQuestionsProps {
  questions: string[];
  onSelect: (question: string) => void;
  disabled?: boolean;
}

export const FollowUpQuestions: React.FC<FollowUpQuestionsProps> = ({ questions, onSelect, disabled }) => {
  if (!questions || questions.length === 0) return null;

  return (
    <div className="follow-up-section" aria-label="Suggested follow-up legal questions">
      <div className="follow-up-header">
        <HelpCircle size={16} className="follow-up-icon" />
        <span>Suggested follow-up questions (grounded in retrieved context):</span>
      </div>

      <div className="follow-up-grid">
        {questions.map((q, idx) => (
          <button
            key={idx}
            type="button"
            className="follow-up-chip"
            onClick={() => onSelect(q)}
            disabled={disabled}
            aria-label={`Ask follow-up: ${q}`}
          >
            <span className="chip-text">{q}</span>
            <ArrowRight size={14} className="chip-arrow" />
          </button>
        ))}
      </div>
    </div>
  );
};
