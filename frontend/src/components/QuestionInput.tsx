import React, { useState, useRef, useEffect } from 'react';
import { Send, CornerDownLeft, X } from 'lucide-react';

interface QuestionInputProps {
  onSubmit: (question: string) => void;
  loading: boolean;
  prefill?: string;
}

const MAX_CHARS = 2000;

export const QuestionInput: React.FC<QuestionInputProps> = ({ onSubmit, loading, prefill }) => {
  const [question, setQuestion] = useState(prefill || '');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (prefill !== undefined) {
      setQuestion(prefill);
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    }
  }, [prefill]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const clean = question.trim();
    if (!clean || loading || clean.length > MAX_CHARS) return;
    onSubmit(clean);
    setQuestion('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const charCount = question.length;
  const isOverLimit = charCount > MAX_CHARS;
  const isEmpty = question.trim().length === 0;

  return (
    <form className="question-input-form" onSubmit={handleSubmit}>
      <div className="input-box-wrapper">
        <textarea
          ref={textareaRef}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask any question about Right to Information or Consumer Protection in India..."
          rows={3}
          disabled={loading}
          maxLength={MAX_CHARS + 50}
          aria-label="Legal Question Input"
          aria-required="true"
        />

        <div className="input-footer">
          <div className="counter-container">
            <span className={`char-counter ${isOverLimit ? 'over-limit' : ''}`}>
              {charCount}/{MAX_CHARS}
            </span>
            {question && (
              <button
                type="button"
                className="clear-btn"
                onClick={() => setQuestion('')}
                aria-label="Clear text"
                disabled={loading}
              >
                <X size={14} /> Clear
              </button>
            )}
          </div>

          <div className="action-row">
            <span className="enter-hint">
              Press <kbd>Enter <CornerDownLeft size={10} /></kbd> to ask
            </span>
            <button
              type="submit"
              className="ask-btn"
              disabled={isEmpty || isOverLimit || loading}
              aria-label="Submit legal question"
            >
              {loading ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  <span>Searching Laws...</span>
                </>
              ) : (
                <>
                  <span>Ask NyayaGuide</span>
                  <Send size={16} />
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </form>
  );
};
