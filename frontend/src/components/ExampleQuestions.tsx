import React from 'react';
import { Sparkles } from 'lucide-react';

interface ExampleQuestionsProps {
  onSelectQuestion: (question: string) => void;
  disabled?: boolean;
}

const EXAMPLES = [
  { id: 1, text: "How can I file an RTI application?", category: "RTI" },
  { id: 2, text: "What information can a citizen request under RTI?", category: "RTI" },
  { id: 3, text: "What are my rights as a consumer?", category: "Consumer" },
  { id: 4, text: "How can I file a consumer complaint?", category: "Consumer" },
  { id: 5, text: "What does the District Consumer Commission do?", category: "Consumer" },
];

export const ExampleQuestions: React.FC<ExampleQuestionsProps> = ({ onSelectQuestion, disabled }) => {
  return (
    <div className="example-questions-container" aria-label="Suggested standard questions">
      <div className="example-header">
        <Sparkles size={16} className="sparkle-icon" />
        <span>Try asking one of these benchmark civic questions:</span>
      </div>
      <div className="example-chips">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.id}
            type="button"
            className={`example-chip ${ex.category.toLowerCase()}-chip`}
            onClick={() => onSelectQuestion(ex.text)}
            disabled={disabled}
            aria-label={`Ask: ${ex.text}`}
          >
            <span className="category-tag">{ex.category}</span>
            <span className="question-text">{ex.text}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
