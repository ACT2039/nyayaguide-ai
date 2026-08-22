import React from 'react';
import { User, Bot, AlertTriangle, BookMarked, Cpu } from 'lucide-react';
import { ConversationTurn } from '../types/api';
import { SourceCard } from './SourceCard';
import { FollowUpQuestions } from './FollowUpQuestions';
import { AbstentionNotice } from './AbstentionNotice';
import { FormattedLegalText } from './FormattedLegalText';

interface MessageListProps {
  turns: ConversationTurn[];
  onSelectFollowUp: (question: string) => void;
  disabled?: boolean;
}

export const MessageList: React.FC<MessageListProps> = ({ turns, onSelectFollowUp, disabled }) => {
  if (turns.length === 0) return null;

  return (
    <div className="conversation-timeline" aria-live="polite" aria-label="Legal conversation history">
      {turns.map((turn, turnIdx) => (
        <div key={turn.id} className="conversation-turn">
          {/* User Question */}
          <div className="user-message-card">
            <div className="avatar user-avatar" aria-hidden="true">
              <User size={18} />
            </div>
            <div className="message-content">
              <div className="message-header">
                <span className="speaker-name">Citizen Query</span>
                <span className="message-time">{turn.timestamp}</span>
              </div>
              <p className="question-text-rendered">{turn.question}</p>
            </div>
          </div>

          {/* Assistant Answer or Loading State */}
          <div className="assistant-message-card">
            <div className="avatar bot-avatar" aria-hidden="true">
              <Bot size={18} />
            </div>
            <div className="message-content">
              <div className="message-header">
                <span className="speaker-name">NyayaGuide AI</span>
                {turn.response?.model_used && (
                  <span className="model-tag">
                    <Cpu size={12} /> {turn.response.model_used}
                  </span>
                )}
                {turn.response?.top_score !== undefined && turn.response.top_score > 0 && (
                  <span className="relevance-tag">
                    Cosine Match: {(turn.response.top_score * 100).toFixed(1)}%
                  </span>
                )}
              </div>

              {turn.loading && (
                <div className="loading-state" role="status" aria-label="Retrieving legal documents and generating answer">
                  <div className="legal-pulse-loader">
                    <div className="pulse-bar"></div>
                    <div className="pulse-bar"></div>
                    <div className="pulse-bar"></div>
                  </div>
                  <div className="loading-labels">
                    <p className="loading-primary">Retrieving official gazettes & bare acts from FAISS index...</p>
                    <p className="loading-secondary">Applying 13 strict grounding rules & validating citations</p>
                  </div>
                </div>
              )}

              {turn.error && (
                <div className="error-box" role="alert">
                  <AlertTriangle size={18} className="error-icon" />
                  <div className="error-text">
                    <strong>Unable to complete request</strong>
                    <p>{turn.error}</p>
                  </div>
                </div>
              )}

              {turn.response && (
                <div className="response-body">
                  {turn.response.is_abstention ? (
                    <AbstentionNotice
                      message={turn.response.answer}
                      onSelectExample={onSelectFollowUp}
                    />
                  ) : (
                    <>
                      {/* Generated Answer Body with Safe Bold Markdown & Contextual Legal Emojis */}
                      <div className="answer-prose">
                        {turn.response.answer.split('\n\n').map((paragraph, pIdx) => {
                          if (paragraph.startsWith('Sources:')) return null; // Sources rendered cleanly in SourceCards below
                          return (
                            <FormattedLegalText key={pIdx} content={paragraph} />
                          );
                        })}
                      </div>

                      {/* Structured Source Citations */}
                      {turn.response.sources && turn.response.sources.length > 0 && (
                        <div className="sources-container" aria-label="Grounded legal sources">
                          <div className="sources-header">
                            <BookMarked size={16} className="sources-header-icon" />
                            <h3 className="sources-heading">
                              Verified Legal Sources ({turn.response.sources.length})
                            </h3>
                            <span className="sources-note">
                              Programmatically extracted directly from source chunk metadata
                            </span>
                          </div>
                          <div className="sources-grid">
                            {turn.response.sources.map((src, srcIdx) => (
                              <SourceCard key={src.chunk_id || srcIdx} source={src} index={srcIdx} />
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Suggested Follow-up Questions (Only on the latest turn) */}
                      {turnIdx === turns.length - 1 && turn.response.follow_up_questions && turn.response.follow_up_questions.length > 0 && (
                        <FollowUpQuestions
                          questions={turn.response.follow_up_questions}
                          onSelect={onSelectFollowUp}
                          disabled={disabled}
                        />
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
