import React, { useState } from 'react';
import { Header } from './components/Header';
import { ExampleQuestions } from './components/ExampleQuestions';
import { QuestionInput } from './components/QuestionInput';
import { MessageList } from './components/MessageList';
import { Disclaimer } from './components/Disclaimer';
import { KnowledgeBaseDashboard } from './components/KnowledgeBase/KnowledgeBaseDashboard';
import { askQuestion } from './services/api';
import { ConversationTurn } from './types/api';
import './styles/App.css';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'chat' | 'kb'>('chat');
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [inputPrefill, setInputPrefill] = useState<string | undefined>(undefined);

  const handleAsk = async (questionText: string) => {
    if (!questionText.trim() || loading) return;

    const turnId = `turn-${Date.now()}`;
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const newTurn: ConversationTurn = {
      id: turnId,
      question: questionText.trim(),
      timestamp,
      loading: true,
    };

    setTurns((prev) => [...prev, newTurn]);
    setLoading(true);
    setInputPrefill(undefined);

    try {
      const response = await askQuestion(questionText.trim());
      setTurns((prev) =>
        prev.map((t) =>
          t.id === turnId
            ? { ...t, loading: false, response, error: null }
            : t
        )
      );
    } catch (err: any) {
      setTurns((prev) =>
        prev.map((t) =>
          t.id === turnId
            ? { ...t, loading: false, error: err.message || 'An unexpected error occurred.' }
            : t
        )
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSelectExample = (questionText: string) => {
    handleAsk(questionText);
  };

  return (
    <div className="app-layout">
      <Header activeTab={activeTab} onTabChange={(tab) => setActiveTab(tab)} />

      <main className="main-content" id="main-content">
        <div className="container">
          {activeTab === 'kb' ? (
            <KnowledgeBaseDashboard />
          ) : (
            <>
              {turns.length === 0 && (
                <div className="welcome-banner">
                  <h2>Welcome to NyayaGuide AI</h2>
                  <p>
                    Ask questions about your civic and legal rights in India. All answers are strictly grounded in official Government of India bare acts and statutory rules with verified source citations.
                  </p>
                  <ExampleQuestions onSelectQuestion={handleSelectExample} disabled={loading} />
                </div>
              )}

              <MessageList
                turns={turns}
                onSelectFollowUp={handleSelectExample}
                disabled={loading}
              />

              <div className="sticky-input-container">
                {turns.length > 0 && (
                  <div className="input-prompt-label">
                    <span>Ask another question or explore follow-ups:</span>
                  </div>
                )}
                <QuestionInput
                  onSubmit={handleAsk}
                  loading={loading}
                  prefill={inputPrefill}
                />
              </div>
            </>
          )}
        </div>
      </main>

      <Disclaimer />
    </div>
  );
};

export default App;
