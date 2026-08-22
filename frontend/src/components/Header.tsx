import React from 'react';
import { Scale, ShieldCheck, MessageSquare, Database } from 'lucide-react';

interface HeaderProps {
  activeTab: 'chat' | 'kb';
  onTabChange: (tab: 'chat' | 'kb') => void;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, onTabChange }) => {
  return (
    <header className="site-header">
      <div className="header-container">
        <div className="brand-section">
          <div className="logo-badge">
            <Scale className="logo-icon" size={28} />
          </div>
          <div>
            <div className="title-row">
              <h1 className="brand-title">NyayaGuide AI</h1>
              <span className="kb-badge">
                <ShieldCheck size={14} className="badge-icon" />
                Official Govt of India Knowledge Base
              </span>
            </div>
            <p className="brand-subtitle">
              Source-Grounded Civic Rights & Legal Assistant • Right to Information (RTI) & Consumer Protection
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="header-nav-tabs" aria-label="Main Navigation">
          <button
            className={`nav-tab-btn ${activeTab === 'chat' ? 'nav-tab-active' : ''}`}
            onClick={() => onTabChange('chat')}
          >
            <MessageSquare size={16} />
            <span>Citizen Assistant</span>
          </button>

          <button
            className={`nav-tab-btn ${activeTab === 'kb' ? 'nav-tab-active' : ''}`}
            onClick={() => onTabChange('kb')}
          >
            <Database size={16} />
            <span>Knowledge Base</span>
          </button>
        </nav>
      </div>
    </header>
  );
};
