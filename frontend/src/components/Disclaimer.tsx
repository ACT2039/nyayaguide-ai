import React from 'react';
import { ShieldAlert } from 'lucide-react';

export const Disclaimer: React.FC = () => {
  return (
    <footer className="site-disclaimer" role="contentinfo">
      <div className="disclaimer-inner">
        <ShieldAlert size={16} className="disclaimer-icon" />
        <p className="disclaimer-text">
          <strong>Official Legal Disclaimer:</strong> NyayaGuide AI provides information derived strictly from its current Government of India document knowledge base and is not a substitute for professional legal counsel or personalized legal advice.
        </p>
      </div>
    </footer>
  );
};
