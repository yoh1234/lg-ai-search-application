import React from 'react';

interface ThinkingIndicatorProps {
  message: string;
}

export const ThinkingIndicator: React.FC<ThinkingIndicatorProps> = ({ message }) => {
  return (
    <div className="thinking-indicator">
      <div className="thinking-content">
        <div className="thinking-spinner"></div>
        <span className="thinking-text">{message}</span>
      </div>
    </div>
  );
};

