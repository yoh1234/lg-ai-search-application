import React from 'react';

interface ThinkingIndicatorProps {
  message: string;
  thinkingSteps: any[];
}

export const ThinkingIndicator: React.FC<ThinkingIndicatorProps> = ({ message, thinkingSteps }) => {
  return (
    <div className="thinking-indicator">
      <div className="thinking-content">
        <div className="thinking-spinner"></div>
        <div className="thinking-details">
          <span className="thinking-text">{message}</span>
          
          {/* Show detailed thinking steps */}
          {thinkingSteps.length > 0 && (
            <div className="thinking-steps">
              {thinkingSteps.map((step, index) => (
                <div key={index} className="thinking-step">
                  {step.type === 'phase_start' && (
                    <div className="phase-start">
                      <strong>Phase: {step.phase.replace('_', ' ')}</strong>
                      <p>{step.description}</p>
                    </div>
                  )}
                  {step.type === 'thinking_detail' && (
                    <div className="thinking-detail">
                      <div className="thinking-phase">{step.phase.replace('_', ' ')}</div>
                      <div className="thinking-paragraph">{step.thinking}</div>
                      {step.result && (
                        <div className="thinking-result">
                          Result: {JSON.stringify(step.result)}
                        </div>
                      )}
                    </div>
                  )}
                  <div className="thinking-timestamp">
                    {new Date(step.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};