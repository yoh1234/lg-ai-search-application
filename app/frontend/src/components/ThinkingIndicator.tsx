import React, { useEffect, useState } from 'react';

interface ThinkingIndicatorProps {
  message: string;
  thinkingSteps: any[];
}

export const ThinkingIndicator: React.FC<ThinkingIndicatorProps> = ({ message, thinkingSteps }) => {
  const [visibleSteps, setVisibleSteps] = useState<any[]>([]);
  const [currentPhase, setCurrentPhase] = useState<string>('');
  const [isCompleted, setIsCompleted] = useState(false);

  useEffect(() => {
    // Show steps one by one with a slower delay for better readability
    if (thinkingSteps.length > visibleSteps.length) {
      const timer = setTimeout(() => {
        setVisibleSteps(prev => {
          const nextIndex = prev.length;
          if (nextIndex < thinkingSteps.length) {
            return [...prev, thinkingSteps[nextIndex]];
          }
          return prev;
        });
      }, 800); // Slower: 800ms between steps
      return () => clearTimeout(timer);
    } else if (thinkingSteps.length > 0 && visibleSteps.length === thinkingSteps.length) {
      // Mark as completed when all steps are visible
      const timer = setTimeout(() => {
        setIsCompleted(true);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [thinkingSteps, visibleSteps.length]);

  useEffect(() => {
    // Update current phase based on latest step
    if (thinkingSteps.length > 0) {
      const latestStep = thinkingSteps[thinkingSteps.length - 1];
      if (latestStep.type === 'phase_start') {
        setCurrentPhase(latestStep.phase);
      }
    }
  }, [thinkingSteps]);

  // Show more steps when completed, fewer when actively thinking
  const stepsToShow = isCompleted ? 8 : 4;
  const recentSteps = visibleSteps.slice(-stepsToShow);
  
  return (
    <div className={`thinking-indicator ${isCompleted ? 'completed' : 'active'}`}>
      <div className="thinking-content">
        <div className={`thinking-spinner ${isCompleted ? 'completed' : ''}`}>
          {isCompleted && <span className="completion-check">✓</span>}
        </div>
        <div className="thinking-details">
          {/* Current main thinking message */}
          <span className="thinking-text">
            {isCompleted ? 'Thinking process completed' : message}
          </span>
          
          {/* Current phase indicator */}
          {currentPhase && !isCompleted && (
            <div className="current-phase">
              <strong>Phase: {currentPhase.replace(/_/g, ' ')}</strong>
            </div>
          )}

          {/* Summary when completed */}
          {isCompleted && (
            <div className="completion-summary">
              <strong>Completed {thinkingSteps.length} thinking steps</strong>
            </div>
          )}
          
          {/* Show thinking steps with slower reveal */}
          {recentSteps.length > 0 && (
            <div className="thinking-steps">
              {recentSteps.map((step, index) => {
                const stepKey = visibleSteps.indexOf(step);
                return (
                  <div 
                    key={stepKey} 
                    className={`thinking-step ${step.type === 'phase_start' ? 'phase-step' : 'detail-step'}`}
                    style={{ 
                      animation: `fadeInUp 0.6s ease-out both`,
                      animationDelay: `${index * 0.2}s`
                    }}
                  >
                    {step.type === 'phase_start' && (
                      <div className="phase-start">
                        <div className="phase-icon">🔍</div>
                        <div className="phase-content">
                          <strong>{step.phase.replace(/_/g, ' ')}</strong>
                          <p>{step.description}</p>
                        </div>
                      </div>
                    )}
                    {step.type === 'thinking_detail' && (
                      <div className="thinking-detail">
                        <div className="thinking-bullet">•</div>
                        <div className="thinking-content">
                          <div className="thinking-paragraph">{step.thinking}</div>
                          {step.result && (
                            <div className="thinking-result">
                              ✓ {typeof step.result === 'string' ? step.result : JSON.stringify(step.result)}
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    <div className="thinking-timestamp">
                      {new Date(step.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          
          {/* Progress indicator - different when completed */}
          <div className="thinking-progress">
            <div className="progress-dots">
              <span className={`dot ${isCompleted ? 'completed' : 'active'}`}></span>
              <span className={`dot ${isCompleted ? 'completed' : 'active'}`}></span>
              <span className={`dot ${isCompleted ? 'completed' : ''}`}></span>
            </div>
            {isCompleted && (
              <div className="completion-text">Analysis complete</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};