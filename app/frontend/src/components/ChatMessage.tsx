import React, { useState } from 'react';
import { ChatMessage as ChatMessageType } from '../types';
import { ProductCard } from './ProductCard';

interface ChatMessageProps {
  message: ChatMessageType;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const [showThinking, setShowThinking] = useState(false);
  const isUser = message.role === 'user';
  
  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-content">
        <p className="message-text">{message.content}</p>
        
        {/* Show thinking process for assistant messages */}
        {!isUser && message.thinkingSteps && message.thinkingSteps.length > 0 && (
          <div className="thinking-summary">
            <button 
              className="thinking-toggle"
              onClick={() => setShowThinking(!showThinking)}
            >
              {showThinking ? '🔽' : '🔼'} View AI Thinking Process ({message.thinkingSteps.length} steps)
            </button>
            
            {showThinking && (
              <div className="thinking-details-preserved">
                <div className="thinking-header">
                  <span className="thinking-completion-badge">✓ Analysis Complete</span>
                </div>
                
                <div className="thinking-steps-preserved">
                  {message.thinkingSteps.map((step, index) => (
                    <div 
                      key={index}
                      className={`thinking-step-preserved ${step.type === 'phase_start' ? 'phase-step' : 'detail-step'}`}
                    >
                      {step.type === 'phase_start' && (
                        <div className="phase-start-preserved">
                          <div className="phase-icon">🔍</div>
                          <div className="phase-content">
                            <strong>{step.phase.replace(/_/g, ' ')}</strong>
                            <p>{step.description}</p>
                          </div>
                        </div>
                      )}
                      {step.type === 'thinking_detail' && (
                        <div className="thinking-detail-preserved">
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
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        
        {message.products && message.products.length > 0 && (
          <div className="products-grid">
            {message.products.map((product, index) => (
              <ProductCard key={product.sku || index} product={product} />
            ))}
          </div>
        )}
        
        <div className="message-timestamp">
          {new Date(message.timestamp).toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
};