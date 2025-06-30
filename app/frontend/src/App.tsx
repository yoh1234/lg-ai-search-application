import React, { useEffect, useRef } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { ThinkingIndicator } from './components/ThinkingIndicator';
import './index.css';

function App() {
  const sessionId = 'user-session-1';
  const { messages, isConnected, isThinking, currentThinking, thinkingSteps, sendMessage } = useWebSocket(sessionId);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isThinking, thinkingSteps]);

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>LG Product Search with AI Thinking</h1>
        <div className="connection-status">
          <div className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></div>
          <span className="status-text">
            {isConnected ? 'Connected' : 'Connecting...'}
          </span>
        </div>
      </header>

      {/* Messages */}
      <div className="messages-container">
        {messages.length === 0 && (
          <div className="welcome-message">
            <h2>Welcome to LG Product Search with AI Thinking!</h2>
            <p>Ask me anything about LG TVs, monitors, soundbars, and accessories.</p>
            <p>You'll see my detailed thinking process in real-time!</p>
          </div>
        )}
        
        {messages.map((message, index) => (
          <ChatMessage key={index} message={message} />
        ))}
        
        {isThinking && <ThinkingIndicator message={currentThinking} thinkingSteps={thinkingSteps} />}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <ChatInput onSend={sendMessage} disabled={!isConnected || isThinking} />
    </div>
  );
}

export default App;