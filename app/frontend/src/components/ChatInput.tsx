import React, { useState } from 'react';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSend, disabled }) => {
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSend(input.trim());
      setInput('');
    }
  };

  const examples = [
    "I need a gaming monitor under $500",
    "Best OLED TV for movies",
    "OLED65C4PUA",
    "Soundbar compatible with LG TV",
    "65 inch TV comparison"
  ];

  return (
    <div className="chat-input-container">
      <form onSubmit={handleSubmit} className="chat-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about LG products..."
          className="chat-input"
          disabled={disabled}
        />
        <button
          type="submit"
          disabled={!input.trim() || disabled}
          className="send-button"
        >
          Send
        </button>
      </form>
      
      <div className="examples-container">
        <span className="examples-label">Example queries:</span>
        {examples.map((example, index) => (
          <button
            key={index}
            onClick={() => setInput(example)}
            className="example-button"
            disabled={disabled}
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
};
