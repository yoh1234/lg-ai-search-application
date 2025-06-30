import React from 'react';
import { ChatMessage as ChatMessageType } from '../types';
import { ProductCard } from './ProductCard';

interface ChatMessageProps {
  message: ChatMessageType;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  
  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-content">
        <p className="message-text">{message.content}</p>
        
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