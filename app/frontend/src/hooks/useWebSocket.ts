import { useState, useRef, useEffect, useCallback } from 'react';
import { ChatMessage, Product, ThinkingStep } from '../types';

export const useWebSocket = (sessionId: string) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [currentThinking, setCurrentThinking] = useState<string>('');
  
  const ws = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    ws.current = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
    
    ws.current.onopen = () => {
      setIsConnected(true);
    };
    
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'thinking') {
        setIsThinking(true);
        setCurrentThinking(data.description);
      } else if (data.type === 'result') {
        setIsThinking(false);
        setCurrentThinking('');
        
        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: data.explanation,
          timestamp: new Date().toISOString(),
          products: data.products
        };
        
        setMessages(prev => [...prev, assistantMessage]);
      }
    };
    
    ws.current.onclose = () => {
      setIsConnected(false);
      setTimeout(connect, 3000); // reconnect
    };
  }, [sessionId]);

  useEffect(() => {
    connect();
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [connect]);

  const sendMessage = useCallback((message: string) => {
    if (ws.current && isConnected) {
      const userMessage: ChatMessage = {
        role: 'user',
        content: message,
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, userMessage]);
      
      ws.current.send(JSON.stringify({
        type: 'chat',
        query: message
      }));
    }
  }, [isConnected]);

  return {
    messages,
    isConnected,
    isThinking,
    currentThinking,
    sendMessage
  };
};