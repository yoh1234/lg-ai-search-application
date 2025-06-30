// frontend/src/hooks/useWebSocket.ts
import { useState, useRef, useEffect, useCallback } from 'react';
import { ChatMessage, Product, ThinkingStep } from '../types';

export const useWebSocket = (sessionId: string) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [currentThinking, setCurrentThinking] = useState<string>('');
  const [thinkingSteps, setThinkingSteps] = useState<any[]>([]);
  
  const ws = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    ws.current = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
    
    ws.current.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    };
    
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('Received:', data);
      
      if (data.type === 'thinking_start') {
        setIsThinking(true);
        setCurrentThinking(data.description);
        setThinkingSteps(prev => [...prev, {
          type: 'phase_start',
          phase: data.phase,
          description: data.description,
          timestamp: data.timestamp
        }]);
      } 
      else if (data.type === 'thinking_detail') {
        setCurrentThinking(data.thinking);
        setThinkingSteps(prev => [...prev, {
          type: 'thinking_detail',
          phase: data.phase,
          thinking: data.thinking,
          result: data.result,
          timestamp: data.timestamp
        }]);
      }
      else if (data.type === 'result') {
        setIsThinking(false);
        setCurrentThinking('');
        
        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: data.explanation,
          timestamp: new Date().toISOString(),
          products: data.products,
          thinkingSteps: [...thinkingSteps] // Save thinking steps with the message
        };
        
        setMessages(prev => [...prev, assistantMessage]);
        setThinkingSteps([]); // Clear for next query
      }
    };
    
    ws.current.onclose = () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
      setTimeout(connect, 3000);
    };
    
    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
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
      setThinkingSteps([]); // Clear previous thinking steps
      
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
    thinkingSteps,
    sendMessage
  };
};