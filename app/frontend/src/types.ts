export interface Product {
  product_name: string;
  sku: string;
  price?: number;
  size?: number;
  product_url: string;
  image_urls: string[];
  key_features?: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  products?: Product[];
}

export interface ThinkingStep {
  type: 'thinking';
  step: string;
  description: string;
}