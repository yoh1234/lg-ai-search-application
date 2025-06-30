from openai import AsyncOpenAI
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from .models import Product

class LLMService:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def analyze_query(self, query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """쿼리 분석 - 의도, 엔티티, 필터 추출"""
        
        # 대화 히스토리를 컨텍스트로 추가
        context = ""
        if chat_history:
            recent_history = chat_history[-6:]  # 최근 3턴만
            context = "\n이전 대화:\n" + "\n".join([
                f"{msg['role']}: {msg['content']}" for msg in recent_history
            ])
        
        system_prompt = f"""
        Analyze the user's LG product search query.
        
        {context}
        
        Extract the following information as JSON:
        {{
            "intent": "product_search|price_comparison|feature_inquiry|compatibility_check|sku_lookup",
            "product_types": ["tv", "monitor", "soundbar", "accessory"],
            "price_range": {{"min": 100, "max": 2000}},
            "size_range": {{"min": 32, "max": 85}},
            "features": ["gaming", "4k", "oled", "mount"],
            "sku_mentioned": "OLED65C4PUA",
            "is_comparison": false,
            "confidence": 0.9
        }}
        
        - SKU pattern: uppercase+numbers (e.g., OLED65C4PUA)
        - Price keywords: "under $500", "budget", "cheap", "expensive", etc.
        - Size: "55 inch", "big screen", "large", etc.
        - Comparison: "vs", "compare", "difference", etc.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            # JSON 추출
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "{" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                content = content[start:end]
                
            return json.loads(content)
            
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return {
                "intent": "product_search",
                "product_types": [],
                "price_range": {},
                "size_range": {},
                "features": [],
                "sku_mentioned": "",
                "is_comparison": False,
                "confidence": 0.5
            }
    
    async def generate_response(
        self, 
        query: str, 
        search_results: List[Product], 
        analysis: Dict[str, Any],
        chat_history: List[Dict[str, str]] = None
    ) -> str:
        """검색 결과 기반 응답 생성"""
        
        # 대화 히스토리 컨텍스트
        context = ""
        if chat_history:
            recent_history = chat_history[-4:]
            context = "\n이전 대화:\n" + "\n".join([
                f"{msg['role']}: {msg['content']}" for msg in recent_history
            ])
        
        # 검색 결과 요약
        results_summary = []
        for product in search_results[:5]:
            results_summary.append(
                f"- {product.product_name} ({product.sku}): "
                f"${product.price if product.price else 'N/A'}"
            )
        
        system_prompt = f"""
        Generate a helpful response for the user's LG product search.
        
        {context}
        
        Current query: {query}
        Search intent: {analysis.get('intent', 'unknown')}
        Results count: {len(search_results)}
        
        Search results:
        {chr(10).join(results_summary) if results_summary else "No results"}
        
        Response guidelines:
        1. Natural English response
        2. Brief summary of search results
        3. Recommendations matching user intent
        4. Additional filtering suggestions (if needed)
        5. Maintain conversation flow
        
        Keep it short and clear, 3-4 sentences maximum.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"검색 결과에 대해 설명해주세요: {query}"}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Response generation error: {e}")
            return f"'{query}'에 대한 검색 결과 {len(search_results)}개를 찾았습니다."
