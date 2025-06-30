from typing import Dict, List, Any
from .models import ChatSession, ChatMessage, Product, SearchResult
from .elasticsearch_service import ElasticsearchService
from .llm_service import LLMService
from datetime import datetime

class ChatService:
    def __init__(self, openai_api_key: str, es_url: str = "http://localhost:9200"):
        self.es_service = ElasticsearchService(es_url)
        self.llm_service = LLMService(openai_api_key)
        self.sessions: Dict[str, ChatSession] = {}  # 메모리에 세션 저장
    
    def get_session(self, session_id: str) -> ChatSession:
        """세션 조회 또는 생성"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(session_id=session_id)
        return self.sessions[session_id]
    
    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """대화 히스토리 가져오기"""
        session = self.get_session(session_id)
        return [
            {"role": msg.role, "content": msg.content}
            for msg in session.messages[-10:]  # 최근 10개만
        ]
    
    async def chat(self, session_id: str, user_message: str) -> SearchResult:
        """메인 챗봇 함수"""
        session = self.get_session(session_id)
        
        # 사용자 메시지 저장
        session.messages.append(ChatMessage(
            role="user",
            content=user_message,
            timestamp=datetime.now()
        ))
        
        # 대화 히스토리 가져오기
        chat_history = self.get_chat_history(session_id)
        
        # 1. 쿼리 분석
        analysis = await self.llm_service.analyze_query(user_message, chat_history)
        
        # 2. 검색 실행
        search_results = await self._execute_search(user_message, analysis)
        
        # 3. 응답 생성
        explanation = await self.llm_service.generate_response(
            user_message, search_results, analysis, chat_history
        )
        
        # 어시스턴트 응답 저장
        session.messages.append(ChatMessage(
            role="assistant",
            content=explanation,
            timestamp=datetime.now()
        ))
        
        return SearchResult(
            products=search_results,
            total_count=len(search_results),
            explanation=explanation
        )
    
    async def _execute_search(self, query: str, analysis: Dict[str, Any]) -> List[Product]:
        """검색 실행"""
        
        # SKU 직접 검색
        if analysis.get("sku_mentioned"):
            sku_result = await self.es_service.get_product_by_sku(analysis["sku_mentioned"])
            if sku_result:
                return [self._convert_to_product(sku_result["_source"])]
        
        # 일반 검색
        price_range = analysis.get("price_range", {})
        size_range = analysis.get("size_range", {})
        
        es_results = await self.es_service.search_products(
            query=query,
            price_min=price_range.get("min"),
            price_max=price_range.get("max"),
            size_min=size_range.get("min"),
            size_max=size_range.get("max"),
            limit=20
        )
        
        products = []
        for hit in es_results["hits"]:
            products.append(self._convert_to_product(hit["_source"]))
        
        # 간단한 후처리
        if analysis.get("intent") == "price_comparison":
            products.sort(key=lambda x: x.price or float('inf'))
        
        return products
    
    def _convert_to_product(self, source: Dict[str, Any]) -> Product:
        """Elasticsearch 결과를 Product 모델로 변환"""
        image_urls = source.get("image_urls", [])
        if isinstance(image_urls, str):
            image_urls = [image_urls]
        
        return Product(
            product_name=source.get("product_name", ""),
            sku=source.get("sku", ""),
            price=source.get("price"),
            size=source.get("size"),
            product_url=source.get("product_url", ""),
            image_urls=image_urls,
            key_features=source.get("key_features", "")
        )