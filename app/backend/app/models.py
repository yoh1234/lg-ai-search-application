from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class Product(BaseModel):
    product_name: str
    sku: str
    price: Optional[float] = None
    size: Optional[int] = None
    product_url: str
    image_urls: List[str] = []
    key_features: Optional[str] = None

class SearchResult(BaseModel):
    products: List[Product]
    total_count: int
    explanation: str

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = datetime.now()

class ChatSession(BaseModel):
    session_id: str
    messages: List[ChatMessage] = []
    created_at: datetime = datetime.now()

# backend/app/elasticsearch_service.py
from elasticsearch import AsyncElasticsearch
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ElasticsearchService:
    def __init__(self, es_url: str = "http://localhost:9200"):
        self.client = AsyncElasticsearch([es_url])
        self.index_name = "lg_products"
    
    async def search_products(
        self,
        query: str,
        price_max: Optional[float] = None,
        price_min: Optional[float] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """하이브리드 검색 (키워드 + 벡터)"""
        try:
            # 기본 쿼리
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["product_name^2", "key_features", "sku^3"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ],
                        "filter": []
                    }
                },
                "size": limit,
                "sort": [{"_score": {"order": "desc"}}]
            }
            
            # 필터 추가
            filters = search_body["query"]["bool"]["filter"]
            
            if price_min or price_max:
                price_filter = {"range": {"price": {}}}
                if price_min:
                    price_filter["range"]["price"]["gte"] = price_min
                if price_max:
                    price_filter["range"]["price"]["lte"] = price_max
                filters.append(price_filter)
            
            if size_min or size_max:
                size_filter = {"range": {"size": {}}}
                if size_min:
                    size_filter["range"]["size"]["gte"] = size_min
                if size_max:
                    size_filter["range"]["size"]["lte"] = size_max
                filters.append(size_filter)
            
            response = await self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            return {
                "hits": response["hits"]["hits"],
                "total": response["hits"]["total"]["value"]
            }
            
        except Exception as e:
            logger.error(f"Elasticsearch error: {e}")
            return {"hits": [], "total": 0}

    async def get_product_by_sku(self, sku: str) -> Optional[Dict[str, Any]]:
        """SKU로 제품 조회"""
        try:
            response = await self.client.search(
                index=self.index_name,
                body={
                    "query": {"term": {"sku": sku}},
                    "size": 1
                }
            )
            hits = response["hits"]["hits"]
            return hits[0] if hits else None
        except Exception as e:
            logger.error(f"SKU search error: {e}")
            return None