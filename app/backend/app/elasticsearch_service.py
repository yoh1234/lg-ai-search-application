from elasticsearch import AsyncElasticsearch
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class ElasticsearchService:
    def __init__(self, es_url: str = "http://localhost:9200"):
        self.client = AsyncElasticsearch([es_url])
        self.index_name = "products"
        # 임베딩 모델 로드 (벡터 검색용)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def get_query_embedding(self, query: str) -> List[float]:
        """쿼리를 벡터로 변환"""
        try:
            embedding = self.embedding_model.encode(query)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [0.0] * 384
    
    async def hybrid_search(
        self,
        query: str,
        price_max: Optional[float] = None,
        price_min: Optional[float] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        limit: int = 20,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7
    ) -> Dict[str, Any]:
        """하이브리드 검색 (키워드 + 벡터)"""
        try:
            # 쿼리를 벡터로 변환
            query_vector = self.get_query_embedding(query)
            
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            # 키워드 검색
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["product_name^2", "key_features", "sku^3"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO",
                                    "boost": keyword_weight
                                }
                            },
                            # 벡터 검색
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'combined_embedding') + 1.0",
                                        "params": {"query_vector": query_vector}
                                    },
                                    "boost": vector_weight
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
            logger.error(f"Hybrid search error: {e}")
            # 실패시 키워드 검색으로 폴백
            return await self.search_products(query, price_max, price_min, size_min, size_max, limit)
    
    async def search_products(
        self,
        query: str,
        price_max: Optional[float] = None,
        price_min: Optional[float] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """기본 키워드 검색 (폴백용)"""
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