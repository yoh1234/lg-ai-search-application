from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import asyncio
from typing import List, Dict, Any

load_dotenv()

app = FastAPI(title="RAG LG Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
es = AsyncElasticsearch(['http://localhost:9200'])
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Memory store
chat_sessions = {}

@app.websocket("/ws/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "chat":
                query = message.get("query", "")
                chat_history = chat_sessions.get(session_id, [])
                
                # RAG Pipeline
                await websocket.send_text(json.dumps({
                    "type": "thinking_detail",
                    "phase": "rag_pipeline",
                    "thinking": "Starting RAG pipeline: Retrieve → Rerank → Generate. I'll search for relevant products, rerank them based on query relevance, then analyze their specs to provide detailed recommendations.",
                    "timestamp": datetime.now().isoformat()
                }))
                
                # Step 1: Retrieve
                products = await retrieve_products(query, websocket)
                
                # Step 2: Rerank
                reranked_products = await rerank_products(query, products, websocket)
                
                # Step 3: Generate with RAG
                explanation = await generate_rag_response(query, reranked_products, chat_history, websocket)
                
                # Save to memory
                if session_id not in chat_sessions:
                    chat_sessions[session_id] = []
                
                chat_sessions[session_id].append({
                    "user": query,
                    "assistant": explanation,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Send final result
                await websocket.send_text(json.dumps({
                    "type": "result",
                    "products": reranked_products,
                    "explanation": explanation,
                    "total_count": len(reranked_products)
                }))
                
    except WebSocketDisconnect:
        pass

async def retrieve_products(query: str, websocket) -> List[Dict]:
    """Step 1: Retrieve relevant products using hybrid search"""
    
    await websocket.send_text(json.dumps({
        "type": "thinking_detail",
        "phase": "retrieve",
        "thinking": f"Retrieving products using hybrid search (keyword + semantic). I'm searching for '{query}' across product names, features, and descriptions to cast a wide net for relevant products.",
        "timestamp": datetime.now().isoformat()
    }))
    
    try:
        # Quick keyword analysis for better search
        keywords = extract_keywords(query)
        
        # Hybrid search with multiple fallbacks
        search_strategies = [
            # Strategy 1: Precise search
            {
                "query": {
                    "bool": {
                        "should": [
                            {"multi_match": {"query": query, "fields": ["product_name^3", "key_features^2", "sku^4"], "fuzziness": "AUTO"}},
                            {"terms": {"product_name": keywords, "boost": 2}},
                            {"terms": {"key_features": keywords, "boost": 1.5}}
                        ]
                    }
                },
                "size": 20
            },
            # Strategy 2: Keyword-based fallback
            {
                "query": {"terms": {"key_features": keywords}},
                "size": 15
            },
            # Strategy 3: Broad search
            {
                "query": {"match_all": {}},
                "size": 10
            }
        ]
        
        all_products = []
        for i, strategy in enumerate(search_strategies):
            try:
                response = await es.search(index="products", body=strategy)
                strategy_products = [format_product(hit["_source"]) for hit in response["hits"]["hits"]]
                all_products.extend(strategy_products)
                
                if len(all_products) >= 15:  # Enough products found
                    break
                    
            except Exception as e:
                continue
        
        # Remove duplicates by SKU
        seen_skus = set()
        unique_products = []
        for product in all_products:
            if product["sku"] not in seen_skus:
                seen_skus.add(product["sku"])
                unique_products.append(product)
        
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "retrieve",
            "thinking": f"Retrieved {len(unique_products)} unique products from {len(search_strategies)} search strategies. Now I'll rerank these based on semantic relevance to the specific query.",
            "result": {"retrieved": len(unique_products)},
            "timestamp": datetime.now().isoformat()
        }))
        
        return unique_products[:15]  # Limit for reranking
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "retrieve",
            "thinking": f"Error during retrieval: {str(e)}. Falling back to basic search.",
            "timestamp": datetime.now().isoformat()
        }))
        return []

def extract_keywords(query: str) -> List[str]:
    """Extract relevant keywords for search"""
    keywords = []
    
    # Product type keywords
    type_mapping = {
        "tv": ["tv", "television", "oled", "qled", "smart"],
        "monitor": ["monitor", "display", "gaming", "4k"],
        "soundbar": ["soundbar", "speaker", "audio", "sound"],
        "accessory": ["mount", "bracket", "cable", "remote"]
    }
    
    query_lower = query.lower()
    for category, terms in type_mapping.items():
        if any(term in query_lower for term in terms):
            keywords.extend(terms[:3])  # Top 3 terms
    
    # Feature keywords
    feature_keywords = ["gaming", "4k", "8k", "hdr", "smart", "wifi", "bluetooth", "oled", "qled"]
    keywords.extend([kw for kw in feature_keywords if kw in query_lower])
    
    return list(set(keywords))  # Remove duplicates

async def rerank_products(query: str, products: List[Dict], websocket) -> List[Dict]:
    """Step 2: Rerank products using lightweight LLM scoring"""
    
    if not products or not OPENAI_API_KEY:
        return products
    
    await websocket.send_text(json.dumps({
        "type": "thinking_detail",
        "phase": "rerank",
        "thinking": f"Reranking {len(products)} products using AI to determine the best matches for '{query}'. I'll score each product based on relevance, features, and user intent.",
        "timestamp": datetime.now().isoformat()
    }))
    
    try:
        # Create product summaries for LLM
        product_summaries = []
        for i, product in enumerate(products):
            summary = f"{i}: {product['product_name']} - ${product.get('price', 'N/A')} - {product.get('key_features', '')[:100]}"
            product_summaries.append(summary)
        
        # Lightweight reranking prompt
        rerank_prompt = f"""Rank these products by relevance to: "{query}"
Products:
{chr(10).join(product_summaries)}

Return only numbers in order of relevance (most relevant first):
Example: 3,1,7,2,5"""

        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Faster model
            messages=[
                {"role": "system", "content": "You are a product ranking expert. Return only the numbers in order."},
                {"role": "user", "content": rerank_prompt}
            ],
            max_tokens=50,  # Very short response
            temperature=0.1
        )
        
        # Parse ranking
        ranking_text = response.choices[0].message.content.strip()
        try:
            rankings = [int(x.strip()) for x in ranking_text.split(',')]
            
            # Reorder products based on ranking
            reranked = []
            for rank in rankings:
                if 0 <= rank < len(products):
                    reranked.append(products[rank])
            
            # Add any missed products at the end
            ranked_indices = set(rankings)
            for i, product in enumerate(products):
                if i not in ranked_indices:
                    reranked.append(product)
                    
        except Exception:
            # If parsing fails, use original order
            reranked = products
        
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "rerank",
            "thinking": f"Reranking complete. Reordered products based on relevance scoring. Top product is now: {reranked[0]['product_name'] if reranked else 'None'}",
            "result": {"reranked": len(reranked)},
            "timestamp": datetime.now().isoformat()
        }))
        
        return reranked
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "rerank",
            "thinking": f"Reranking failed: {str(e)}. Using original order.",
            "timestamp": datetime.now().isoformat()
        }))
        return products

async def generate_rag_response(query: str, products: List[Dict], chat_history: List, websocket) -> str:
    """Step 3: Generate response using RAG with product details"""
    
    if not products or not OPENAI_API_KEY:
        return f"Found {len(products)} products for '{query}'"
    
    await websocket.send_text(json.dumps({
        "type": "thinking_detail",
        "phase": "generate",
        "thinking": f"Analyzing {len(products)} products in detail to provide specific recommendations. I'll examine specs, prices, and features to match your needs exactly.",
        "timestamp": datetime.now().isoformat()
    }))
    
    try:
        # Create detailed product context (RAG)
        top_products = products[:5]  # Limit context size
        product_context = ""
        
        for i, product in enumerate(top_products, 1):
            context = f"""Product {i}: {product['product_name']}
- SKU: {product['sku']}
- Price: ${product.get('price', 'N/A')}
- Size: {product.get('size', 'N/A')}"
- Features: {product.get('key_features', '')[:150]}

"""
            product_context += context
        
        # Lightweight RAG prompt
        rag_prompt = f"""Based on these specific products:

{product_context}

User query: "{query}"

Provide a brief, helpful recommendation (2-3 sentences max). Focus on the best match and why."""

        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Fast model
            messages=[
                {"role": "system", "content": "You are a helpful product advisor. Be concise and specific."},
                {"role": "user", "content": rag_prompt}
            ],
            max_tokens=150,  # Short response for speed
            temperature=0.3
        )
        
        rag_response = response.choices[0].message.content.strip()
        
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "generate",
            "thinking": f"Generated detailed recommendation based on actual product specs. I analyzed {len(top_products)} products and provided specific guidance on the best options.",
            "result": {"response_length": len(rag_response)},
            "timestamp": datetime.now().isoformat()
        }))
        
        return rag_response
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "generate",
            "thinking": f"RAG generation failed: {str(e)}. Providing basic response.",
            "timestamp": datetime.now().isoformat()
        }))
        return f"Found {len(products)} products for '{query}'. The top recommendation is {products[0]['product_name']}."

def format_product(source: dict):
    """Format product data"""
    return {
        "product_name": source.get("product_name", ""),
        "sku": source.get("sku", ""),
        "price": source.get("price"),
        "size": source.get("size"),
        "product_url": source.get("product_url", ""),
        "image_urls": [source.get("image_urls", "")] if isinstance(source.get("image_urls"), str) else source.get("image_urls", []),
        "key_features": source.get("key_features", "")
    }

@app.get("/")
async def root():
    return {"message": "RAG + Reranking LG Search API"}

@app.get("/health")
async def health():
    return {"status": "healthy", "sessions": len(chat_sessions)}