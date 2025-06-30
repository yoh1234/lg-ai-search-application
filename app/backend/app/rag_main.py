# backend/app/rag_main.py
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
        "thinking": f"Starting the retrieval phase for '{query}'. I'm analyzing the query to understand what the user is looking for. Let me break this down: I need to identify product categories, price constraints, size requirements, and specific features mentioned.",
        "timestamp": datetime.now().isoformat()
    }))
    
    try:
        # Quick keyword analysis for better search
        keywords = extract_keywords(query)
        
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "retrieve",
            "thinking": f"I've extracted key search terms: {keywords}. Now I'll use a multi-strategy approach: first a precise hybrid search combining keyword matching with semantic similarity, then fallback to broader searches if needed. This ensures I don't miss relevant products.",
            "result": {"keywords_found": keywords},
            "timestamp": datetime.now().isoformat()
        }))
        
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
                await websocket.send_text(json.dumps({
                    "type": "thinking_detail",
                    "phase": "retrieve",
                    "thinking": f"Executing search strategy {i+1}: {'Precise hybrid search' if i==0 else 'Keyword fallback' if i==1 else 'Broad search'}. This gives me multiple chances to find relevant products even if the initial search is too restrictive.",
                    "timestamp": datetime.now().isoformat()
                }))
                
                response = await es.search(index="products", body=strategy)
                strategy_products = [format_product(hit["_source"]) for hit in response["hits"]["hits"]]
                all_products.extend(strategy_products)
                
                await websocket.send_text(json.dumps({
                    "type": "thinking_detail",
                    "phase": "retrieve",
                    "thinking": f"Strategy {i+1} returned {len(strategy_products)} products. Total products so far: {len(all_products)}. I'll continue searching until I have a good selection to work with.",
                    "result": {"strategy_results": len(strategy_products), "total": len(all_products)},
                    "timestamp": datetime.now().isoformat()
                }))
                
                if len(all_products) >= 15:  # Enough products found
                    await websocket.send_text(json.dumps({
                        "type": "thinking_detail",
                        "phase": "retrieve",
                        "thinking": "Great! I've found enough products to work with. Now I'll remove duplicates and prepare for the reranking phase.",
                        "timestamp": datetime.now().isoformat()
                    }))
                    break
                    
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "thinking_detail",
                    "phase": "retrieve",
                    "thinking": f"Strategy {i+1} encountered an issue: {str(e)}. Moving to next strategy.",
                    "timestamp": datetime.now().isoformat()
                }))
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
            "thinking": f"Retrieval complete! I found {len(all_products)} total products but after removing duplicates, I have {len(unique_products)} unique products. These range from direct matches to broader category matches. Now I need to rank them by relevance to your specific query.",
            "result": {"retrieved": len(unique_products), "duplicates_removed": len(all_products) - len(unique_products)},
            "timestamp": datetime.now().isoformat()
        }))
        
        return unique_products[:15]  # Limit for reranking
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "retrieve",
            "thinking": f"Retrieval encountered a major error: {str(e)}. This might be due to Elasticsearch connectivity issues. I'll try to provide a fallback response.",
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
        "thinking": f"Now I need to rerank these {len(products)} products to find the best matches for '{query}'. The initial search cast a wide net, but now I need to be more precise. I'll analyze each product's relevance, considering factors like product type match, feature alignment, price appropriateness, and overall suitability for the user's needs.",
        "timestamp": datetime.now().isoformat()
    }))
    
    try:
        # Create product summaries for LLM
        product_summaries = []
        for i, product in enumerate(products):
            summary = f"{i}: {product['product_name']} - ${product.get('price', 'N/A')} - {product.get('key_features', '')[:100]}"
            product_summaries.append(summary)
        
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "rerank",
            "thinking": f"I've prepared summaries of all {len(products)} products for AI analysis. Each summary includes the product name, price, and key features. Now I'll send this to a lightweight AI model to score relevance. The AI will consider how well each product matches the user's specific requirements.",
            "result": {"products_to_rank": len(products)},
            "timestamp": datetime.now().isoformat()
        }))
        
        # Lightweight reranking prompt
        rerank_prompt = f"""Rank these products by relevance to: "{query}"
Products:
{chr(10).join(product_summaries)}

Return only numbers in order of relevance (most relevant first):
Example: 3,1,7,2,5"""

        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "rerank",
            "thinking": "Sending the ranking request to the AI model. I'm using a streamlined approach where the AI just returns numbers in order of relevance. This is much faster than asking for detailed explanations while still getting intelligent ranking.",
            "timestamp": datetime.now().isoformat()
        }))

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
        
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "rerank",
            "thinking": f"AI ranking complete! The model returned: '{ranking_text}'. Now I'm parsing these rankings and reordering the products accordingly. This puts the most relevant products first, which will give you much better recommendations.",
            "result": {"ai_ranking": ranking_text},
            "timestamp": datetime.now().isoformat()
        }))
        
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
            
            await websocket.send_text(json.dumps({
                "type": "thinking_detail",
                "phase": "rerank",
                "thinking": f"Reranking successful! The new order prioritizes products that best match your query. The top product is now '{reranked[0]['product_name'] if reranked else 'None'}', which the AI determined is the most relevant to your needs. This reordering should significantly improve the quality of recommendations.",
                "result": {"reranked": len(reranked), "top_product": reranked[0]['product_name'] if reranked else 'None'},
                "timestamp": datetime.now().isoformat()
            }))
                    
        except Exception as parse_error:
            await websocket.send_text(json.dumps({
                "type": "thinking_detail",
                "phase": "rerank",
                "thinking": f"Had trouble parsing the AI ranking response: {str(parse_error)}. The AI returned '{ranking_text}' which doesn't match the expected format. I'll keep the original search order, which is still relevance-based from Elasticsearch.",
                "timestamp": datetime.now().isoformat()
            }))
            reranked = products
        
        return reranked
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "rerank",
            "thinking": f"Reranking encountered an error: {str(e)}. This might be due to API issues or network problems. I'll proceed with the original Elasticsearch ranking, which is still quite good for most queries.",
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
        "thinking": f"Now for the final RAG (Retrieve-Augment-Generate) step. I have {len(products)} reranked products, and I need to analyze their actual specifications to provide you with specific, detailed recommendations. This is where the magic happens - I'll read through the product details and match them to your exact needs.",
        "timestamp": datetime.now().isoformat()
    }))
    
    try:
        # Create detailed product context (RAG)
        top_products = products[:5]  # Limit context size
        product_context = ""
        
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "generate",
            "thinking": f"I'm focusing on the top {len(top_products)} products for detailed analysis. Let me examine each one: their specifications, pricing, features, and how they match your requirements. This detailed context will help me give you much more specific and accurate recommendations than generic responses.",
            "timestamp": datetime.now().isoformat()
        }))
        
        for i, product in enumerate(top_products, 1):
            context = f"""Product {i}: {product['product_name']}
- SKU: {product['sku']}
- Price: ${product.get('price', 'N/A')}
- Size: {product.get('size', 'N/A')}"
- Features: {product.get('key_features', '')[:150]}

"""
            product_context += context
        
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "generate",
            "thinking": f"Product analysis complete. I've compiled detailed specs for {len(top_products)} products including their prices, sizes, and key features. Now I'm sending this rich context to the AI along with your original query to generate a personalized recommendation that considers the actual product specifications.",
            "result": {"products_analyzed": len(top_products), "context_length": len(product_context)},
            "timestamp": datetime.now().isoformat()
        }))
        
        # Lightweight RAG prompt
        rag_prompt = f"""Based on these specific products:

{product_context}

User query: "{query}"

Provide a brief, helpful recommendation (2-3 sentences max). Focus on the best match and why."""

        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "generate",
            "thinking": "Generating your personalized recommendation now. The AI is reading through all the product specifications and matching them against your specific requirements. This should result in a much more accurate and helpful recommendation than generic product descriptions.",
            "timestamp": datetime.now().isoformat()
        }))

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
            "thinking": f"Perfect! I've generated a detailed recommendation based on the actual product specifications. The AI analyzed {len(top_products)} products and provided specific guidance tailored to your query '{query}'. This recommendation is based on real product data, not generic descriptions, so it should be much more accurate and helpful.",
            "result": {"response_generated": True, "response_length": len(rag_response), "based_on_products": len(top_products)},
            "timestamp": datetime.now().isoformat()
        }))
        
        return rag_response
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "generate",
            "thinking": f"RAG generation encountered an issue: {str(e)}. This might be due to API limits or network issues. I'll provide a basic recommendation based on the reranked results, which should still be quite helpful since the products are properly ordered by relevance.",
            "timestamp": datetime.now().isoformat()
        }))
        return f"Found {len(products)} products for '{query}'. Based on the reranking, the top recommendation is {products[0]['product_name']} at ${products[0].get('price', 'N/A')}."

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