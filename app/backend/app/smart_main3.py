# backend/app/smart_main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI
import json
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

load_dotenv()

app = FastAPI(title="Smart Search API")

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
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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
                query = message.get("query", "").strip()
                chat_history = chat_sessions.get(session_id, [])
                
                await send_thinking(websocket, "starting", f"Processing query: '{query}'")
                
                # Step 1: Try simple keyword search first
                keyword_results = await simple_keyword_search(query, websocket)
                
                if keyword_results and len(keyword_results) >= 3:
                    # Success with keyword search
                    explanation = f"Found {len(keyword_results)} products using keyword search."
                    products = keyword_results
                    
                    await send_thinking(websocket, "keyword_success", 
                        f"Keyword search successful! Found {len(keyword_results)} results.")
                else:
                    # Go to LLM pipeline
                    await send_thinking(websocket, "llm_fallback", 
                        f"Keyword search insufficient. Using LLM pipeline...")
                    
                    products, explanation = await llm_pipeline(query, chat_history, websocket)
                
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
                    "products": products,
                    "explanation": explanation,
                    "total_count": len(products)
                }))
                
    except WebSocketDisconnect:
        pass

async def send_thinking(websocket: WebSocket, phase: str, thinking: str):
    """Send real-time thinking to frontend"""
    await websocket.send_text(json.dumps({
        "type": "thinking",
        "phase": phase,
        "thinking": thinking,
        "timestamp": datetime.now().isoformat()
    }))

async def simple_keyword_search(query: str, websocket) -> Optional[List[Dict]]:
    """Simple keyword search in SKU and product_name only"""
    
    await send_thinking(websocket, "keyword_search", 
        "Trying exact and fuzzy matching in SKU and product name...")
    
    try:
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        # Exact SKU match (highest priority)
                        {
                            "term": {
                                "sku": {
                                    "value": query.upper(),
                                    "boost": 20
                                }
                            }
                        },
                        {
                            "term": {
                                "sku": {
                                    "value": query,
                                    "boost": 20
                                }
                            }
                        },
                        # Exact product name phrase
                        {
                            "match_phrase": {
                                "product_name": {
                                    "query": query,
                                    "boost": 10
                                }
                            }
                        },
                        # Fuzzy SKU matching
                        {
                            "fuzzy": {
                                "sku": {
                                    "value": query.upper(),
                                    "fuzziness": 1,
                                    "boost": 15
                                }
                            }
                        },
                        # Fuzzy product name matching
                        {
                            "fuzzy": {
                                "product_name": {
                                    "value": query,
                                    "fuzziness": "AUTO",
                                    "boost": 8
                                }
                            }
                        },
                        # Wildcard matching for partial matches
                        {
                            "wildcard": {
                                "sku": {
                                    "value": f"*{query.upper()}*",
                                    "boost": 12
                                }
                            }
                        },
                        {
                            "wildcard": {
                                "product_name": {
                                    "value": f"*{query}*",
                                    "boost": 6
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": 15
        }
        
        response = await es.search(index="products", body=search_body)
        results = [format_product(hit["_source"]) for hit in response["hits"]["hits"] if hit["_score"] > 1.0]
        
        await send_thinking(websocket, "keyword_results", 
            f"Keyword search found {len(results)} results with good scores.")
        
        return results if results else None
        
    except Exception as e:
        await send_thinking(websocket, "keyword_error", f"Keyword search error: {str(e)}")
        return None

async def llm_pipeline(query: str, chat_history: List, websocket) -> tuple[List[Dict], str]:
    """LLM pipeline with support for single and multi-product queries"""
    
    if not OPENAI_API_KEY:
        return [], "LLM not available"
    
    # Step 1: Analyze query (single vs multi)
    await send_thinking(websocket, "llm_analyzing", 
        "Using LLM to understand query type and extract search parameters...")
    
    analysis = await analyze_query_with_llm(query, websocket)
    
    # Step 2: Execute search(es) based on query type
    if analysis.get("query_type") == "multi":
        # Multi-product query: execute multiple searches
        await send_thinking(websocket, "multi_search", 
            f"Executing {len(analysis.get('searches', []))} separate searches...")
        
        all_products = []
        for i, search_params in enumerate(analysis.get("searches", [])):
            await send_thinking(websocket, f"search_{i+1}", 
                f"Search {i+1}: {search_params.get('product_type')} with filters...")
            
            products = await comprehensive_search_with_strict_filtering(search_params, websocket)
            
            # If no products found for this search, try fallback
            if not products:
                await send_thinking(websocket, f"search_{i+1}_fallback", 
                    f"No results for search {i+1}, trying fallback...")
                
                # Create a mini-query for this specific search
                mini_query = f"{search_params.get('product_type', '')} {' '.join(search_params.get('search_terms', []))}"
                fallback_products = await vector_similarity_fallback(mini_query.strip(), websocket)
                
                # Filter fallback by product type if specified
                if search_params.get('product_type') and fallback_products:
                    fallback_products = [
                        p for p in fallback_products 
                        if p.get('product_type') == search_params.get('product_type')
                    ]
                
                products = fallback_products[:5]  # Limit fallback results
            
            # Tag products with search origin
            for product in products:
                product["search_origin"] = f"{search_params.get('product_type')}_search_{i+1}"
            
            all_products.extend(products)
        
        # Step 3: Rerank combined results
        await send_thinking(websocket, "multi_reranking", 
            f"Reranking combined results from {len(analysis.get('searches', []))} searches...")
        
        reranked_products = await rerank_multi_products(query, all_products, websocket)
        
    else:
        # Single product query: normal search
        await send_thinking(websocket, "single_search", 
            "Single product search with strict filtering...")
        
        products = await comprehensive_search_with_strict_filtering(analysis, websocket)
        reranked_products = products
    
    # Step 4: Generate explanation (this will now handle fallback internally)
    await send_thinking(websocket, "generating_explanation", 
        "Generating explanation based on results...")
    
    explanation = await generate_explanation(query, reranked_products, websocket)
    
    return reranked_products, explanation

async def comprehensive_search_with_strict_filtering(params: Dict, websocket) -> List[Dict]:
    """Comprehensive search with strict product type filtering"""
    
    search_terms = " ".join(params.get("search_terms", []))
    product_type = params.get("product_type")
    
    # Build search with STRICT product type filtering
    search_body = {
        "query": {
            "bool": {
                "must": [],  # Strict requirements
                "should": [],  # Scoring preferences
                "filter": []   # Hard filters
            }
        },
        "size": 20
    }
    
    # MANDATORY product type filter
    if product_type:
        await send_thinking(websocket, "strict_product_filter", 
            f"Applying STRICT product type filter: must be '{product_type}'")
        
        search_body["query"]["bool"]["must"].append({
            "term": {
                "product_type": product_type
            }
        })
    
    # Search term scoring (if we have search terms)
    if search_terms.strip():
        search_body["query"]["bool"]["should"].extend([
            # SKU matches
            {
                "match": {
                    "sku": {
                        "query": search_terms,
                        "boost": 20
                    }
                }
            },
            # Product name exact phrase
            {
                "match_phrase": {
                    "product_name": {
                        "query": search_terms,
                        "boost": 18
                    }
                }
            },
            # Product name fuzzy matches
            {
                "match": {
                    "product_name": {
                        "query": search_terms,
                        "fuzziness": "AUTO",
                        "boost": 15
                    }
                }
            },
            # Key features matches
            {
                "match": {
                    "key_features": {
                        "query": search_terms,
                        "fuzziness": "AUTO",
                        "boost": 8
                    }
                }
            },
            # Vector similarity
            {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'combined_embedding') + 1.0",
                        "params": {"query_vector": embedding_model.encode(search_terms).tolist()}
                    },
                    "boost": 10
                }
            }
        ])
        
        # Require at least some relevance if we have search terms
        search_body["query"]["bool"]["minimum_should_match"] = 1
    
    # Apply price and size filters
    await apply_strict_filters(search_body, params)
    
    try:
        response = await es.search(index="products", body=search_body)
        products = [format_product(hit["_source"]) for hit in response["hits"]["hits"]]
        
        await send_thinking(websocket, "search_complete", 
            f"Search found {len(products)} products matching {product_type} criteria.")
        
        return products
        
    except Exception as e:
        await send_thinking(websocket, "search_error", f"Search error: {str(e)}")
        return []

async def apply_strict_filters(search_body: Dict, params: Dict):
    """Apply strict price and size filters"""
    
    filters = search_body["query"]["bool"]["filter"]
    
    # Price filtering
    if params.get("price_min") or params.get("price_max"):
        price_script = """
            if (doc['price'].size() == 0 || doc['price'].value == '' || doc['price'].value == 'N/A') {
                return false;
            }
            try {
                double price = Double.parseDouble(doc['price'].value);
        """
        conditions = []
        script_params = {}
        
        if params.get("price_min"):
            price_min = float(params["price_min"]) if isinstance(params["price_min"], str) else params["price_min"]
            conditions.append("price >= params.price_min")
            script_params["price_min"] = price_min
            
        if params.get("price_max"):
            price_max = float(params["price_max"]) if isinstance(params["price_max"], str) else params["price_max"]
            conditions.append("price <= params.price_max")
            script_params["price_max"] = price_max
        
        price_script += f"return {' && '.join(conditions)};"
        price_script += """
            } catch (NumberFormatException e) {
                return false;
            }
        """
        
        filters.append({
            "script": {
                "script": {
                    "source": price_script,
                    "params": script_params
                }
            }
        })
    
    # Size filtering
    size_target = params.get("size_target")
    size_min = params.get("size_min")
    size_max = params.get("size_max")
    
    if size_target:
        # Convert string to int if needed
        try:
            size_target_num = int(size_target) if isinstance(size_target, str) else size_target
        except (ValueError, TypeError):
            size_target_num = 0
            
        # For size target, allow ±1 inch variance
        size_script = f"""
            if (doc['size'].size() == 0 || doc['size'].value == '' || doc['size'].value == 'N/A') {{
                return false;
            }}
            try {{
                double size = Double.parseDouble(doc['size'].value);
                return size >= {size_target_num - 1} && size <= {size_target_num + 1};
            }} catch (NumberFormatException e) {{
                return false;
            }}
        """
        
        filters.append({
            "script": {
                "script": {
                    "source": size_script
                }
            }
        })
    elif size_min or size_max:
        # Range-based size filtering with type conversion
        size_script = """
            if (doc['size'].size() == 0 || doc['size'].value == '' || doc['size'].value == 'N/A') {
                return false;
            }
            try {
                double size = Double.parseDouble(doc['size'].value);
        """
        conditions = []
        script_params = {}
        
        if size_min:
            size_min_num = int(size_min) if isinstance(size_min, str) else size_min
            conditions.append("size >= params.size_min")
            script_params["size_min"] = size_min_num
            
        if size_max:
            size_max_num = int(size_max) if isinstance(size_max, str) else size_max
            conditions.append("size <= params.size_max")
            script_params["size_max"] = size_max_num
        
        size_script += f"return {' && '.join(conditions)};"
        size_script += """
            } catch (NumberFormatException e) {
                return false;
            }
        """
        
        filters.append({
            "script": {
                "script": {
                    "source": size_script,
                    "params": script_params
                }
            }
        })

async def rerank_multi_products(query: str, products: List[Dict], websocket) -> List[Dict]:
    """Rerank products from multiple searches with balanced representation"""
    
    if len(products) <= 3:
        return products
    
    try:
        # Create product summaries with search origin
        product_summaries = []
        search_origins = {}
        
        for i, product in enumerate(products[:15]):  # Limit for LLM context
            summary = f"{i}: {product['product_name']}"
            if product.get('price') and product['price'] != 'N/A':
                summary += f" (${product['price']})"
            if product.get('size') and product['size'] != 'N/A':
                summary += f" {product['size']}\""
            summary += f" - {product.get('key_features', '')[:80]}"
            
            # Track search origin for balanced representation
            origin = product.get('search_origin', 'unknown')
            summary += f" [From: {origin}]"
            product_summaries.append(summary)
            
            # Count products from each search
            if origin not in search_origins:
                search_origins[origin] = []
            search_origins[origin].append(i)
        
        rerank_prompt = f"""Rerank these products from multiple searches to provide a balanced, helpful result.

User Query: "{query}"

Products from different searches:
{chr(10).join(product_summaries)}

Search origins: {list(search_origins.keys())}

Instructions:
- Provide a BALANCED representation from each product category
- For related types (like tvs + tvs_accessories), mix them appropriately
- Show the most relevant items from each category first
- Don't favor one category too heavily over others
- Consider the user wants to see variety across the requested types
- Return ONLY the numbers in order from best to worst match
- Example: "3,7,1,0,5,2,8,4,6"

Balanced Ranking:"""

        await send_thinking(websocket, "multi_reranking_llm", 
            f"Using LLM to rerank {len(product_summaries)} products from {len(search_origins)} different searches for balanced representation...")

        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at creating balanced product rankings from multi-category searches. Ensure fair representation of each category while maintaining relevance. Return only the ranking numbers."},
                {"role": "user", "content": rerank_prompt}
            ],
            max_tokens=100,
            temperature=0.2
        )
        
        ranking_text = response.choices[0].message.content.strip()
        
        # Parse and apply ranking
        rankings = [int(x.strip()) for x in ranking_text.split(',') if x.strip().isdigit()]
        reranked = []
        
        for rank in rankings:
            if 0 <= rank < len(products):
                reranked.append(products[rank])
        
        # Add any missed products, ensuring category balance
        ranked_indices = set(rankings)
        missed_by_origin = {}
        
        for i, product in enumerate(products):
            if i not in ranked_indices:
                origin = product.get('search_origin', 'unknown')
                if origin not in missed_by_origin:
                    missed_by_origin[origin] = []
                missed_by_origin[origin].append(product)
        
        # Add missed products in a balanced way
        for origin, missed_products in missed_by_origin.items():
            reranked.extend(missed_products)
        
        await send_thinking(websocket, "multi_reranking_complete", 
            f"Multi-search reranking complete. Balanced results from {len(search_origins)} categories: {list(search_origins.keys())}")
        
        return reranked
        
    except Exception as e:
        await send_thinking(websocket, "multi_reranking_error", 
            f"Multi-search reranking failed: {str(e)}. Using original order with category grouping.")
        
        # Fallback: group by search origin for balance
        grouped = {}
        for product in products:
            origin = product.get('search_origin', 'unknown')
            if origin not in grouped:
                grouped[origin] = []
            grouped[origin].append(product)
        
        # Interleave results from different origins
        balanced = []
        max_items = max(len(items) for items in grouped.values()) if grouped else 0
        
        for i in range(max_items):
            for origin, items in grouped.items():
                if i < len(items):
                    balanced.append(items[i])
        
        return balanced

async def analyze_query_with_llm(query: str, websocket) -> Dict:
   """Enhanced LLM analysis with support for multi-product queries"""
   
   prompt = f"""Analyze this product search query very carefully and determine if it's a single or multi-product query:

Query: "{query}"

Rules:
1. If asking for ONE product type → single query
2. If asking for MULTIPLE product types OR related types (like tvs + tvs_accessories) → multi-query
3. Extract EXACT product types (tvs, monitors, soundbars, tvs_accessories, monitors_accessories)
4. For multi-queries, create separate search parameters for each product type

Examples:
Single: "gaming monitor under $500" → ONE search for monitors
Multi: "tv under $1000 and monitor over $300" → TWO searches (tvs + monitors)
Multi: "TV and TV accessories" → TWO searches (tvs + tvs_accessories)
Multi: "monitor and accessories for it" → TWO searches (monitors + monitors_accessories)

For SINGLE queries, respond:
{{
   "query_type": "single",
   "product_type": "monitors",
   "search_terms": ["gaming"],
   "price_min": null,
   "price_max": 500,
   "size_target": null,
   "size_min": null,
   "size_max": null
}}

For MULTI queries, respond:
{{
   "query_type": "multi",
   "searches": [
       {{
           "product_type": "tvs",
           "search_terms": [],
           "price_min": null,
           "price_max": 1000,
           "size_target": null,
           "size_min": null,
           "size_max": null
       }},
       {{
           "product_type": "tvs_accessories", 
           "search_terms": [],
           "price_min": 300,
           "price_max": null,
           "size_target": null,
           "size_min": null,
           "size_max": null
       }}
   ]
}}

Analyze the query and respond with appropriate JSON:"""

   try:
       response = await openai_client.chat.completions.create(
           model="gpt-3.5-turbo",
           messages=[
               {"role": "system", "content": "You are a precise query analyzer. Detect single vs multi-product queries and extract exact parameters. Respond with JSON only."},
               {"role": "user", "content": prompt}
           ],
           temperature=0,
           max_tokens=400
       )
       
       content = response.choices[0].message.content.strip()
       
       # Extract JSON
       if "```json" in content:
           start = content.find("```json") + 7
           end = content.find("```", start)
           content = content[start:end]
       elif "{" in content:
           start = content.find("{")
           end = content.rfind("}") + 1
           content = content[start:end]
       
       params = json.loads(content)
       
       if params.get("query_type") == "multi":
           await send_thinking(websocket, "llm_analysis_done", 
               f"LLM detected MULTI-query: {len(params.get('searches', []))} separate searches needed")
       else:
           await send_thinking(websocket, "llm_analysis_done", 
               f"LLM detected SINGLE query: product_type='{params.get('product_type')}', "
               f"price_max={params.get('price_max')}")
       
       return params
       
   except Exception as e:
       await send_thinking(websocket, "llm_analysis_error", f"LLM analysis failed: {str(e)}")
       return {
           "query_type": "single",
           "product_type": None,
           "search_terms": [query],
           "price_min": None,
           "price_max": None,
           "size_target": None,
           "size_min": None,
           "size_max": None
       }
# Remove the unused comprehensive_search function since we use comprehensive_search_with_strict_filtering
async def vector_similarity_fallback(query: str, websocket) -> List[Dict]:
    """Fallback vector similarity search when strict search fails"""
    
    try:
        # Generate embedding for the query
        query_embedding = embedding_model.encode(query).tolist()
        
        # Pure vector similarity search without strict filters
        search_body = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'combined_embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            },
            "size": 10,
            "_source": ["product_name", "sku", "price", "size", "product_url", "image_urls", "key_features", "product_type"]
        }
        
        response = await es.search(index="products", body=search_body)
        
        # Filter out very low similarity scores (< 1.3 means very poor match)
        products = []
        for hit in response["hits"]["hits"]:
            if hit["_score"] > 1.3:  # Reasonable similarity threshold
                product = format_product(hit["_source"])
                product["similarity_score"] = hit["_score"]
                product["is_fallback"] = True  # Mark as fallback result
                products.append(product)
        
        await send_thinking(websocket, "vector_fallback_complete", 
            f"Vector similarity found {len(products)} reasonable alternatives (similarity > 1.3)")
        
        return products
        
    except Exception as e:
        await send_thinking(websocket, "vector_fallback_error", 
            f"Vector similarity fallback failed: {str(e)}")
        return []
    
async def generate_explanation(query: str, products: List[Dict], websocket) -> str:
    """Generate explanation with fallback search if no products found"""
    
    if not products:
        # Fallback: Try vector similarity search to find something relevant
        await send_thinking(websocket, "fallback_search", 
            "No products found with strict criteria. Trying vector similarity search...")
        
        fallback_products = await vector_similarity_fallback(query, websocket)
        
        if fallback_products:
            await send_thinking(websocket, "fallback_success", 
                f"Found {len(fallback_products)} similar products as alternatives.")
            
            # Update products list with fallback results
            products.extend(fallback_products)
            
            # Generate explanation for fallback results
            try:
                top_products = fallback_products[:3]
                product_info = "\n".join([
                    f"- {p['product_name']} (${p.get('price', 'N/A')})"
                    for p in top_products
                ])
                
                prompt = f"""User searched for: "{query}"

No exact matches found, but here are similar products:
{product_info}

Write a brief helpful response (1-2 sentences) explaining these are similar alternatives."""

                response = await openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Provide brief, helpful explanations for alternative product suggestions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100,
                    temperature=0.3
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                await send_thinking(websocket, "fallback_explanation_error", f"Fallback explanation error: {str(e)}")
                return f"No exact matches found, but here are {len(fallback_products)} similar products that might interest you."
        else:
            return "No products found matching your criteria. Please try a different search or broader terms."
    
    # Original logic for when products are found
    try:
        top_products = products[:3]
        product_info = "\n".join([
            f"- {p['product_name']} (${p.get('price', 'N/A')})"
            for p in top_products
        ])
        
        prompt = f"""User searched for: "{query}"

Top results:
{product_info}

Write a brief helpful response (1-2 sentences) explaining the results."""

        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Provide brief, helpful product recommendations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        await send_thinking(websocket, "explanation_error", f"Explanation error: {str(e)}")
        return f"Found {len(products)} products. Top result: {products[0]['product_name']}."

def format_product(source: dict) -> dict:
    """Format product data"""
    return {
        "product_name": source.get("product_name", ""),
        "sku": source.get("sku", ""),
        "price": source.get("price", ""),
        "size": source.get("size", ""),
        "product_url": source.get("product_url", ""),
        "image_urls": source.get("image_urls", []) if isinstance(source.get("image_urls"), list) else [source.get("image_urls", "")],
        "key_features": source.get("key_features", "")
    }

@app.get("/")
async def root():
    return {
        "message": "Smart Search API",
        "strategy": "Simple keyword search → LLM fallback"
    }

@app.get("/health")
async def health():
    es_status = "connected" if await es.ping() else "disconnected"
    llm_status = "available" if OPENAI_API_KEY else "unavailable"
    
    return {
        "status": "healthy",
        "elasticsearch": es_status,
        "llm": llm_status,
        "active_sessions": len(chat_sessions)
    }

# Run with: uvicorn app.smart_main:app --reload --host 0.0.0.0 --port 8000