# backend/app/simple_main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import asyncio

load_dotenv()

app = FastAPI(title="Simple LG Search")

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

# Simple memory store
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
                
                # Get conversation history for context
                chat_history = chat_sessions.get(session_id, [])
                
                # Step 1: Deep query analysis with AI thinking
                await stream_ai_thinking(websocket, "query_analysis", query, chat_history)
                analysis = await deep_analyze_query(query, chat_history, websocket)
                
                # Step 2: Search strategy planning
                await stream_ai_thinking(websocket, "search_planning", query, chat_history, analysis)
                search_strategy = await plan_search_strategy(query, analysis, websocket)
                
                # Step 3: Execute search with reasoning
                await stream_ai_thinking(websocket, "search_execution", query, chat_history, analysis, search_strategy)
                products = await execute_search_with_reasoning(query, analysis, search_strategy, websocket)
                
                # Step 4: Result analysis and response generation
                await stream_ai_thinking(websocket, "response_generation", query, chat_history, analysis, products)
                explanation = await generate_contextual_response(query, products, analysis, chat_history, websocket)
                
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

async def stream_ai_thinking(websocket, phase, query, chat_history, analysis=None, strategy=None, products=None):
    """Stream AI thinking process to frontend"""
    await websocket.send_text(json.dumps({
        "type": "thinking_start",
        "phase": phase,
        "description": f"AI is thinking about {phase.replace('_', ' ')}...",
        "timestamp": datetime.now().isoformat()
    }))

async def deep_analyze_query(query: str, chat_history: list, websocket):
    """Deep AI-powered query analysis with streaming thoughts"""
    
    if not OPENAI_API_KEY:
        return {"intent": "product_search", "reasoning": "No AI analysis available"}
    
    # Build context from chat history
    context = ""
    if chat_history:
        recent_history = chat_history[-3:]
        context = "\nRecent conversation:\n" + "\n".join([
            f"User: {msg['user']}\nAssistant: {msg['assistant'][:100]}..." 
            for msg in recent_history
        ])
    
    system_prompt = f"""You are analyzing a user's query for LG product search. Think step by step and explain your reasoning in detail.

Current query: "{query}"
{context}

Think through this systematically:
1. What is the user really looking for?
2. What product categories are involved?
3. Are there any constraints (price, size, features)?
4. How does this relate to previous conversation?
5. What search strategy would work best?

Provide your analysis in JSON format:
{{
    "thinking": "Your detailed step-by-step reasoning (2-3 paragraphs)",
    "intent": "product_search|sku_lookup|comparison|feature_inquiry",
    "product_types": ["tv", "monitor", "soundbar", "accessory"],
    "constraints": {{"price_max": 500, "size_min": 55}},
    "keywords": ["gaming", "4k", "budget"],
    "search_approach": "Your recommended search strategy",
    "confidence": 0.95
}}"""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this query: {query}"}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end]
        elif "{" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            content = content[start:end]
        
        analysis = json.loads(content)
        
        # Stream the thinking process
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "query_analysis",
            "thinking": analysis.get("thinking", "Analyzing the user's intent..."),
            "result": {
                "intent": analysis.get("intent"),
                "confidence": analysis.get("confidence", 0.8)
            },
            "timestamp": datetime.now().isoformat()
        }))
        
        return analysis
        
    except Exception as e:
        fallback_analysis = {
            "thinking": f"I'm analyzing the query '{query}'. Since this appears to be a product search, I'll look for relevant LG products and apply any constraints I can identify.",
            "intent": "product_search",
            "product_types": [],
            "constraints": {},
            "keywords": query.split(),
            "confidence": 0.6
        }
        
        await websocket.send_text(json.dumps({
            "type": "thinking_detail", 
            "phase": "query_analysis",
            "thinking": fallback_analysis["thinking"],
            "result": {"intent": "product_search", "confidence": 0.6},
            "timestamp": datetime.now().isoformat()
        }))
        
        return fallback_analysis

async def plan_search_strategy(query: str, analysis: dict, websocket):
    """AI plans the search strategy"""
    
    if not OPENAI_API_KEY:
        return {"approach": "basic_search"}
    
    system_prompt = f"""Based on the query analysis, plan the optimal search strategy.

Query: "{query}"
Analysis: {json.dumps(analysis, indent=2)}

Think through:
1. Should this be a direct SKU lookup or broader search?
2. What Elasticsearch query structure would work best?
3. How should results be filtered and ranked?
4. What fallback strategies if no results?

Respond in JSON:
{{
    "thinking": "Your detailed reasoning about the search strategy (2-3 paragraphs)",
    "approach": "sku_lookup|keyword_search|hybrid_search|filtered_search",
    "elasticsearch_strategy": "Description of ES query approach",
    "ranking_factors": ["price", "relevance", "features"],
    "expected_results": "What you expect to find"
}}"""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Plan the search strategy"}
            ],
            temperature=0.2,
            max_tokens=600
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end]
        elif "{" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            content = content[start:end]
        
        strategy = json.loads(content)
        
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "search_planning", 
            "thinking": strategy.get("thinking", "Planning the search approach..."),
            "result": {
                "approach": strategy.get("approach"),
                "strategy": strategy.get("elasticsearch_strategy", "")
            },
            "timestamp": datetime.now().isoformat()
        }))
        
        return strategy
        
    except Exception as e:
        fallback_strategy = {
            "thinking": "I'll use a standard keyword search approach, looking for matches in product names and features, and apply any price or size constraints identified.",
            "approach": "keyword_search",
            "elasticsearch_strategy": "Multi-match query with relevance scoring"
        }
        
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "search_planning",
            "thinking": fallback_strategy["thinking"],
            "result": {"approach": "keyword_search"},
            "timestamp": datetime.now().isoformat()
        }))
        
        return fallback_strategy

async def execute_search_with_reasoning(query: str, analysis: dict, strategy: dict, websocket):
    """Execute search and explain the process"""
    
    # Stream search execution thinking
    search_thinking = f"""Now I'm executing the search strategy. Based on my analysis, I determined this is a {analysis.get('intent', 'product_search')} with {strategy.get('approach', 'keyword_search')} approach. 

I'll search the Elasticsearch index 'products' using the following strategy: {strategy.get('elasticsearch_strategy', 'standard search')}. I'm looking for products that match the user's criteria and will rank them by relevance and any specific constraints identified."""

    await websocket.send_text(json.dumps({
        "type": "thinking_detail",
        "phase": "search_execution",
        "thinking": search_thinking,
        "result": {"status": "executing"},
        "timestamp": datetime.now().isoformat()
    }))
    
    try:
        # Check for SKU first
        if analysis.get("intent") == "sku_lookup" or (len(query.split()) == 1 and any(c.isupper() for c in query)):
            response = await es.search(
                index="products",
                body={"query": {"term": {"sku": query.strip().upper()}}}
            )
            if response["hits"]["hits"]:
                result = [format_product(response["hits"]["hits"][0]["_source"])]
                
                await websocket.send_text(json.dumps({
                    "type": "thinking_detail",
                    "phase": "search_execution", 
                    "thinking": f"Perfect! I found an exact match for SKU '{query}'. This appears to be a direct product lookup, so I'm returning the specific product details.",
                    "result": {"found": "exact_sku_match", "count": 1},
                    "timestamp": datetime.now().isoformat()
                }))
                
                return result
        
        # Step 1: Try precise search first
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["product_name^2", "key_features", "sku^3"],
                                "fuzziness": "AUTO"
                            }
                        }
                    ],
                    "filter": []
                }
            },
            "size": 10
        }
        
        # Apply constraints
        constraints_applied = []
        if analysis.get("constraints", {}).get("price_max"):
            price_max = analysis["constraints"]["price_max"]
            search_body["query"]["bool"]["filter"].append({
                "range": {"price": {"lte": price_max}}
            })
            constraints_applied.append(f"price ≤ ${price_max}")
        
        if analysis.get("constraints", {}).get("size_min"):
            size_min = analysis["constraints"]["size_min"]
            search_body["query"]["bool"]["filter"].append({
                "range": {"size": {"gte": size_min}}
            })
            constraints_applied.append(f"size ≥ {size_min}\"")

        response = await es.search(index="products", body=search_body)
        products = [format_product(hit["_source"]) for hit in response["hits"]["hits"]]
        
        # If no results, try fallback searches
        if len(products) == 0:
            await websocket.send_text(json.dumps({
                "type": "thinking_detail",
                "phase": "search_execution",
                "thinking": "No results with the initial search. Let me try broader approaches to find relevant products.",
                "result": {"fallback": "trying_broader_search"},
                "timestamp": datetime.now().isoformat()
            }))
            
            # Step 2: Try product type based search
            product_keywords = []
            if "tv" in query.lower() or "television" in query.lower():
                product_keywords = ["tv", "television", "oled", "qled"]
            elif "monitor" in query.lower():
                product_keywords = ["monitor", "display"]
            elif "soundbar" in query.lower() or "speaker" in query.lower():
                product_keywords = ["soundbar", "speaker", "audio"]
            
            if product_keywords:
                fallback_search = {
                    "query": {
                        "bool": {
                            "should": [
                                {"terms": {"product_name": product_keywords}},
                                {"terms": {"key_features": product_keywords}}
                            ],
                            "filter": []
                        }
                    },
                    "size": 10
                }
                
                # Apply same constraints
                if constraints_applied:
                    if analysis.get("constraints", {}).get("price_max"):
                        fallback_search["query"]["bool"]["filter"].append({
                            "range": {"price": {"lte": analysis["constraints"]["price_max"]}}
                        })
                
                response = await es.search(index="products", body=fallback_search)
                products = [format_product(hit["_source"]) for hit in response["hits"]["hits"]]
            
            # Step 3: If still no results, try very broad search (remove constraints)
            if len(products) == 0:
                await websocket.send_text(json.dumps({
                    "type": "thinking_detail",
                    "phase": "search_execution",
                    "thinking": "Still no results. Let me search more broadly by removing some constraints to show available options.",
                    "result": {"fallback": "removing_constraints"},
                    "timestamp": datetime.now().isoformat()
                }))
                
                # Very broad search - just get some products
                broad_search = {
                    "query": {"match_all": {}},
                    "size": 10,
                    "sort": [{"_score": {"order": "desc"}}]
                }
                
                # Add product type filter if identified
                if product_keywords:
                    broad_search["query"] = {
                        "bool": {
                            "should": [
                                {"terms": {"product_name": product_keywords}},
                                {"terms": {"key_features": product_keywords}}
                            ]
                        }
                    }
                
                response = await es.search(index="products", body=broad_search)
                products = [format_product(hit["_source"]) for hit in response["hits"]["hits"]]
                
                # Sort by price if budget was mentioned
                if "budget" in query.lower() or "cheap" in query.lower() or "affordable" in query.lower():
                    products.sort(key=lambda x: x.get("price", 999999))
        
        # Explain search results
        constraints_text = f" with constraints: {', '.join(constraints_applied)}" if constraints_applied else ""
        
        if len(products) > 0:
            result_thinking = f"""Search completed! I found {len(products)} products matching the query "{query}"{constraints_text}. 

The search strategy included multiple fallback approaches to ensure I could provide helpful recommendations even if the exact terms didn't match. I prioritized products that best match the user's intent."""
            
            if len(products) > 5:
                result_thinking += f" I'm showing the top {min(len(products), 10)} most relevant results."
        else:
            result_thinking = "Despite trying multiple search strategies including broad matching and constraint removal, I couldn't find any products in the database."

        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "search_execution",
            "thinking": result_thinking,
            "result": {"found": len(products), "constraints": constraints_applied},
            "timestamp": datetime.now().isoformat()
        }))
        
        return products
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "search_execution",
            "thinking": f"I encountered an error while searching: {str(e)}. This might be due to Elasticsearch connectivity issues or malformed query syntax.",
            "result": {"error": str(e)},
            "timestamp": datetime.now().isoformat()
        }))
        return []

async def generate_contextual_response(query: str, products: list, analysis: dict, chat_history: list, websocket):
    """Generate response with AI reasoning"""
    
    if not OPENAI_API_KEY:
        return f"Found {len(products)} products for '{query}'"
    
    # Stream response generation thinking
    response_thinking = f"""Now I need to craft a helpful response for the user. I found {len(products)} products for their query "{query}". 

I should consider: the user's original intent ({analysis.get('intent')}), the quality and relevance of results, any constraints that were applied, and how this fits into our conversation flow. I want to be helpful and informative while keeping the response natural and conversational."""

    await websocket.send_text(json.dumps({
        "type": "thinking_detail",
        "phase": "response_generation",
        "thinking": response_thinking,
        "result": {"status": "generating"},
        "timestamp": datetime.now().isoformat()
    }))
    
    if not products:
        no_results_thinking = f"""Since I didn't find any products, I need to help the user understand why and suggest alternatives. This could be due to overly specific search terms, constraints that are too restrictive, or the product not being in our LG catalog."""
        
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "response_generation",
            "thinking": no_results_thinking,
            "result": {"response_type": "no_results_help"},
            "timestamp": datetime.now().isoformat()
        }))
        
        return f"I couldn't find any LG products matching '{query}'. Try using broader search terms or check if the product model is correct."
    
    try:
        # Build context
        context = ""
        if chat_history:
            context = f"\nPrevious conversation context: {chat_history[-1].get('user', '')} -> {chat_history[-1].get('assistant', '')[:100]}..."
        
        # Product summary
        product_names = [p["product_name"][:60] for p in products[:3]]
        product_summary = "\n".join([f"- {name}" for name in product_names])
        
        system_prompt = f"""Generate a helpful, conversational response about the LG product search results.

Query: "{query}"
Intent: {analysis.get('intent')}
Results found: {len(products)}
Top products:
{product_summary}

Context: {context}

Be natural, helpful, and informative. Highlight key features or benefits that match the user's needs."""

        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Explain these search results for: {query}"}
            ],
            max_tokens=200,
            temperature=0.4
        )
        
        final_response = response.choices[0].message.content.strip()
        
        final_thinking = f"""I've crafted a response that acknowledges the {len(products)} products found and tries to highlight the most relevant options based on the user's query. I'm focusing on being helpful while keeping the tone conversational and not overwhelming them with too much technical detail."""
        
        await websocket.send_text(json.dumps({
            "type": "thinking_detail",
            "phase": "response_generation",
            "thinking": final_thinking,
            "result": {"response_ready": True, "length": len(final_response)},
            "timestamp": datetime.now().isoformat()
        }))
        
        return final_response
        
    except Exception as e:
        return f"Found {len(products)} LG products for '{query}'. Here are the best matches!"

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
    return {"message": "Simple LG Search API with AI Thinking"}

@app.get("/health")
async def health():
    return {"status": "healthy", "sessions": len(chat_sessions)}