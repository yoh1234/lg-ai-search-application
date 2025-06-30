# backend/app/smart_main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI
import json
import os
import re
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio
from sentence_transformers import SentenceTransformer

load_dotenv()

app = FastAPI(title="Smart Keyword Search with LLM Fallback")

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

class SmartQueryProcessor:
    """Enhanced query processing with suggester integration"""
    
    @staticmethod
    def analyze_query_type(query: str) -> Dict[str, Any]:
        """Analyze if query is simple keyword or complex natural language"""
        words = query.strip().split()
        word_count = len(words)
        
        # Check for natural language indicators
        has_question_words = any(word.lower() in ['what', 'which', 'how', 'where', 'when', 'why', 'who'] for word in words)
        has_sentences = any(punct in query for punct in ['.', '?', '!'])
        has_connectors = any(word.lower() in ['and', 'or', 'but', 'with', 'for', 'under', 'over', 'between'] for word in words)
        has_phrases = any(phrase in query.lower() for phrase in [
            'looking for', 'i need', 'i want', 'recommend', 'suggest', 'best', 'good', 
            'compare', 'vs', 'versus', 'budget', 'cheap', 'expensive', 'around', 'under'
        ])
        
        # Scoring system
        complexity_score = 0
        if word_count > 4: complexity_score += 2
        if has_question_words: complexity_score += 3
        if has_sentences: complexity_score += 2
        if has_connectors: complexity_score += 1
        if has_phrases: complexity_score += 3
        
        is_complex = complexity_score >= 4
        
        return {
            "is_complex": is_complex,
            "complexity_score": complexity_score,
            "word_count": word_count,
            "indicators": {
                "question_words": has_question_words,
                "sentences": has_sentences,
                "connectors": has_connectors,
                "phrases": has_phrases
            },
            "strategy": "llm_pipeline" if is_complex else "keyword_first"
        }
    
    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those'}
        
        # Clean and split
        clean_query = re.sub(r'[^\w\s]', ' ', query.lower())
        words = [word.strip() for word in clean_query.split() if word.strip() and word not in stop_words and len(word) > 2]
        
        return words

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
                
                # Real-time thinking: Query analysis
                await send_thinking(websocket, "query_analysis", 
                    f"Analyzing query: '{query}'. Let me first determine if this is a simple keyword search or needs complex natural language processing...")
                
                processor = SmartQueryProcessor()
                query_analysis = processor.analyze_query_type(query)
                
                await send_thinking(websocket, "strategy_decision", 
                    f"Query analysis complete. Complexity score: {query_analysis['complexity_score']}/10. "
                    f"Word count: {query_analysis['word_count']}. Strategy: {query_analysis['strategy']}. "
                    f"This query {'requires LLM processing' if query_analysis['is_complex'] else 'can use fast keyword search'}.")
                
                if query_analysis["strategy"] == "keyword_first":
                    # Try keyword search with suggestors first
                    keyword_results = await enhanced_keyword_search(query, websocket)
                    
                    if keyword_results and len(keyword_results) >= 3:
                        # Success with keyword search
                        explanation = f"Found {len(keyword_results)} products using fast keyword search with typo correction and fuzzy matching."
                        products = keyword_results
                        
                        await send_thinking(websocket, "keyword_success", 
                            f"Excellent! Keyword search found {len(keyword_results)} relevant results. No need for expensive LLM processing. Fast and accurate!")
                    else:
                        # Fallback to LLM
                        await send_thinking(websocket, "llm_fallback", 
                            f"Keyword search insufficient ({len(keyword_results) if keyword_results else 0} results). Activating LLM pipeline for better understanding...")
                        
                        products, explanation = await advanced_llm_pipeline(query, chat_history, websocket)
                else:
                    # Direct to LLM for complex queries
                    await send_thinking(websocket, "llm_direct", 
                        "Query is complex - going straight to LLM pipeline for natural language understanding...")
                    
                    products, explanation = await advanced_llm_pipeline(query, chat_history, websocket)
                
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
                    "total_count": len(products),
                    "strategy_used": query_analysis["strategy"]
                }))
                
    except WebSocketDisconnect:
        pass

async def send_thinking(websocket: WebSocket, phase: str, thinking: str, result: Dict = None):
    """Send real-time thinking process to frontend"""
    await websocket.send_text(json.dumps({
        "type": "thinking",
        "phase": phase,
        "thinking": thinking,
        "result": result or {},
        "timestamp": datetime.now().isoformat()
    }))

async def enhanced_keyword_search(query: str, websocket) -> Optional[List[Dict]]:
    """Enhanced keyword search with suggestors and typo correction"""
    
    processor = SmartQueryProcessor()
    keywords = processor.extract_keywords(query)
    
    await send_thinking(websocket, "keyword_extraction", 
        f"Extracted keywords: {keywords}. Now trying multiple search strategies including suggestors for typo correction...")
    
    try:
        # Strategy 1: Try exact/phrase matching first
        exact_results = await try_exact_search(query, websocket)
        if exact_results and len(exact_results) >= 3:
            await send_thinking(websocket, "exact_match_success", 
                f"Exact phrase matching succeeded! Found {len(exact_results)} products.")
            return exact_results
        
        # Strategy 2: Use completion suggestors for typo correction
        corrected_query = await try_suggester_correction(query, keywords, websocket)
        if corrected_query != query:
            await send_thinking(websocket, "typo_correction", 
                f"Typo detected! Corrected '{query}' to '{corrected_query}'. Searching with corrected terms...")
            
            corrected_results = await try_exact_search(corrected_query, websocket)
            if corrected_results and len(corrected_results) >= 3:
                return corrected_results
        
        # Strategy 3: Fuzzy matching with individual keywords
        fuzzy_results = await try_fuzzy_search(keywords, websocket)
        if fuzzy_results and len(fuzzy_results) >= 3:
            await send_thinking(websocket, "fuzzy_success", 
                f"Fuzzy keyword matching found {len(fuzzy_results)} products.")
            return fuzzy_results
        
        # Strategy 4: Vector similarity search as last resort
        vector_results = await try_vector_search(query, websocket)
        if vector_results and len(vector_results) >= 2:
            await send_thinking(websocket, "vector_fallback", 
                f"Vector similarity search found {len(vector_results)} products as keyword fallback.")
            return vector_results
        
        await send_thinking(websocket, "keyword_insufficient", 
            "All keyword strategies exhausted. Results insufficient for good user experience.")
        return None
        
    except Exception as e:
        await send_thinking(websocket, "keyword_error", f"Keyword search error: {str(e)}")
        return None

async def try_exact_search(query: str, websocket) -> List[Dict]:
    """Try exact phrase matching"""
    search_body = {
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["product_name^5", "sku^4", "key_features^3"],
                            "type": "phrase",
                            "boost": 10
                        }
                    },
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["product_name^3", "key_features^2"],
                            "type": "phrase_prefix",
                            "boost": 5
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        },
        "size": 15,
        "_source": ["product_name", "sku", "price", "size", "product_url", "image_urls", "key_features"]
    }
    
    response = await es.search(index="products", body=search_body)
    return [format_product(hit["_source"]) for hit in response["hits"]["hits"] if hit["_score"] > 1.0]

async def try_suggester_correction(query: str, keywords: List[str], websocket) -> str:
    """Use completion suggestors to correct typos"""
    
    await send_thinking(websocket, "suggester_check", 
        f"Checking suggestors for typo correction on keywords: {keywords}")
    
    corrected_parts = []
    
    for keyword in keywords:
        # Try product name suggester
        suggest_body = {
            "suggest": {
                "product_name_suggest": {
                    "prefix": keyword,
                    "completion": {
                        "field": "product_name_suggest",
                        "size": 5
                    }
                },
                "features_suggest": {
                    "prefix": keyword,
                    "completion": {
                        "field": "key_features_suggest", 
                        "size": 5
                    }
                },
                "sku_suggest": {
                    "prefix": keyword,
                    "completion": {
                        "field": "sku_suggest",
                        "size": 3
                    }
                }
            }
        }
        
        try:
            response = await es.search(index="products", body=suggest_body)
            
            # Get best suggestions
            all_suggestions = []
            
            for suggest_type in ["product_name_suggest", "features_suggest", "sku_suggest"]:
                if response.get("suggest", {}).get(suggest_type):
                    options = response["suggest"][suggest_type][0].get("options", [])
                    for option in options:
                        all_suggestions.append((option["text"], option["_score"]))
            
            if all_suggestions:
                # Find best match (highest score)
                best_suggestion = max(all_suggestions, key=lambda x: x[1])
                if best_suggestion[1] > 2.0:  # Good confidence
                    corrected_parts.append(best_suggestion[0])
                    continue
            
            # No good suggestion found, keep original
            corrected_parts.append(keyword)
            
        except Exception as e:
            await send_thinking(websocket, "suggester_error", f"Suggester error for '{keyword}': {str(e)}")
            corrected_parts.append(keyword)
    
    corrected_query = " ".join(corrected_parts)
    
    if corrected_query != " ".join(keywords):
        await send_thinking(websocket, "typo_found", 
            f"Suggestors found potential corrections: '{' '.join(keywords)}' → '{corrected_query}'")
    
    return corrected_query

async def try_fuzzy_search(keywords: List[str], websocket) -> List[Dict]:
    """Fuzzy search with individual keywords"""
    search_body = {
        "query": {
            "bool": {
                "should": []
            }
        },
        "size": 15
    }
    
    for keyword in keywords:
        search_body["query"]["bool"]["should"].extend([
            {
                "fuzzy": {
                    "product_name": {
                        "value": keyword,
                        "fuzziness": "AUTO",
                        "boost": 3
                    }
                }
            },
            {
                "fuzzy": {
                    "key_features": {
                        "value": keyword,
                        "fuzziness": "AUTO", 
                        "boost": 2
                    }
                }
            }
        ])
    
    search_body["query"]["bool"]["minimum_should_match"] = 1
    
    response = await es.search(index="products", body=search_body)
    return [format_product(hit["_source"]) for hit in response["hits"]["hits"] if hit["_score"] > 0.5]

async def try_vector_search(query: str, websocket) -> List[Dict]:
    """Vector similarity search"""
    query_vector = embedding_model.encode(query).tolist()
    
    search_body = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'combined_embedding') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        },
        "size": 10,
        "min_score": 1.5
    }
    
    response = await es.search(index="products", body=search_body)
    return [format_product(hit["_source"]) for hit in response["hits"]["hits"]]

async def advanced_llm_pipeline(query: str, chat_history: List, websocket) -> tuple[List[Dict], str]:
    """Advanced LLM pipeline with real-time thinking"""
    
    if not OPENAI_API_KEY:
        return [], "LLM features not available - no API key configured"
    
    # Step 1: Deep query analysis with real thinking
    await send_thinking(websocket, "llm_analysis_start", 
        "Starting deep query analysis. I need to understand user intent, extract entities, identify constraints, and plan the optimal search strategy...")
    
    improved_query = await llm_query_improvement(query, chat_history, websocket)
    
    # Step 2: Enhanced retrieval with filtering and sorting
    await send_thinking(websocket, "retrieval_start", 
        f"Query analysis complete. Now executing enhanced retrieval with filters and sorting based on the insights...")
    
    candidates = await llm_guided_retrieval(improved_query, websocket)
    
    # Step 3: Intelligent reranking
    if len(candidates) > 3:
        await send_thinking(websocket, "reranking_start", 
            f"Retrieved {len(candidates)} candidates. Now applying AI reranking to optimize relevance order...")
        
        reranked_products = await intelligent_reranking(query, candidates, websocket)
    else:
        reranked_products = candidates
    
    # Step 4: Generate explanation with real product analysis
    await send_thinking(websocket, "explanation_start", 
        "Analyzing the final product set to generate personalized recommendations based on actual specifications...")
    
    explanation = await generate_intelligent_explanation(query, reranked_products, websocket)
    
    return reranked_products, explanation

async def llm_query_improvement(query: str, chat_history: List, websocket) -> Dict[str, Any]:
    """LLM-powered query analysis with real-time thinking"""
    
    # Build context from chat history
    context = ""
    if chat_history:
        recent = chat_history[-3:]
        context = "\nRecent conversation context:\n" + "\n".join([
            f"User: {msg['user']}\nAssistant: {msg['assistant'][:150]}..."
            for msg in recent
        ])
    
    analysis_prompt = f"""You are an expert product search analyst. Analyze this query and think step-by-step about how to search for the best products.

Query: "{query}"
{context}

Think through this carefully:
1. What specific product category is the user looking for?
2. What are the key features or requirements they mentioned?
3. Are there any price, size, or performance constraints?
4. What would be the best search terms for Elasticsearch?
5. How should results be filtered and sorted?
6. What is the user's likely intent and priority?

Respond with your thinking process and structured analysis in JSON:
{{
    "thinking_process": "Your detailed step-by-step reasoning about the query (4-5 sentences explaining your analysis)",
    "product_category": "tv|monitor|soundbar|accessory|general",
    "key_requirements": ["requirement1", "requirement2", "requirement3"],
    "search_terms": ["optimized", "search", "terms"],
    "filters": {{
        "price_range": {{"min": 100, "max": 2000}},
        "size_range": {{"min": 32, "max": 85}},
        "must_have_features": ["feature1", "feature2"]
    }},
    "sort_preference": "price_asc|price_desc|size_asc|size_desc|relevance",
    "user_intent": "Detailed description of what the user is trying to achieve",
    "confidence_level": 0.85
}}"""

    try:
        await send_thinking(websocket, "llm_analyzing", 
            "Sending query to LLM for deep analysis. The AI will break down user intent, extract requirements, and plan the search strategy...")
        
        response = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert product search analyst. Think step by step and provide detailed reasoning."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.1,
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
        
        analysis = json.loads(content)
        
        await send_thinking(websocket, "llm_analysis_complete", 
            f"LLM Analysis: {analysis.get('thinking_process', 'Analysis completed')}. "
            f"Category: {analysis.get('product_category')}. Intent: {analysis.get('user_intent', 'General search')}. "
            f"Confidence: {analysis.get('confidence_level', 0.5):.0%}")
        
        return analysis
        
    except Exception as e:
        await send_thinking(websocket, "llm_analysis_error", 
            f"LLM analysis failed: {str(e)}. Using intelligent fallback analysis...")
        
        # Intelligent fallback
        processor = SmartQueryProcessor()
        keywords = processor.extract_keywords(query)
        
        return {
            "thinking_process": "Using keyword-based analysis due to LLM unavailability",
            "product_category": "general",
            "key_requirements": keywords[:3],
            "search_terms": keywords,
            "filters": {},
            "sort_preference": "relevance",
            "user_intent": f"Search for products related to: {', '.join(keywords)}",
            "confidence_level": 0.6
        }

async def llm_guided_retrieval(query_analysis: Dict, websocket) -> List[Dict]:
    """Enhanced retrieval with LLM-guided filtering and sorting"""
    
    search_terms = " ".join(query_analysis.get("search_terms", []))
    filters = query_analysis.get("filters", {})
    category = query_analysis.get("product_category", "")
    sort_pref = query_analysis.get("sort_preference", "relevance")
    
    await send_thinking(websocket, "retrieval_planning", 
        f"Planning retrieval: terms='{search_terms}', category='{category}', sort='{sort_pref}', filters={len(filters)} constraints")
    
    try:
        # Build complex search query
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        # Exact phrase matching (highest priority)
                        {
                            "multi_match": {
                                "query": search_terms,
                                "fields": ["product_name^5", "key_features^3"],
                                "type": "phrase",
                                "boost": 10
                            }
                        },
                        # Fuzzy matching for typos
                        {
                            "multi_match": {
                                "query": search_terms,
                                "fields": ["product_name^4", "key_features^2"],
                                "fuzziness": "AUTO",
                                "boost": 5
                            }
                        }
                    ],
                    "filter": [],
                    "minimum_should_match": 1
                }
            },
            "size": 25
        }
        
        # Add category boost
        if category and category != "general":
            search_body["query"]["bool"]["should"].append({
                "match": {
                    "product_type": {
                        "query": category,
                        "boost": 3
                    }
                }
            })
        
        # Add intelligent filters from LLM analysis
        filter_clause = search_body["query"]["bool"]["filter"]
        
        price_range = filters.get("price_range", {})
        if price_range.get("min") or price_range.get("max"):
            price_filter = {"range": {"price": {}}}
            if price_range.get("min"):
                price_filter["range"]["price"]["gte"] = price_range["min"]
            if price_range.get("max"):
                price_filter["range"]["price"]["lte"] = price_range["max"]
            filter_clause.append(price_filter)
        
        size_range = filters.get("size_range", {})
        if size_range.get("min") or size_range.get("max"):
            size_filter = {"range": {"size": {}}}
            if size_range.get("min"):
                size_filter["range"]["size"]["gte"] = size_range["min"]
            if size_range.get("max"):
                size_filter["range"]["size"]["lte"] = size_range["max"]
            filter_clause.append(size_filter)
        
        # Add must-have features
        must_have = filters.get("must_have_features", [])
        for feature in must_have:
            filter_clause.append({
                "match": {
                    "key_features": feature
                }
            })
        
        # Add intelligent sorting
        if sort_pref == "price_asc":
            search_body["sort"] = [{"price": {"order": "asc", "missing": "_last"}}]
        elif sort_pref == "price_desc":
            search_body["sort"] = [{"price": {"order": "desc", "missing": "_last"}}]
        elif sort_pref == "size_asc":
            search_body["sort"] = [{"size": {"order": "asc", "missing": "_last"}}]
        elif sort_pref == "size_desc":
            search_body["sort"] = [{"size": {"order": "desc", "missing": "_last"}}]
        # Default: relevance (score) sorting
        
        await send_thinking(websocket, "executing_search", 
            f"Executing enhanced search with {len(filter_clause)} filters and '{sort_pref}' sorting...")
        
        response = await es.search(index="products", body=search_body)
        products = [format_product(hit["_source"]) for hit in response["hits"]["hits"]]
        
        await send_thinking(websocket, "retrieval_complete", 
            f"Enhanced retrieval complete. Found {len(products)} products. Applied {len(filter_clause)} filters. "
            f"Top result: '{products[0]['product_name'] if products else 'None'}'")
        
        return products
        
    except Exception as e:
        await send_thinking(websocket, "retrieval_error", f"Enhanced retrieval error: {str(e)}")
        return []

async def intelligent_reranking(original_query: str, products: List[Dict], websocket) -> List[Dict]:
    """Intelligent reranking with real-time LLM thinking"""
    
    if len(products) <= 3:
        return products
    
    await send_thinking(websocket, "reranking_analysis", 
        f"Starting intelligent reranking. I'll analyze each of the {len(products)} products against the query '{original_query}' to determine the best order...")
    
    try:
        # Create detailed product summaries for LLM
        product_summaries = []
        for i, product in enumerate(products[:15]):  # Limit for performance
            summary = f"{i}: {product['product_name']}"
            if product.get('price'):
                summary += f" (${product['price']})"
            if product.get('size'):
                summary += f" {product['size']}\""
            if product.get('key_features'):
                summary += f" - {product['key_features'][:100]}"
            product_summaries.append(summary)
        
        rerank_prompt = f"""You are an expert product ranker. Analyze these products for relevance to the query and rank them by best match.

Query: "{original_query}"

Products to rank:
{chr(10).join(product_summaries)}

Think step by step:
1. Which products best match the query intent?
2. Consider product type, features, price appropriateness, and overall suitability
3. Rank from most relevant to least relevant

Provide your ranking as a comma-separated list of numbers (most relevant first):
Example: "3,1,7,2,5,0,8,4,6,10,11,12,13,14"

Ranking:"""

        await send_thinking(websocket, "reranking_llm", 
            f"Sending {len(product_summaries)} product summaries to LLM for intelligent relevance ranking...")

        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert product ranker. Consider all factors and return only the ranking numbers."},
                {"role": "user", "content": rerank_prompt}
            ],
            max_tokens=100,
            temperature=0.1
        )
        
        ranking_text = response.choices[0].message.content.strip()
        
        try:
            # Parse ranking
            rankings = [int(x.strip()) for x in ranking_text.split(',') if x.strip().isdigit()]
            reranked = []
            
            # Apply ranking
            for rank in rankings:
                if 0 <= rank < len(products):
                    reranked.append(products[rank])
            
            # Add any missed products
            ranked_indices = set(rankings)
            for i, product in enumerate(products):
                if i not in ranked_indices:
                    reranked.append(product)
            
            await send_thinking(websocket, "reranking_success", 
                f"Reranking successful! LLM reordered products by relevance. "
                f"New top product: '{reranked[0]['product_name'] if reranked else 'None'}'. "
                f"Applied ranking: {rankings[:5]}...")
            
            return reranked
            
        except Exception as parse_error:
            await send_thinking(websocket, "reranking_parse_error", 
                f"Ranking parse failed: {str(parse_error)}. LLM returned: '{ranking_text}'. Using original order.")
            return products
            
    except Exception as e:
        await send_thinking(websocket, "reranking_error", 
            f"Reranking failed: {str(e)}. Using original relevance order.")
        return products

async def generate_intelligent_explanation(query: str, products: List[Dict], websocket) -> str:
    """Generate intelligent explanation with real product analysis"""
    
    if not products:
        return "No products found matching your criteria."
    
    await send_thinking(websocket, "explanation_analysis", 
        f"Analyzing top {min(len(products), 5)} products to generate personalized recommendations based on actual specifications...")
    
    try:
        # Create detailed product context
        top_products = products[:5]
        product_context = ""
        
        for i, product in enumerate(top_products, 1):
            context = f"""Product {i}: {product['product_name']}
Price: ${product.get('price', 'N/A')}
Size: {product.get('size', 'N/A')}"
Key Features: {product.get('key_features', 'No features listed')[:150]}
SKU: {product.get('sku', 'N/A')}

"""
            product_context += context
        
        explanation_prompt = f"""You are a knowledgeable product expert. Based on these specific product details, provide a helpful recommendation for the user's query.

User Query: "{query}"

Available Products:
{product_context}

Instructions:
1. Think about which product(s) best match the user's needs
2. Explain WHY these products are good matches (mention specific features, prices, sizes)
3. Provide actionable advice
4. Be conversational and helpful
5. Keep it concise (2-3 sentences max)

Your recommendation:"""

        await send_thinking(websocket, "explanation_generating", 
            "Generating personalized recommendation based on actual product specifications and user needs...")

        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful product expert. Be specific, mention actual product details, and provide actionable advice."},
                {"role": "user", "content": explanation_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        explanation = response.choices[0].message.content.strip()
        
        await send_thinking(websocket, "explanation_complete", 
            f"Generated personalized recommendation based on {len(top_products)} products' actual specifications. "
            f"Recommendation highlights the best matches and explains why they suit the user's needs.")
        
        return explanation
        
    except Exception as e:
        await send_thinking(websocket, "explanation_error", 
            f"Explanation generation failed: {str(e)}. Providing basic product summary.")
        
        # Fallback explanation
        if products:
            top = products[0]
            return f"Found {len(products)} products. Top recommendation: {top['product_name']} at ${top.get('price', 'N/A')} - {top.get('key_features', 'Good features')[:100]}."
        else:
            return "No suitable products found for your query."

def format_product(source: dict) -> dict:
    """Format product data for response"""
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
        "message": "Smart Keyword Search with LLM Fallback API",
        "features": [
            "Smart query analysis",
            "Enhanced keyword search with suggestors",
            "Typo correction and fuzzy matching", 
            "LLM fallback for complex queries",
            "Real-time thinking process",
            "Intelligent filtering and sorting",
            "AI-powered reranking",
            "RAG-based explanations"
        ]
    }

@app.get("/health")
async def health():
    es_status = "connected" if await es.ping() else "disconnected"
    llm_status = "available" if OPENAI_API_KEY else "unavailable"
    
    return {
        "status": "healthy",
        "elasticsearch": es_status,
        "llm": llm_status,
        "active_sessions": len(chat_sessions),
        "embedding_model": "all-MiniLM-L6-v2"
    }

@app.get("/stats")
async def get_stats():
    """Get search statistics"""
    try:
        # Get index stats
        stats = await es.indices.stats(index="products")
        doc_count = stats["indices"]["products"]["total"]["docs"]["count"]
        
        return {
            "total_products": doc_count,
            "active_chat_sessions": len(chat_sessions),
            "search_strategies": [
                "exact_phrase_matching",
                "completion_suggestors", 
                "fuzzy_search",
                "vector_similarity",
                "llm_enhanced_retrieval"
            ]
        }
    except Exception as e:
        return {"error": str(e)}

# Run with: uvicorn app.smart_main:app --reload --host 0.0.0.0 --port 8000