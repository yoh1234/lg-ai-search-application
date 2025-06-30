# backend/load_data_complete.py
import asyncio
import json
import os
import glob
from elasticsearch import AsyncElasticsearch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import time

class DataLoader:
    def __init__(self, es_url: str = "http://localhost:9200"):
        self.es = AsyncElasticsearch([es_url])
        self.index_name = "products"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def setup_index(self):
        """Create index and set up mappings"""
        print("🔧 Setting up index...")
        
        # Delete existing index (optional)
        try:
            await self.es.indices.delete(index=self.index_name)
            print("  Existing index deleted")
        except:
            print("  No existing index found")
        
        # Create new index mapping
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "autocomplete": {
                            "tokenizer": "autocomplete",
                            "filter": ["lowercase"]
                        },
                        "autocomplete_search": {
                            "tokenizer": "lowercase"
                        }
                    },
                    "tokenizer": {
                        "autocomplete": {
                            "type": "edge_ngram",
                            "min_gram": 2,
                            "max_gram": 10,
                            "token_chars": ["letter", "digit"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "product_name": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "autocomplete": {
                                "type": "text", 
                                "analyzer": "autocomplete",
                                "search_analyzer": "autocomplete_search"
                            }
                        }
                    },
                    "sku": {
                        "type": "keyword"
                    },
                    "price": {
                        "type": "keyword"
                    },
                    "size": {
                        "type": "keyword"
                    },
                    "product_url": {
                        "type": "keyword"
                    },
                    "image_urls": {
                        "type": "keyword"
                    },
                    "key_features": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "key_features_combined": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "product_type": {
                        "type": "keyword"
                    },
                    # Embedding 필드
                    "combined_embedding": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    },
                    # Suggester 필드들
                    "product_name_suggest": {
                        "type": "completion",
                        "analyzer": "simple",
                        "preserve_separators": True,
                        "preserve_position_increments": True,
                        "max_input_length": 50,
                        "contexts": [
                            {
                                "name": "product_type_context",
                                "type": "category"
                            }
                        ]
                    },
                    "key_features_suggest": {
                        "type": "completion",
                        "analyzer": "simple",
                        "max_input_length": 100
                    },
                    "sku_suggest": {
                        "type": "completion",
                        "analyzer": "keyword",
                        "max_input_length": 20
                    }
                }
            }
        }
        
        await self.es.indices.create(index=self.index_name, body=mapping)
        print("✅ New index created successfully!")
        
    def load_json_files(self, data_directory: str = "data") -> List[Dict]:
        """Load JSON files from directory"""
        print(f"📁 Searching for JSON files in {data_directory} folder...")
        
        json_files = glob.glob(os.path.join(data_directory, "*.json"))
        print(f"  Found {len(json_files)} JSON files")
        
        all_products = []
        
        for json_file in json_files:
            print(f"  📄 Loading: {json_file}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract metadata
                metadata = data.get("metadata", {})
                product_type = metadata.get("Type", "unknown").lower()
                products = data.get("products", [])
                
                print(f"    Type: {product_type}, Products: {len(products)}")
                
                # Add type info to each product (preserve metadata)
                for product in products:
                    product["product_type"] = product_type
                    product["metadata_type"] = metadata.get("Type", "unknown")  # Preserve original Type value
                    
                all_products.extend(products)
                
            except Exception as e:
                print(f"❌ Failed to load {json_file}: {e}")
        
        print(f"📦 Total {len(all_products)} products loaded!")
        return all_products
    
    def process_product_data(self, products: List[Dict]) -> List[Dict]:
        """Process product data without preprocessing price/size"""
        print("🔄 Processing product data...")
        
        processed = []
        
        for product in products:
            # Process key_features (list to string)
            key_features = product.get("key_features", [])
            if isinstance(key_features, list):
                key_features_combined = " ".join(str(feature) for feature in key_features if feature)
            else:
                key_features_combined = str(key_features) if key_features else ""
            
            # Process image_urls
            image_urls = product.get("image_urls", "")
            if isinstance(image_urls, str):
                image_urls = [image_urls] if image_urls else []
            
            # Keep price as is - if null or N/A, keep as empty string or original value
            price = product.get("price", "")
            if price is None:
                price = ""
            
            # Keep size as is - if null, keep as empty string or original value  
            size = product.get("size", "")
            if size is None:
                size = ""
            
            processed_product = {
                "product_name": product.get("product_name", ""),
                "sku": product.get("sku", ""),
                "price": price,
                "size": size,
                "product_url": product.get("product_url", ""),
                "image_urls": image_urls,
                "key_features": key_features_combined,
                "key_features_combined": key_features_combined,
                "product_type": product.get("product_type", "unknown"),
                "metadata_type": product.get("metadata_type", "unknown")
            }
            
            processed.append(processed_product)
        
        print(f"✅ Processed {len(processed)} products!")
        return processed
    
    def generate_embeddings(self, products: List[Dict]) -> List[Dict]:
        """Generate embeddings including metadata Type, product name, and key_features"""
        print("🧠 Generating embeddings...")
        
        # Generate embeddings in batches for performance optimization
        texts = []
        for product in products:
            # Combine metadata Type + product name + key features for embedding
            metadata_type = product.get('metadata_type', '')
            product_name = product.get('product_name', '')
            key_features = product.get('key_features_combined', '')
            
            # Combine only meaningful text
            text_parts = []
            if metadata_type and metadata_type != 'unknown':
                text_parts.append(metadata_type)
            if product_name:
                text_parts.append(product_name)
            if key_features:
                text_parts.append(key_features)
            
            combined_text = " ".join(text_parts)
            texts.append(combined_text if combined_text.strip() else "unknown product")
        
        print(f"  Generating embeddings for {len(texts)} texts...")
        print(f"  Example embedding text: '{texts[0][:100]}...'")
        
        # Batch embedding generation (fast)
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Add embeddings to products
        for i, product in enumerate(products):
            product["combined_embedding"] = embeddings[i].tolist()
        
        print("✅ Embedding generation complete!")
        return products
    
    def generate_suggestions(self, products: List[Dict]) -> List[Dict]:
        """Generate suggester data"""
        print("💡 Generating suggester data...")
        
        for product in products:
            product_name = product.get("product_name", "")
            sku = product.get("sku", "")
            key_features = product.get("key_features_combined", "")
            product_type = product.get("product_type", "")
            
            # 1. Product Name Suggestions
            name_inputs = []
            if product_name:
                name_inputs.append(product_name)
                # Split product name by words
                name_words = [word.strip() for word in product_name.lower().split() if len(word.strip()) > 2]
                name_inputs.extend(name_words)
            
            product["product_name_suggest"] = {
                "input": list(set(name_inputs)),
                "weight": 10,
                "contexts": {
                    "product_type_context": product_type
                }
            }
            
            # 2. Key Features Suggestions
            feature_inputs = []
            if key_features:
                # Extract key words
                feature_words = []
                for word in key_features.lower().split():
                    word = word.strip('.,!?()[]"\'')
                    if len(word) > 3 and word not in ['with', 'from', 'that', 'this', 'your']:
                        feature_words.append(word)
                
                feature_inputs = list(set(feature_words[:15]))  # Top 15
            
            product["key_features_suggest"] = {
                "input": feature_inputs,
                "weight": 5
            }
            
            # 3. SKU Suggestions
            sku_inputs = []
            if sku:
                sku_inputs.append(sku)
                sku_inputs.append(sku.lower())
            
            product["sku_suggest"] = {
                "input": sku_inputs,
                "weight": 15  # SKU has highest weight
            }
        
        print("✅ Suggester data generation complete!")
        return products
    
    async def bulk_index_products(self, products: List[Dict]):
        """Bulk index product data to Elasticsearch"""
        print(f"📤 Indexing {len(products)} products to Elasticsearch...")
        
        # Prepare bulk operations
        actions = []
        for product in products:
            action = {
                "_index": self.index_name,
                "_id": product["sku"] if product["sku"] else f"product_{len(actions)}",  # Auto ID if no SKU
                "_source": product
            }
            actions.append(action)
        
        # Index in batches to save memory
        batch_size = 50
        total_batches = (len(actions) + batch_size - 1) // batch_size
        
        for i in range(0, len(actions), batch_size):
            batch = actions[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"  Indexing batch {batch_num}/{total_batches}... ({len(batch)} products)")
            
            try:
                # Use Bulk API
                body = []
                for action in batch:
                    body.append({"index": {"_index": action["_index"], "_id": action["_id"]}})
                    body.append(action["_source"])
                
                response = await self.es.bulk(body=body)
                
                # Check for errors
                if response["errors"]:
                    print(f"    ⚠️  Some documents failed to index")
                    for item in response["items"]:
                        if "error" in item.get("index", {}):
                            print(f"      Error: {item['index']['error']}")
                else:
                    print(f"    ✅ Batch {batch_num} successful")
                
            except Exception as e:
                print(f"    ❌ Batch {batch_num} failed: {e}")
        
        print("📤 Indexing complete!")
    
    async def test_search_features(self):
        """Test search functionality"""
        print("\n🧪 Testing search features...")
        
        # 1. Basic search test
        print("  1. Basic search test...")
        search_response = await self.es.search(
            index=self.index_name,
            body={"query": {"match": {"product_name": "TV"}}, "size": 3}
        )
        print(f"    'TV' search results: {search_response['hits']['total']['value']} items")
        
        # 2. Completion Suggester test
        print("  2. Completion Suggester test...")
        suggest_response = await self.es.search(
            index=self.index_name,
            body={
                "suggest": {
                    "product_suggest": {
                        "prefix": "lg",
                        "completion": {
                            "field": "product_name_suggest",
                            "size": 5
                        }
                    }
                }
            }
        )
        
        suggestions = suggest_response["suggest"]["product_suggest"][0]["options"]
        print(f"    'lg' prefix suggestions: {len(suggestions)} items")
        for suggestion in suggestions[:3]:
            print(f"      - {suggestion['text']}")
        
        # 3. Vector search test (includes metadata Type)
        print("  3. Vector search test...")
        test_query = "gaming monitor with high refresh rate"
        query_vector = self.embedding_model.encode(test_query).tolist()
        
        vector_response = await self.es.search(
            index=self.index_name,
            body={
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'combined_embedding') + 1.0",
                            "params": {"query_vector": query_vector}
                        }
                    }
                },
                "size": 3
            }
        )
        
        print(f"    '{test_query}' vector search: {vector_response['hits']['total']['value']} items")
        
        # 4. Data quality check
        print("  4. Data quality check...")
        
        # Check empty price count
        empty_price_response = await self.es.search(
            index=self.index_name,
            body={
                "query": {"term": {"price": ""}},
                "size": 0
            }
        )
        empty_prices = empty_price_response['hits']['total']['value']
        
        # Check empty size count
        empty_size_response = await self.es.search(
            index=self.index_name,
            body={
                "query": {"term": {"size": ""}},
                "size": 0
            }
        )
        empty_sizes = empty_size_response['hits']['total']['value']
        
        print(f"    Products with empty price: {empty_prices}")
        print(f"    Products with empty size: {empty_sizes}")
        
        print("✅ All feature tests complete!")
        response = await self.es.search(
            index=self.index_name,
            body={
                "query": {"bool": {"must_not": {"exists": {"field": "size"}}}},
                "size": 0
            }
        )
        
        print("✅ 모든 기능 테스트 완료!")
    
    async def close(self):
        """Close connection"""
        await self.es.close()

async def main():
    """Main execution function"""
    print("🚀 Starting LG product data loading!")
    print("=" * 50)
    
    loader = DataLoader()
    
    try:
        # 1. Set up index
        await loader.setup_index()
        
        # 2. Load JSON files
        products = loader.load_json_files("C:/Users/chzhf/Downloads/lg-interview/lg-ai-search-application/indexer/data")  # Load JSON files from data folder
        
        if not products:
            print("❌ No product data found. Please check if JSON files exist in 'data' folder.")
            return
        
        # 3. Process data (keep original values for price/size)
        products = loader.process_product_data(products)
        
        # 4. Generate embeddings (includes metadata Type)
        products = loader.generate_embeddings(products)
        
        # 5. Generate suggester data
        products = loader.generate_suggestions(products)
        
        # 6. Index to Elasticsearch
        await loader.bulk_index_products(products)
        
        # 7. Test search features
        await loader.test_search_features()
        
        print("\n🎉 All tasks completed!")
        print(f"📊 Total {len(products)} products successfully loaded.")
        print(f"🔍 Index name: {loader.index_name}")
        print("🎯 Available features:")
        print("  - Keyword search")
        print("  - Vector similarity search (metadata Type + product name + features)")
        print("  - Completion/Phrase/Term Suggester")
        print("  - Hybrid search")
        print("  - Original value preservation (price N/A, size null)")
        
    except Exception as e:
        print(f"❌ Error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await loader.close()

if __name__ == "__main__":
    
    asyncio.run(main())