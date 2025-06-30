from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import json

es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def create_simple_hybrid_index():
    """Create index with a combined embedding"""
    
    index_mapping = {
        "mappings": {
            "properties": {
                "product_name": {"type": "text"},
                "sku": {"type": "keyword"},
                "price": {"type": "float", "ignore_malformed": True},
                "size": {"type": "integer"},
                "product_url": {"type": "keyword"},
                "image_urls": {"type": "keyword"},
                "key_features": {"type": "text"},
                "combined_embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    if es.indices.exists(index="products"):
        es.indices.delete(index="products")
        print("Deleted existing index")
    
    es.indices.create(index="products", body=index_mapping)
    print("✅ Hybrid index created!")

def add_combined_embeddings():
    """Add data with combined embeddings"""
    
    json_filename = "TVs_products.json"  # change if needed
    with open(json_filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for product in data['products']:
        features_text = " ".join(product['key_features'])
        
        # Combine all text for embedding
        combined_text = f"{product['product_name']} {features_text}"
        
        print(f"Creating embedding for: {product['product_name']}")
        combined_embedding = model.encode(combined_text).tolist()
        
        doc = {
            "product_name": product['product_name'],
            "sku": product['sku'],
            "price": product['price'],
            "size": product['size'],
            "product_url": product['product_url'],
            "image_urls": product['image_urls'],
            "key_features": features_text,
            "combined_embedding": combined_embedding
        }
        
        es.index(index="products", id=product['sku'], body=doc)
        print(f"✅ Indexed: {product['product_name']}")

def hybrid_search(query, boost_semantic=1.0, boost_keyword=1.0):
    """Hybrid search: semantic + keyword"""
    
    print(f"\n🔍 Searching for: '{query}'")
    query_embedding = model.encode(query).tolist()
    
    search_body = {
        "query": {
            "bool": {
                "should": [
                    # Semantic search
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'combined_embedding') + 1.0",
                                "params": {"query_vector": query_embedding}
                            },
                            "boost": boost_semantic
                        }
                    },
                    # Keyword search
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["product_name^2", "key_features"],
                            "fuzziness": "AUTO"
                        }
                    }
                ]
            }
        }
    }
    
    response = es.search(index="products", body=search_body)
    
    print(f"Total found: {response['hits']['total']['value']}")
    for hit in response['hits']['hits']:
        product = hit['_source']
        score = hit['_score']
        print(f"Score: {score:.2f} | {product['product_name']} | ${product['price']}")

def semantic_only_search(query):
    """Semantic search only"""
    print(f"\n🧠 Semantic only: '{query}'")
    query_embedding = model.encode(query).tolist()
    
    search_body = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'combined_embedding') + 1.0",
                    "params": {"query_vector": query_embedding}
                }
            }
        }
    }
    
    response = es.search(index="products", body=search_body)
    for hit in response['hits']['hits']:
        product = hit['_source']
        score = hit['_score']
        print(f"Score: {score:.2f} | {product['product_name']}")

def keyword_only_search(query):
    """Keyword search only"""
    print(f"\n🔤 Keyword only: '{query}'")
    
    search_body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["product_name^2", "key_features"],
                "fuzziness": "AUTO"
            }
        }
    }
    
    response = es.search(index="products", body=search_body)
    for hit in response['hits']['hits']:
        product = hit['_source']
        score = hit['_score']
        print(f"Score: {score:.2f} | {product['product_name']}")

# Setup and test
if __name__ == "__main__":
    print("Setting up hybrid search...")
    create_simple_hybrid_index()
    add_combined_embeddings()
    
    print("\n" + "="*60)
    print("🧪 TESTING DIFFERENT SEARCH TYPES")
    
    test_queries = [
        "gaming TV",
        "movie watching", 
        "77 inch OLED",
        "affordable large screen"
    ]
    
    for query in test_queries:
        print("\n" + "="*40)
        hybrid_search(query)
        semantic_only_search(query) 
        keyword_only_search(query)