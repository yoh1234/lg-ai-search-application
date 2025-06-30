from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

def search_products(query):
    """Simple search function"""
    search_body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["product_name^2", "key_features"],  # product_name has 2x weight
                "fuzziness": "AUTO"  # handles typos
            }
        }
    }
    
    response = es.search(index="products", body=search_body)
    
    print(f"\n=== Search results for '{query}' ===")
    print(f"Total found: {response['hits']['total']['value']}")
    
    for hit in response['hits']['hits']:
        product = hit['_source']
        score = hit['_score']
        
        print(f"\nScore: {score:.2f}")
        print(f"Product: {product['product_name']}")
        print(f"Price: ${product['price']}")
        print(f"Size: {product['size']} inches")
        print(f"SKU: {product['sku']}")

def search_with_filters(query, min_price=None, max_price=None, size_filter=None):
    """Search with price and size filters"""
    
    # Base query
    must_query = {
        "multi_match": {
            "query": query,
            "fields": ["product_name^2", "key_features"],
            "fuzziness": "AUTO"
        }
    }
    
    # Add filters
    filter_conditions = []
    if min_price is not None:
        filter_conditions.append({"range": {"price": {"gte": min_price}}})
    if max_price is not None:
        filter_conditions.append({"range": {"price": {"lte": max_price}}})
    if size_filter is not None:
        filter_conditions.append({"term": {"size": size_filter}})
    
    search_body = {
        "query": {
            "bool": {
                "must": must_query,
                "filter": filter_conditions
            }
        }
    }
    
    response = es.search(index="products", body=search_body)
    
    print(f"\n=== Filtered search: '{query}' ===")
    if min_price: print(f"Min price: ${min_price}")
    if max_price: print(f"Max price: ${max_price}")
    if size_filter: print(f"Size: {size_filter} inches")
    print(f"Total found: {response['hits']['total']['value']}")
    
    for hit in response['hits']['hits']:
        product = hit['_source']
        score = hit['_score']
        print(f"\nScore: {score:.2f}")
        print(f"Product: {product['product_name']}")
        print(f"Price: ${product['price']}")

# Test different searches
print("🔍 Testing basic searches...")

test_queries = [
    "OLED",
    "gaming",
    "83 inch", 
    "Dolby",
    "4K"
]

for query in test_queries:
    search_products(query)

print("\n" + "="*60)
print("🔍 Testing filtered searches...")

# Test with filters
search_with_filters("TV", max_price=2500)
search_with_filters("OLED", size_filter=77)

print("\n" + "="*60)
print("✅ Search testing complete!")