from elasticsearch import Elasticsearch
import json

# Connect to Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

print("Step 1: Creating index...")

# Create index mapping
index_mapping = {
    "mappings": {
        "properties": {
            "product_name": {"type": "text"},
            "sku": {"type": "keyword"},
            "price": {"type": "float", "ignore_malformed": True},
            "size": {"type": "integer"},
            "product_url": {"type": "keyword"},
            "image_urls": {"type": "keyword"},
            "key_features": {"type": "text"}
        }
    }
}

# Delete existing index if exists
if es.indices.exists(index="products"):
    es.indices.delete(index="products")
    print("Deleted existing index")

# Create new index
es.indices.create(index="products", body=index_mapping)
print("✅ Index created successfully!")

print("\nStep 2: Loading data from JSON...")

# Load JSON file (change filename if needed)
json_filename = "TVs_products.json"  # 👈 your JSON file name here

try:
    with open(json_filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f"✅ Loaded JSON file: {json_filename}")
except FileNotFoundError:
    print(f"❌ File not found: {json_filename}")
    print("Please check the filename!")
    exit()

print("\nStep 3: Inserting products...")

# Process and insert each product
for product in data['products']:
    # Combine key_features array into single text
    features_text = " ".join(product['key_features'])
    
    # Prepare document
    doc = {
        "product_name": product['product_name'],
        "sku": product['sku'],
        "price": product['price'],
        "size": product['size'],
        "product_url": product['product_url'],
        "image_urls": product['image_urls'],
        "key_features": features_text
    }
    
    # Insert into Elasticsearch
    response = es.index(
        index="products",
        id=product['sku'],
        body=doc
    )
    print(f"✅ Inserted: {product['product_name']}")

print(f"\n🎉 Successfully inserted {len(data['products'])} products!")
print("Ready for search testing!")