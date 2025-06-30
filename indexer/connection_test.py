from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Test connection
try:
    if es.ping():
        print("✅ Elasticsearch connection successful!")
        
        # Print info
        info = es.info()
        print(f"Name: {info['name']}")
        print(f"Version: {info['version']['number']}")
        print(f"Cluster: {info['cluster_name']}")
        
    else:
        print("❌ Connection failed")
        
except Exception as e:
    print(f"Error: {e}")
    print("Please install elasticsearch package: pip install elasticsearch")