from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

res = es.search(index="products", size=1)
for doc in res["hits"]["hits"]:
    print(doc["_source"])