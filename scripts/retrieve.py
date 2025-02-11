import requests
import json
from sentence_transformers import SentenceTransformer

INDEX_NAME = "documents"
OPENSEARCH_URL = "http://localhost:9200"

class HybridRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def bm25_search(self, query, top_k=5):
        body = {
            "query": {
                "match": {
                    "text": query
                }
            },
            "size": top_k
        }
        r = requests.get(f"{OPENSEARCH_URL}/{INDEX_NAME}/_search",
                         data=json.dumps(body),
                         headers={"Content-Type": "application/json"})
        return r.json()

    def vector_search(self, query, top_k=5):
        query_emb = self.model.encode(query).tolist()
        body = {
            "size": top_k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_emb,
                        "k": top_k,
                        "num_candidates": 50
                    }
                }
            }
        }
        r = requests.post(f"{OPENSEARCH_URL}/{INDEX_NAME}/_search",
                          data=json.dumps(body),
                          headers={"Content-Type": "application/json"})
        return r.json()

    def hybrid_search(self, query, top_k=5):
        bm25_res = self.bm25_search(query, top_k=top_k)
        vector_res = self.vector_search(query, top_k=top_k)

        combined = {}
        # Merge BM25 hits
        if "hits" in bm25_res and "hits" in bm25_res["hits"]:
            for hit in bm25_res["hits"]["hits"]:
                combined[hit["_id"]] = hit
        # Merge Vector hits
        if "hits" in vector_res and "hits" in vector_res["hits"]:
            for hit in vector_res["hits"]["hits"]:
                combined[hit["_id"]] = hit

        return list(combined.values())