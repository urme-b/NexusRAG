import requests
import json

INDEX_NAME = "documents"
OPENSEARCH_URL = "http://localhost:9200"

def create_index():
    """
    Creates or resets an index named 'documents' for BM25 + vector search.
    """
    url = f"{OPENSEARCH_URL}/{INDEX_NAME}"
    mapping = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},   # BM25
                "vector": {
                    "type": "knn_vector",
                    "dimension": 384        # match embedding model dimension
                },
                "metadata": {
                    "properties": {
                        "chunk_id": {"type": "integer"},
                        "source_file": {"type": "keyword"}
                    }
                }
            }
        }
    }

    # Optional: delete existing index
    requests.delete(url)
    resp = requests.put(url, data=json.dumps(mapping),
                        headers={"Content-Type": "application/json"})
    print(resp.json())