import os
import requests
import json
from sentence_transformers import SentenceTransformer

INDEX_NAME = "documents"
OPENSEARCH_URL = "http://localhost:9200"

def index_chunks(chunk_dir, model_name='all-MiniLM-L6-v2'):
    """
    For each .txt chunk in chunk_dir, generate an embedding, then index
    'text', 'vector', and 'metadata' into the 'documents' index.
    """
    model = SentenceTransformer(model_name)
    bulk_data = []

    for filename in os.listdir(chunk_dir):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(chunk_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        embedding = model.encode(text).tolist()
        # parse chunk id from filename if desired
        base_name = filename.replace(".txt", "")
        try:
            chunk_id = int(base_name.split('_')[-1])
        except:
            chunk_id = 0

        doc_body = {
            "text": text,
            "vector": embedding,
            "metadata": {
                "chunk_id": chunk_id,
                "source_file": filename
            }
        }

        action = {"index": {"_index": INDEX_NAME}}
        bulk_data.append(json.dumps(action))
        bulk_data.append(json.dumps(doc_body))

    if bulk_data:
        bulk_payload = "\n".join(bulk_data) + "\n"
        resp = requests.post(f"{OPENSEARCH_URL}/_bulk",
                             data=bulk_payload,
                             headers={"Content-Type": "application/json"})
        print(resp.json())
    else:
        print("No .txt chunks found in", chunk_dir)