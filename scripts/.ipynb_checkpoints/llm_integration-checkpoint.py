from retrieve import HybridRetriever
from gpt4all import GPT4All

class NexusRAG:
    def __init__(self,
                 retriever_model='all-MiniLM-L6-v2',
                 llm_model_path='gpt4all-lora-quantized.bin'):
        self.retriever = HybridRetriever(model_name=retriever_model)
        self.llm = GPT4All(llm_model_path)

    def generate_answer(self, user_query, top_k=3):
        hits = self.retriever.hybrid_search(user_query, top_k=top_k)

        # Build context from hits
        context_blocks = []
        for h in hits:
            src = h["_source"]
            context_blocks.append(src["text"])

        context_str = "\n\n".join(context_blocks)
        prompt = f"""
You are an AI assistant with the following context:
{context_str}

User question: {user_query}

Please provide a helpful answer using only the above text. Cite relevant details if needed.
"""

        response = self.llm.generate(
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        return response