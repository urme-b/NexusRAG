NexusRAG (Nexus Retrieval Augmented Generation) is a locally hosted AI assistant that ingests your private PDFs (including scanned documents) and answers natural-language questions offline—without sending data to any external service.

This is achieved through:
1. OCR (Optical Character Recognition) for scanned files.
2. Hybrid retrieval (keyword + vector embeddings).
3. A local Large Language Model to synthesize final answers.
4. A simple Streamlit user interface to query your system.


Core Concepts & Components
1. OCR & PDF Extraction
- Scanned PDFs: recognized via Tesseract OCR
- Normal PDFs: extracted via PyMuPDF (fitz).

2. Chunking & Indexing
- Large documents are broken into smaller segments (~300 words).
- OpenSearch is configured with BM25 for keywords & knn_vector for embeddings.

3. Hybrid Retrieval
- Combines keyword search with vector search (via Sentence Transformers).
- Produces top-ranked chunks relevant to the user’s question.

4. Local LLM
- Offline text-generation model (e.g., GPT4All) synthesizes a final answer referencing retrieved chunks.

5. Streamlit UI
- Minimalistic web interface for typing queries and receiving answers, all running locally.


System Architecture
1. Raw PDFs → data/raw/
2. OCR & Extraction → .txt files in data/processed/
3. Chunking → text segments stored in data/chunks/
4. Index Setup → create a local OpenSearch index with BM25 + vector fields
5. Embedding & Bulk Index → chunk embeddings + metadata stored in OpenSearch
6. Query → user asks a question in Streamlit
7. Hybrid Retrieval → BM25 & vector results are merged
8. Local LLM → top chunks form the prompt, LLM answers offline
9. Answer Display → Streamlit shows the final response


Scripts
- process_all_docs.py: Detects normal vs. scanned PDFs; runs extraction & OCR, outputs .txt.
- chunking.py: Splits .txt files into smaller segments for more precise retrieval.
- setup_index.py & index_chunks.py: Creates OpenSearch index, generates vector embeddings, and bulk indexes chunk data.
- retrieval.py: Hybrid retrieval logic (BM25 + vector).
- llm_integration.py: Combines retrieved text with a local LLM to craft an answer.
- app.py: Streamlit UI for user queries and response display.


Key Advantages & Future Directions
1. Full Data Privacy: No external servers—crucial for sensitive or proprietary documents.
2. Accurate Retrieval: Combines BM25 (keyword) + vector embeddings (semantic search), capturing both exact matches and conceptual relationships.
3. Domain-Specific Answers: Because the local LLM references only your docs, answers are highly tuned to your domain.
4. Potential Enhancements:
- Neural Re-Ranking: Use a cross-encoder to reorder final results for even better precision.
- Fine-Tuning: Adapt your Sentence Transformer or local LLM to your specialized domain.
- Access Control: If certain docs are restricted to certain roles.
- Advanced Summarization: Summarize large docs or multi-chunk answers if your LLM context is limited.


Final Notes
- Tested with a small corpus of PDFs (both normal and scanned) to confirm OCR accuracy, chunk-based retrieval, and local LLM integration.
- For large-scale deployments or specialized domains, you can tune chunk sizes, re-ranking logic, or LLM prompts.
- If you have any issues, please open an issue in this repository or consult the docs/ folder for additional usage examples.

NexusRAG aims to streamline private document Q&A with an offline, AI-driven approach—helping you harness your own text corpus without compromising security or compliance.
