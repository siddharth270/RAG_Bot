# RAG_Bot

Recently, I worked on building a RAG-based chatbot that takes in any document or PDF, reads through it, and answers questions based specifically on that content just like a focused assistant that doesn’t hallucinate outside the given scope.

The best part?
No paid APIs. All open-source and completely free.

Here’s what powered it:

✅PyPDF2 – to break down and parse the PDF files

✅HuggingFace Transformers – used the sentence-transformers/all-MiniLM-L6-v2 model for chunk-level vector embeddings

✅FAISS – for blazing-fast similarity search across vectorized content

✅Ollama – to run local LLM inference efficiently


