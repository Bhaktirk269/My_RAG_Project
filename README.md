# Knowledge-Base Search Engine (RAG)

A robust Retrieval-Augmented Generation (RAG) system that allows you to chat with your PDF documents locally. This project leverages the power of Llama 3 and Nomic embeddings for high-performance, private document intelligence.

## 🚀 Features
* **Document Ingestion:** PDF processing and intelligent text chunking.
* **Vector Embeddings:** Local text vectorization using `nomic-embed-text`.
* **RAG Pipeline:** Context-aware retrieval powered by **ChromaDB**.
* **Synthesis:** Succinct, high-quality answer generation via **Llama 3**.

## 🛠 Tech Stack
* **Backend:** FastAPI (Python)
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **LLM Engine:** Ollama (Llama 3 & Nomic-Embed-Text)

## 📋 Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed and running.

### Prepare the AI Models
Open your terminal and pull the necessary models:
```bash
ollama pull llama3
ollama pull nomic-embed-text
