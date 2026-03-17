# Sandbox Policy Assistant (RAG)

A Retrieval-Augmented Generation (RAG) application to answer questions based on Sandbox company policy documents.

## Features
- **Document Ingestion**: Processes Markdown and PDF documents.
- **Vector Search**: Uses ChromaDB and HuggingFace local embeddings.
- **LLM Engine**: Powered by Groq (Llama-3 models).
- **Web Interface**: Simple, responsive chat UI.
- **Evaluation**: Built-in metrics for latency and accuracy.

## Tech Stack
- **Backend**: Flask
- **Orchestration**: LangChain
- **LLM**: Groq API
- **Embeddings**: Sentence-Transformers (Local)
- **Database**: ChromaDB

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- A [Groq API Key](https://console.groq.com/)

### 2. Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Copy `.env.example` to `.env` and add your Groq API key:
```bash
GROQ_API_KEY=your_api_key_here
```

### 4. Ingestion
Process the company documents into the vector database:
```bash
python ingest.py
```

### 5. Running the Application
Start the Flask server:
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

## Evaluation
Run the evaluation script to see system metrics:
```bash
python evaluate.py
```

## Repository Structure
- `corpus/`: Policy documents (Markdown).
- `app.py`: Flask backend and API endpoints.
- `ingest.py`: Document processing and vector storage logic.
- `rag_engine.py`: LLM and retrieval orchestration.
- `evaluate.py`: Performance and accuracy testing.
- `db/`: Local Chroma database (generated after ingestion).
- `templates/`: HTML interface.
