# Sandbox Policy Assistant (RAG)

A Retrieval-Augmented Generation (RAG) application to answer questions based on Sandbox company policy documents.

## Features
- **Document Ingestion**: Processes Markdown and PDF documents.
- **Vector Search**: Uses ChromaDB and HuggingFace local embeddings.
- **LLM Engine**: Powered by Groq (Llama-3) with automatic OpenRouter fallback.
- **Web Interface**: Simple, responsive chat UI with source citations.
- **Evaluation**: Built-in metrics for groundedness, citation accuracy, and latency.
- **CI/CD**: GitHub Actions pipeline with automated tests on push/PR.
- **Deployment**: Configured for Render via `render.yaml`.

## Tech Stack
- **Backend**: Flask + Gunicorn
- **Orchestration**: LangChain
- **LLM (Primary)**: Groq API (`llama-3.3-70b-versatile`)
- **LLM (Fallback)**: OpenRouter (`meta-llama/llama-3.3-70b-instruct`)
- **Embeddings**: Sentence-Transformers (local, `all-MiniLM-L6-v2`)
- **Database**: ChromaDB

## Setup Instructions

### 1. Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/)
- (Optional) An [OpenRouter API Key](https://openrouter.ai/) for fallback LLM support

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Configuration
Copy `.env.example` to `.env` and add your API keys:
```bash
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here  # optional fallback
```

### 4. Ingestion
Process the company documents into the vector database:
```bash
python ingest.py
```

### 5. Running the Application
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

## Deployment (Render)
The app is configured for Render via `render.yaml`. Set `GROQ_API_KEY` and `OPENROUTER_API_KEY` as environment variables in the Render dashboard. The start command is:
```bash
gunicorn --bind 0.0.0.0:$PORT app:app
```

## Evaluation
Run the evaluation script (30 queries, LLM-as-a-judge):
```bash
python evaluate.py
```

## Repository Structure
- `corpus/`: Policy documents (Markdown).
- `app.py`: Flask backend and API endpoints (`/`, `/chat`, `/health`).
- `ingest.py`: Document processing and vector storage logic.
- `rag_engine.py`: LLM and retrieval orchestration with primary/fallback support.
- `evaluate.py`: Evaluation script (groundedness, citation accuracy, latency).
- `db/`: Local ChromaDB vector store.
- `templates/`: HTML chat interface.
- `tests/`: Smoke tests run by CI.
- `render.yaml`: Render deployment configuration.
- `design-and-evaluation.md`: Architecture decisions and evaluation results.
- `ai-tooling.md`: AI tools used during development.
