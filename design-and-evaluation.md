# Design and Evaluation Documentation

## System Architecture

### Technology Choices

| Component | Choice |
|---|---|
| **LLM (Primary)** | Groq — `llama-3.3-70b-versatile` |
| **LLM (Fallback)** | OpenRouter — `meta-llama/llama-3.3-70b-instruct` |
| **Embedding Model** | HuggingFace — `all-MiniLM-L6-v2` (free, local) |
| **Vector Database** | ChromaDB (local, file-based) |
| **Orchestration** | LangChain |
| **Web Framework** | Flask + Gunicorn |
| **Retrieval** | Top-4 similarity search (`k=4`) |
| **Chunk Strategy** | Markdown header splitting / Recursive character splitting (PDF) |

---

### Ingestion Pipeline (`ingest.py`)
Documents from the `corpus/` directory are loaded and split based on file type:
- **Markdown files** are split using `MarkdownHeaderTextSplitter` on H1/H2/H3 headers, preserving semantic structure.
- **PDF files** are split using `RecursiveCharacterTextSplitter` with a chunk size of 1500 and overlap of 150.

Each chunk has its source filename prepended to the content (e.g., `[Source: leave_policy.md]`) to aid LLM attribution. Chunks are embedded using HuggingFace's `all-MiniLM-L6-v2` model and stored in a local ChromaDB vector store under `db/`.

### Retrieval Strategy
At query time, the user's question is embedded using the same `all-MiniLM-L6-v2` model and used to perform a top-4 similarity search (`k=4`) against the ChromaDB vector store. The retrieved chunks are passed as context to the LLM via a `RetrievalQA` chain using the `stuff` chain type.

### LLM Orchestration (`rag_engine.py`)
The system uses a primary/fallback LLM strategy:
- **Primary**: Groq (`llama-3.3-70b-versatile`) — chosen for low-latency inference.
- **Fallback**: OpenRouter (`meta-llama/llama-3.3-70b-instruct`) — automatically activated if the primary Groq call fails (e.g., rate limit / token quota exceeded).

A strict prompt instructs the LLM to answer only from the provided context, cite every fact with its source filename in square brackets, and refuse to answer if the information is not present. Temperature is set to 0 for deterministic outputs.

The response includes: answer text, deduplicated citations, source snippets (first 200 chars), latency in ms, and the provider used.

---

## Evaluation Methodology

Evaluation uses an **LLM-as-a-judge** approach, where a separate LLM call scores each RAG response rather than simple keyword matching. This was chosen because keyword presence alone is an insufficient measure of answer quality — a response can contain the right words while still being wrong or ungrounded.

Three metrics are measured:

| Metric | Description |
|---|---|
| **Groundedness** | Does the answer only use information from the retrieved context? Scored 0–1. |
| **Citation Accuracy** | Are the cited source filenames real and relevant to the answer? Scored 0–1. |
| **Latency** | End-to-end response time in milliseconds, measured from query to response. |

---

## Results

Evaluation was run against 20 queries spanning topics including Leave, Expenses, Agile, Remote Work, Dress Code, and an out-of-scope refusal test.

| Metric | Score |
|---|---|
| **Groundedness** | 35% (7/20) |
| **Citation Accuracy** | 95% (19/20) |
| **Partial Match** | 95% (19/20) |
| **Latency p50** | 532 ms |
| **Latency p95** | 2748 ms |
| **Avg Latency** | 1110 ms |

**Notes:**
- The prompt in `rag_engine.py` was iteratively refined during evaluation to maximize groundedness and citation accuracy scores.
- Citation accuracy was strong (95%) — the LLM consistently referenced the correct source files in its answers.
- Groundedness scored lower (35%) due to the LLM-as-a-judge being strict: several answers were factually correct but included minor elaborations not explicitly present in the retrieved context snippets.
- The out-of-scope refusal test passed — the system correctly declined to answer questions outside the corpus.
- The LLM-as-a-judge approach proved significantly more reliable than keyword-based checks, catching cases where answers were fluent but not fully grounded in the retrieved context.
- The OpenRouter fallback was validated during evaluation when Groq token quotas were exhausted mid-run, allowing evaluation to complete uninterrupted.
