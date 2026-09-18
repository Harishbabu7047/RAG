# Pure RAG (Retrieval-Augmented Generation)

A clean, high-performance, modular RAG implementation built in pure Python and FastAPI.

---

## Features

- **Zero-Bloat RAG**: Direct HTTP REST calls to Google Gemini API for embeddings and generation.
- **Pure Python Chunker**: Recursive character text splitter without third-party framework overhead.
- **Multi-Format Ingestion**: Extracts text from `.pdf`, `.txt`, and `.md` using standard `pypdf`.
- **Date-Partitioned Storage**: Automatically organizes document uploads by date (`uploads/YYYY-MM-DD/`).
- **Flexible Vector Storage**:
  - **In-Memory Store** with Cosine Similarity (built-in, zero-setup).
  - Optional **MongoDB Atlas Vector Search** integration.
- **Resilient Generation**: Automatic retry and multi-model fallback chain to handle transient traffic spikes (503 / 429).
- **FastAPI Endpoints**: Full REST API with Swagger documentation and streaming capabilities.

---

## Project Structure

```
├── app/
│   ├── api/
│   │   ├── routes.py           # FastAPI endpoints (/rag/ingest, /rag/upload, /rag/query, /rag/stream, /rag/stats)
│   │   └── schemas.py          # Pydantic request and response schemas
│   ├── core/
│   │   └── config.py           # App configuration and environment variables
│   ├── db/
│   │   └── mongodb.py          # Optional MongoDB Atlas vector storage client
│   ├── rag/
│   │   ├── document_loader.py  # PDF and text document extraction
│   │   ├── text_splitter.py    # Pure Python recursive text splitter
│   │   ├── embeddings.py       # Direct Google Gemini REST API embedder
│   │   ├── vector_store.py     # In-memory cosine similarity & Atlas Vector Store
│   │   └── pipeline.py         # End-to-end RAG pipeline
│   └── main.py                 # FastAPI application
├── uploads/                    # Uploads directory partitioned by date
├── requirements.txt            # Minimal dependencies
├── server.py                   # Server entrypoint
└── .env.example                # Example configuration template
```

---

## Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Harishbabu7047/RAG.git
cd RAG
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and set your Google Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run the Server
```bash
python server.py
```
The server will start on `http://127.0.0.1:8000`.

### 4. Interactive API Documentation
Open your browser at:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Endpoints

- `POST /rag/ingest`: Ingests all documents from the `uploads/` directory.
- `POST /rag/upload`: Uploads and immediately indexes a new `.pdf`, `.txt`, or `.md` file into today's date folder.
- `POST /rag/query`: Answers questions grounded strictly in the retrieved context, citing sources.
- `POST /rag/stream`: Streams answer tokens using Server-Sent Events (SSE).
- `GET /rag/stats`: Returns vector store index statistics.
