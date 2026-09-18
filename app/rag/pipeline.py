import logging
from pathlib import Path
from typing import AsyncIterator
import httpx

from app.core.config import settings
from app.rag.document_loader import load_file, load_documents_from_folder
from app.rag.text_splitter import split_document_into_chunks
from app.rag.embeddings import GeminiEmbedder
from app.rag.vector_store import InMemoryVectorStore, MongoVectorStore

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are an accurate, reliable question-answering assistant.
Your answers must be grounded strictly in the provided Context.

Rules:
1. Base your answer ONLY on the provided Context.
2. If the answer cannot be found in the Context, respond with: "I don't know based on the provided documents."
3. Always cite the document source (e.g. [resume.pdf]) where the information was found.
4. Keep the answer clear, structured, and factual.
"""


class PureRAG:
    """
    Unified end-to-end RAG pipeline.
    """

    def __init__(
        self,
        api_key: str | None = None,
        use_mongo: bool = False
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.embedder = GeminiEmbedder(api_key=self.api_key)
        
        if use_mongo and settings.MONGO_URL:
            logger.info("Using MongoDB Atlas Vector Store.")
            self.store = MongoVectorStore(
                mongo_url=settings.MONGO_URL,
                db_name=settings.DB_NAME,
                collection_name=settings.VECTOR_COLLECTION,
            )
        else:
            logger.info("Using InMemory Vector Store.")
            self.store = InMemoryVectorStore()

    async def ingest_folder(self, folder_path: str | Path) -> int:
        """Loads and indexes all documents from a folder."""
        docs = load_documents_from_folder(folder_path)
        if not docs:
            logger.warning(f"No documents found in {folder_path}")
            return 0

        existing_ids = await self.store.get_existing_ids()
        all_chunks = []
        for doc in docs:
            chunks = split_document_into_chunks(doc)
            new_chunks = [c for c in chunks if c["id"] not in existing_ids]
            all_chunks.extend(new_chunks)

        if not all_chunks:
            logger.info(f"All chunks from {folder_path} already indexed. Skipping.")
            return 0

        logger.info(f"Embedding {len(all_chunks)} chunks from {folder_path}...")
        texts = [c["text"] for c in all_chunks]
        embeddings = await self.embedder.aembed_texts(texts)

        added = await self.store.add(all_chunks, embeddings)
        logger.info(f"Successfully ingested {added} new chunks.")
        return added

    async def ingest_file(self, file_path: str | Path) -> int:
        """Loads and indexes a single document file."""
        path = Path(file_path)
        text = load_file(path)
        if not text.strip():
            logger.warning(f"No readable content found in file {path}")
            return 0

        doc = {"source": path.name, "text": text}
        chunks = split_document_into_chunks(doc)

        existing_ids = await self.store.get_existing_ids()
        new_chunks = [c for c in chunks if c["id"] not in existing_ids]
        if not new_chunks:
            logger.info(f"File {path.name} already indexed.")
            return 0

        texts = [c["text"] for c in new_chunks]
        embeddings = await self.embedder.aembed_texts(texts)
        return await self.store.add(new_chunks, embeddings)

    async def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        """Finds the most relevant document chunks for a query."""
        query_vector = await self.embedder.aembed_query(query)
        return await self.store.search(query_vector, top_k=top_k)

    def _build_context(self, chunks: list[dict]) -> str:
        if not chunks:
            return "(No relevant context found.)"
        formatted = []
        for c in chunks:
            formatted.append(f"[{c['source']}]:\n{c['text']}")
        return "\n\n".join(formatted)

    async def _call_gemini_generate(self, payload: dict) -> str:
        """Calls Gemini generateContent with automatic retry and model fallbacks for 503/429 spikes."""
        import asyncio

        # Fallback list of fast, capable models
        models_to_try = [settings.GENERATION_MODEL, "gemini-3.5-flash", "gemini-3.1-flash-lite"]
        # Remove duplicates while preserving order
        candidate_models = list(dict.fromkeys(models_to_try))

        last_error = None

        async with httpx.AsyncClient(timeout=45.0) as client:
            for model_name in candidate_models:
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model_name}:generateContent?key={self.api_key}"
                )

                # Try up to 2 times for transient errors (503 / 429)
                for attempt in range(2):
                    try:
                        resp = await client.post(url, json=payload)
                        if resp.status_code == 200:
                            data = resp.json()
                            return data["candidates"][0]["content"]["parts"][0]["text"]
                        
                        # If service is temporarily unavailable (503) or rate-limited (429), wait and retry/fallback
                        if resp.status_code in (503, 429):
                            logger.warning(
                                f"Model '{model_name}' returned {resp.status_code} (attempt {attempt + 1}). Retrying/falling back..."
                            )
                            await asyncio.sleep(1.0)
                            continue
                        else:
                            logger.error(f"Gemini generation error on {model_name} ({resp.status_code}): {resp.text[:200]}")
                            break  # Move to next fallback model

                    except Exception as e:
                        logger.warning(f"Error calling {model_name}: {e}")
                        last_error = e
                        await asyncio.sleep(0.5)

        raise RuntimeError(f"All Gemini generation models failed. Last error: {last_error or 'Service Unavailable'}")

    async def answer(self, question: str, top_k: int = 4) -> dict:
        """Retrieves relevant chunks and generates a grounded response using Gemini."""
        chunks = await self.retrieve(question, top_k=top_k)
        context = self._build_context(chunks)

        user_content = f"Context:\n{context}\n\nQuestion: {question}"

        payload = {
            "system_instruction": {"parts": [{"text": RAG_SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": user_content}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1000
            }
        }

        answer_text = await self._call_gemini_generate(payload)

        sources = list({c["source"] for c in chunks})
        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "chunks": chunks
        }

    async def answer_stream(self, question: str, top_k: int = 4) -> AsyncIterator[str]:
        """Streams generated tokens directly from Gemini with fallback models."""
        import json
        import asyncio

        chunks = await self.retrieve(question, top_k=top_k)
        context = self._build_context(chunks)

        user_content = f"Context:\n{context}\n\nQuestion: {question}"
        payload = {
            "system_instruction": {"parts": [{"text": RAG_SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": user_content}]}],
        }

        models_to_try = list(dict.fromkeys([settings.GENERATION_MODEL, "gemini-3.5-flash", "gemini-3.1-flash-lite"]))

        async with httpx.AsyncClient(timeout=60.0) as client:
            for model_name in models_to_try:
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model_name}:streamGenerateContent?alt=sse&key={self.api_key}"
                )
                success = False
                try:
                    async with client.stream("POST", url, json=payload) as response:
                        if response.status_code != 200:
                            logger.warning(f"Stream error on {model_name}: {response.status_code}. Trying fallback...")
                            continue

                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if not data_str:
                                    continue
                                try:
                                    parsed = json.loads(data_str)
                                    text_delta = parsed["candidates"][0]["content"]["parts"][0]["text"]
                                    yield text_delta
                                    success = True
                                except Exception:
                                    continue
                    if success:
                        return
                except Exception as e:
                    logger.warning(f"Streaming exception on {model_name}: {e}")
                    await asyncio.sleep(0.5)


# Global singleton instance
rag_pipeline = PureRAG()

