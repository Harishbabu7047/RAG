import asyncio
import logging
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiEmbedder:
    """
    Direct asynchronous and synchronous embeddings via Google's Gemini REST API.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str | None = None, model: str | None = None, dimensions: int = 768):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.EMBEDDING_MODEL
        self.dimensions = dimensions

    async def _embed_single(self, client: httpx.AsyncClient, text: str) -> list[float]:
        url = f"{self.BASE_URL}/models/{self.model}:embedContent?key={self.api_key}"
        payload = {
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.dimensions
        }
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            logger.error(f"Gemini embedding error ({resp.status_code}): {resp.text}")
            resp.raise_for_status()
        data = resp.json()
        return data["embedding"]["values"]

    async def aembed_texts(self, texts: list[str], concurrency: int = 5) -> list[list[float]]:
        """Asynchronously embeds multiple chunks concurrently."""
        if not texts:
            return []
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        semaphore = asyncio.Semaphore(concurrency)

        async with httpx.AsyncClient(timeout=30.0) as client:
            async def bounded_embed(t: str):
                async with semaphore:
                    return await self._embed_single(client, t)

            tasks = [bounded_embed(t) for t in texts]
            return await asyncio.gather(*tasks)

    async def aembed_query(self, query: str) -> list[float]:
        """Asynchronously embeds a single user question or search query."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        async with httpx.AsyncClient(timeout=20.0) as client:
            return await self._embed_single(client, query)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Synchronous version of embed_texts."""
        if not texts:
            return []
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        results = []
        url = f"{self.BASE_URL}/models/{self.model}:embedContent?key={self.api_key}"
        with httpx.Client(timeout=30.0) as client:
            for t in texts:
                payload = {
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": self.dimensions
                }
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                results.append(resp.json()["embedding"]["values"])
        return results

    def embed_query(self, query: str) -> list[float]:
        """Synchronous version of embed_query."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        url = f"{self.BASE_URL}/models/{self.model}:embedContent?key={self.api_key}"
        payload = {
            "content": {"parts": [{"text": query}]},
            "outputDimensionality": self.dimensions
        }
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["embedding"]["values"]
