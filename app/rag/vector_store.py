import math
import logging
from typing import Protocol

logger = logging.getLogger(__name__)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class BaseVectorStore(Protocol):
    async def add(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        ...
    async def search(self, query_vector: list[float], top_k: int = 4) -> list[dict]:
        ...
    async def get_existing_ids(self) -> set[str]:
        ...


class InMemoryVectorStore:
    """
    Self-contained in-memory vector store with cosine similarity.
    Requires no external database, perfect for local development or fast deployment.
    """

    def __init__(self):
        self._entries: dict[str, dict] = {}

    async def get_existing_ids(self) -> set[str]:
        return set(self._entries.keys())

    async def add(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        added = 0
        for chunk, emb in zip(chunks, embeddings):
            chunk_id = chunk["id"]
            self._entries[chunk_id] = {
                "id": chunk_id,
                "text": chunk["text"],
                "source": chunk.get("source", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
                "embedding": emb,
            }
            added += 1
        return added

    async def search(self, query_vector: list[float], top_k: int = 4) -> list[dict]:
        if not self._entries:
            return []

        scored = []
        for entry in self._entries.values():
            sim = cosine_similarity(query_vector, entry["embedding"])
            scored.append({
                "id": entry["id"],
                "text": entry["text"],
                "source": entry["source"],
                "chunk_index": entry["chunk_index"],
                "score": sim,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._entries)


class MongoVectorStore:
    """
    Direct MongoDB Atlas Vector Store using PyMongo.
    """

    def __init__(self, mongo_url: str, db_name: str, collection_name: str = "vector_documents", index_name: str = "vector_index"):
        from pymongo import MongoClient
        self.client = MongoClient(mongo_url)
        self.collection = self.client[db_name][collection_name]
        self.index_name = index_name

    async def get_existing_ids(self) -> set[str]:
        try:
            cursor = self.collection.find({}, {"_id": 1})
            return {str(doc["_id"]) for doc in cursor}
        except Exception as e:
            logger.warning(f"Failed to fetch MongoDB ids: {e}")
            return set()

    async def add(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        from pymongo import UpdateOne
        if not chunks:
            return 0

        operations = []
        for chunk, emb in zip(chunks, embeddings):
            chunk_id = chunk["id"]
            doc = {
                "_id": chunk_id,
                "id": chunk_id,
                "text": chunk["text"],
                "source": chunk.get("source", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
                "embedding": emb,
            }
            operations.append(UpdateOne({"_id": chunk_id}, {"$set": doc}, upsert=True))

        result = self.collection.bulk_write(operations)
        return result.upserted_count + result.modified_count

    async def search(self, query_vector: list[float], top_k: int = 4) -> list[dict]:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": max(top_k * 10, 50),
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "source": 1,
                    "chunk_index": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            }
        ]
        try:
            results = list(self.collection.aggregate(pipeline))
            return [
                {
                    "id": str(r.get("_id")),
                    "text": r.get("text", ""),
                    "source": r.get("source", "unknown"),
                    "chunk_index": r.get("chunk_index", 0),
                    "score": r.get("score", 0.0),
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"MongoDB vector search error: {e}")
            return []
