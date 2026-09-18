import logging
from pymongo import MongoClient
from app.core.config import settings

logger = logging.getLogger("uvicorn")

class Database:
    client: MongoClient | None = None

db_instance = Database()

def get_mongo_client() -> MongoClient | None:
    if not settings.MONGO_URL:
        return None
    if db_instance.client is None:
        try:
            db_instance.client = MongoClient(settings.MONGO_URL)
            db_instance.client.admin.command("ping")
            logger.info("Connected to MongoDB successfully.")
        except Exception as e:
            logger.warning(f"Failed to connect to MongoDB: {e}")
            db_instance.client = None
    return db_instance.client

def get_vector_collection():
    client = get_mongo_client()
    if client is None:
        return None
    return client[settings.DB_NAME][settings.VECTOR_COLLECTION]
