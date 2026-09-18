from pydantic import BaseModel, Field


class IngestFolderRequest(BaseModel):
    folder_path: str = Field(..., description="Absolute or relative path to folder containing documents.")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer using retrieved documents.")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of top chunks to retrieve.")


class ChunkItem(BaseModel):
    id: str
    text: str
    source: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks: list[ChunkItem]


class IngestResponse(BaseModel):
    status: str
    chunks_ingested: int
    message: str
