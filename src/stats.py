import json
from pydantic import BaseModel, Field
from typing import List

class DocumentStats(BaseModel):
    """
    A Pydantic model to store statistics for a single processed document.
    """
    file_name: str
    file_size: int
    chunk_count: int
    processing_time: float
    avg_chunk_size: float

class ChunkingStats(BaseModel):
    """
    A Pydantic model to aggregate statistics for a full chunking strategy run.
    """
    total_documents: int
    total_chunks: int
    processing_time: float
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    chunks_per_second: float = Field(..., alias='chunks_per_second')
    strategy_name: str
    embedding_model: str
    chunk_size_config: int
    chunk_overlap_config: int
    document_stats: List[DocumentStats] = []

    def to_json(self) -> str:
        """Serializes the model to a JSON string."""
        return self.model_dump_json(indent=4)  # New Pydantic v2 syntax

    class Config:
        """Pydantic configuration for this model."""
        # This allows the use of aliases in the model
        allow_population_by_field_name = True