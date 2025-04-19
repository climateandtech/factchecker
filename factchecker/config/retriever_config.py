from enum import Enum

from pydantic import BaseModel, Field, field_validator


class RetrieverType(str, Enum):
    """Enum class for different retriever types."""

    BASE = "LlamaBaseRetriever"
    COLBERT = "RagatouilleColBERTRetriever"


class RetrieverConfig(BaseModel):
    """Configuration schema for a Retriever."""

    type: RetrieverType = Field(..., description="Type of retriever (must be one of RetrieverType enum)")
    index_name: str = Field(..., description="Name of the index this retriever queries")
    top_k: int = Field(5, description="Number of results to return")
    min_score: float = Field(0.75, description="Minimum similarity score for valid results")

    @field_validator("min_score")
    def check_min_score(cls, v: float) -> float:
        """Ensure that min_score is between 0 and 1."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("min_score must be between 0 and 1")
        return v
