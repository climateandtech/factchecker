
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class IndexerType(str, Enum):
    """Enum class for different indexer types."""

    VECTOR_STORE = "LlamaVectorStoreIndexer"
    COLBERT = "ColBERTIndexer"


class IndexerConfig(BaseModel):
    """Configuration schema for an Indexer."""

    type: IndexerType = Field(..., description="Type of indexer (must be one of IndexerType enum)")
    index_name: str = Field(..., description="Unique identifier for the index")
    source_directory: str = Field(..., description="Directory where documents are stored")

