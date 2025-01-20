import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

def load_embedding_model(
    embedding_type=None,
    model_name=None,
    api_key=None,
    api_base=None,
    **kwargs
):
    """
    Load and configure an embedding model based on specified parameters.

    This function initializes an embedding model instance (OpenAI, HuggingFace, or Ollama)
    with configuration from provided parameters or environment variables. It supports flexible
    configuration through both direct parameters and environment variables.

    Args:
        embedding_type (str, optional): Type of embedding model to use ('openai', 'huggingface', or 'ollama').
            Defaults to env var EMBEDDING_TYPE or 'openai'.
        model_name (str, optional): Name of the model to use. Defaults vary by embedding type:
            - OpenAI: env var OPENAI_EMBEDDING_MODEL or 'text-embedding-ada-002'
            - HuggingFace: env var HUGGINGFACE_EMBEDDING_MODEL or 'BAAI/bge-small-en-v1.5'
            - Ollama: env var OLLAMA_MODEL or 'nomic-embed-text'
        api_key (str, optional): API key for OpenAI. Defaults to env var OPENAI_API_KEY.
        api_base (str, optional): Base API URL. Defaults vary by embedding type:
            - OpenAI: env var OPENAI_API_BASE
            - Ollama: env var OLLAMA_API_BASE_URL or 'http://localhost:11434'
        **kwargs: Additional keyword arguments passed to the embedding model constructor.

    Returns:
        Union[OpenAIEmbedding, HuggingFaceEmbedding, OllamaEmbedding]: Configured embedding model instance.

    Raises:
        ValueError: If OpenAI API key is missing when using OpenAI embeddings
        ValueError: If unsupported embedding type is specified

    Environment Variables:
        EMBEDDING_TYPE: Type of embedding model to use
        OPENAI_EMBEDDING_MODEL: Model name for OpenAI embeddings
        OPENAI_API_KEY: API key for OpenAI
        OPENAI_API_BASE: Base URL for OpenAI API
        HUGGINGFACE_EMBEDDING_MODEL: Model name for HuggingFace embeddings
        OLLAMA_MODEL: Model name for Ollama embeddings
        OLLAMA_API_BASE_URL: Base URL for Ollama API
    """
    embedding_type = embedding_type or os.getenv("EMBEDDING_TYPE", "openai").lower()
    
    if embedding_type == "openai":
        model_name = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        api_base = api_base or os.getenv("OPENAI_API_BASE")
        
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        return OpenAIEmbedding(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            **kwargs
        )
        
    elif embedding_type == "huggingface":
        model_name = model_name or os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        
        return HuggingFaceEmbedding(
            model_name=model_name,
            **kwargs
        )
        
    elif embedding_type == "ollama":
        model_name = model_name or os.getenv("OLLAMA_MODEL", "nomic-embed-text")
        api_base = api_base or os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
        
        return OllamaEmbedding(
            model_name=model_name,
            base_url=api_base,
            **kwargs
        )
        
    elif embedding_type == "mock":
        from tests.conftest import MockEmbedding
        mock_dim = int(os.getenv("MOCK_EMBED_DIM", "384"))
        return MockEmbedding(dim=mock_dim)
        
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")