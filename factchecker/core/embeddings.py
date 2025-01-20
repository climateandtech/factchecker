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
    """Load embedding model based on configuration.
    Similar to load_llm() but for embeddings.
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
        
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}") 