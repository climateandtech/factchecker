import os
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama

def load_llm(
    llm_type=None,
    model=None,
    temperature=0.0,
    request_timeout=None,
    api_key=None,
    organization=None,
    api_base=None,
    context_window=3900,
    embedding_model=None,
    **kwargs
):
    llm_type = llm_type or os.getenv("LLM_TYPE", "openai").lower()
    
    if llm_type == "ollama":
        model = model or os.getenv("OLLAMA_MODEL", "llama2")
        request_timeout = request_timeout or float(os.getenv("OLLAMA_REQUEST_TIMEOUT", 120.0))
        llm = Ollama(base_url=os.getenv("OLLAMA_API_BASE_URL"), model=model, request_timeout=request_timeout, temperature=temperature, context_window=context_window)
    else:  # default to openai
        model = model or os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo-1106")
        temperature = temperature or float(os.getenv("TEMPERATURE", 0.1))
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        organization = organization or os.getenv("OPENAI_ORGANIZATION")
        api_base = api_base or os.getenv("OPENAI_API_BASE")
        
        # Filter out retriever-specific options
        openai_kwargs = {k: v for k, v in kwargs.items() if k not in ['top_k', 'similarity_top_k']}
        
        llm = OpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
            context_window=context_window,
            **openai_kwargs
        )
    
    return llm





