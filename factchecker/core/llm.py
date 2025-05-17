"""
Module for loading and configuring Language Learning Models (LLMs) for the fact-checker.

This module provides functionality to initialize either OpenAI or Ollama LLMs with
appropriate configuration settings from environment variables or direct parameters.
"""

import os
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from typing import Union, List, Optional, Sequence, Dict, Any

class CustomOllama(Ollama):
    """Custom Ollama wrapper that properly handles Ollama's response format."""
    
    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        """Override chat to properly handle Ollama's response format."""
        response = super().chat(messages, **kwargs)
        
        # If it's already a ChatResponse, return it
        if isinstance(response, ChatResponse):
            return response
            
        # If it's a dict (raw Ollama response), convert it
        if isinstance(response, dict):
            # Extract content from response
            content = response.get("message", {}).get("content", "")
            
            # Create a basic usage dict that matches llama-index's expectations
            usage = {
                "prompt_tokens": 0,  # Ollama doesn't provide token counts
                "completion_tokens": 0,
                "total_tokens": 0
            }
            
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=content
                ),
                raw=response,  # Store the original response
                usage=usage  # Add the usage field
            )
            
        # If it's something else, try to handle it gracefully
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=str(response)
            ),
            raw={"content": str(response)},  # Store string response
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )

def load_llm(
    llm_type=None,
    model=None,
    temperature=None,
    request_timeout=None,
    api_key=None,
    organization=None,
    api_base=None,
    context_window=100000,
    embedding_model=None,
    **kwargs
    ) -> Union[OpenAI, CustomOllama]:
    """
    Load and configure a Language Learning Model (LLM) based on specified parameters.

    This function initializes either an OpenAI or Ollama LLM instance with configuration
    from provided parameters or environment variables. It supports flexible configuration
    through both direct parameters and environment variables.

    Args:
        llm_type (str, optional): Type of LLM to use ('openai' or 'ollama'). 
            Defaults to env var LLM_TYPE or 'openai'.
        model (str, optional): Model name to use. Defaults to env var based on LLM type.
        temperature (float, optional): Sampling temperature. 
            Defaults to env var TEMPERATURE or 0.1.
        request_timeout (float, optional): Request timeout in seconds for Ollama.
            Defaults to env var OLLAMA_REQUEST_TIMEOUT or 120.0.
        api_key (str, optional): API key for OpenAI. 
            Defaults to env var OPENAI_API_KEY.
        organization (str, optional): Organization ID for OpenAI.
            Defaults to env var OPENAI_ORGANIZATION.
        api_base (str, optional): Base API URL.
            Defaults to env var OPENAI_API_BASE or OLLAMA_API_BASE_URL.
        context_window (int, optional): Size of context window. Defaults to 100000.
        embedding_model (str, optional): Name of embedding model (unused currently).
        **kwargs: Additional keyword arguments passed to the LLM constructor.

    Returns:
        Union[OpenAI, CustomOllama]: Configured LLM instance ready for use.

    Environment Variables:
        LLM_TYPE: Type of LLM to use ('openai' or 'ollama')
        TEMPERATURE: Model temperature setting
        OLLAMA_MODEL: Model name for Ollama
        OLLAMA_REQUEST_TIMEOUT: Request timeout for Ollama
        OLLAMA_API_BASE_URL: Base URL for Ollama API
        OPENAI_API_MODEL: Model name for OpenAI
        OPENAI_API_KEY: API key for OpenAI
        OPENAI_ORGANIZATION: Organization ID for OpenAI
        OPENAI_API_BASE: Base URL for OpenAI API
    """
    llm_type = llm_type or os.getenv("LLM_TYPE", "openai").lower()
    
    if llm_type == "ollama":
        model = model or os.getenv("OLLAMA_MODEL", "llama2")
        request_timeout = request_timeout or float(os.getenv("OLLAMA_REQUEST_TIMEOUT", 120.0))
        if temperature is None:
            temperature = float(os.getenv("TEMPERATURE", 0.1))
        llm = CustomOllama(
            base_url=os.getenv("OLLAMA_API_BASE_URL"),
            model=model,
            request_timeout=request_timeout,
            temperature=temperature,
            context_window=context_window
        )
    else:  # default to openai
        model = model or os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo-1106")
        if temperature is None:
            temperature = float(os.getenv("TEMPERATURE", 0.1))
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        organization = organization or os.getenv("OPENAI_ORGANIZATION")
        api_base = api_base or os.getenv("OPENAI_API_BASE")
        
        # Filter out retriever-specific options
        openai_kwargs = {k: v for k, v in kwargs.items() if k not in ['top_k', 'similarity_top_k']}
        
        llm = OpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            organization=organization,
            api_base=api_base,
            context_window=context_window,
            **openai_kwargs
        )
    
    return llm
