import pytest
from unittest.mock import patch, MagicMock
from factchecker.core.llm import load_llm

def test_load_llm_ollama():
    with patch.dict('os.environ', {'LLM_TYPE': 'ollama', 'OLLAMA_MODEL': 'llama2', 'OLLAMA_API_BASE_URL': 'http://localhost:11434'}):
        llm = load_llm()
        assert llm is not None
        assert str(type(llm).__name__) == 'Ollama'

def test_load_llm_openai():
    with patch.dict('os.environ', {'LLM_TYPE': 'openai', 'OPENAI_API_KEY': 'test-key', 'OPENAI_API_MODEL': 'gpt-3.5-turbo'}):
        llm = load_llm()
        assert llm is not None
        assert str(type(llm).__name__) == 'OpenAI'

def test_load_llm_with_custom_options():
    with patch.dict('os.environ', {'LLM_TYPE': 'ollama', 'OLLAMA_MODEL': 'llama2'}):
        llm = load_llm(temperature=0.5, context_window=4000)
        assert llm is not None
        assert llm.temperature == 0.5
        assert llm.context_window == 4000 