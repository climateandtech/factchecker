import pytest
from unittest.mock import Mock, MagicMock, patch
from factchecker.steps.advocate import AdvocateStep
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.llms import ChatMessage

@pytest.fixture
def mock_llm():
    """Fixture for mocked LLM"""
    mock = MagicMock()
    mock.chat.return_value = MagicMock(message=MagicMock(content="((correct)): Based on the evidence."))
    return mock

@pytest.fixture
def mock_indexer():
    """Fixture for mocked indexer"""
    mock = MagicMock()
    mock.index = MagicMock()
    return mock

@pytest.fixture
def mock_evidence():
    """Fixture for mock evidence nodes"""
    def create_node(text, score):
        node = NodeWithScore(node=TextNode(text=text), score=score)
        return node
    return [create_node("Evidence 1", 0.9), create_node("Evidence 2", 0.8)]

def test_advocate_initialization(mock_llm, mock_indexer):
    """Test advocate initialization with different options"""
    evidence_options = {
        'indexer': mock_indexer,
        'top_k': 5,
        'min_score': 0.75
    }
    options = {
        'evidence_prompt_template': "Consider this evidence: {evidence}",
        'system_prompt_template': "System: {claim}",
        'format_prompt': "Answer with TRUE or FALSE"
    }
    
    advocate = AdvocateStep(llm=mock_llm, options=options.copy(), evidence_options=evidence_options)
    assert advocate.evidence_prompt_template == "Consider this evidence: {evidence}"
    assert advocate.system_prompt_template == "System: {claim}"
    assert advocate.format_prompt == "Answer with TRUE or FALSE"

def test_default_options(mock_llm, mock_indexer):
    """Test default options when none provided"""
    evidence_options = {'indexer': mock_indexer}
    advocate = AdvocateStep(llm=mock_llm, evidence_options=evidence_options)
    assert advocate.evidence_prompt_template is not None
    assert advocate.system_prompt_template is not None
    assert advocate.format_prompt is not None

def test_evaluate_evidence(mock_llm, mock_indexer):
    """Test evidence evaluation process"""
    evidence_options = {'indexer': mock_indexer}
    advocate = AdvocateStep(llm=mock_llm, evidence_options=evidence_options)
    verdict, reasoning = advocate.evaluate_evidence("Test claim")
    
    assert mock_llm.chat.called
    assert verdict == "CORRECT"
    assert isinstance(reasoning, str)

def test_llm_error_handling(mock_llm, mock_indexer):
    """Test handling of LLM errors"""
    mock_llm.chat.return_value = MagicMock(message=MagicMock(content="Invalid response"))
    evidence_options = {'indexer': mock_indexer}
    advocate = AdvocateStep(llm=mock_llm, evidence_options=evidence_options)
    verdict, reasoning = advocate.evaluate_evidence("Test claim")
    
    assert verdict == "ERROR_PARSING_RESPONSE"
    assert reasoning == "No reasoning available"

def test_retry_mechanism(mock_llm, mock_indexer):
    """Test retry mechanism for invalid responses"""
    responses = [
        MagicMock(message=MagicMock(content="Invalid 1")),
        MagicMock(message=MagicMock(content="Invalid 2")),
        MagicMock(message=MagicMock(content="((correct)): Valid response"))
    ]
    mock_llm.chat.side_effect = responses
    evidence_options = {'indexer': mock_indexer}
    advocate = AdvocateStep(llm=mock_llm, evidence_options=evidence_options)
    
    verdict, reasoning = advocate.evaluate_evidence("Test claim")
    assert verdict == "CORRECT"
    assert mock_llm.chat.call_count == 3 