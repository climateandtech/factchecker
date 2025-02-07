from unittest.mock import MagicMock

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from factchecker.indexing.abstract_indexer import AbstractIndexer
from factchecker.retrieval.abstract_retriever import AbstractRetriever
from factchecker.steps.advocate import AdvocateStep


@pytest.fixture
def mock_llm():
    """Fixture for mocked LLM."""
    mock = MagicMock()
    mock.chat.return_value = MagicMock(message=MagicMock(content="((correct)): Based on the evidence."))
    return mock

@pytest.fixture
def mock_indexer():
    """Fixture for mocked indexer."""
    mock = MagicMock(spec=AbstractIndexer)
    mock.index = MagicMock()
    return mock

@pytest.fixture
def mock_retriever(mock_indexer: MagicMock):
    """Fixture for mocked retriever."""
    mock = MagicMock(spec=AbstractRetriever)
    mock.indexer = mock_indexer
    mock.options = {'top_k': 5}
    mock.retrieve.return_value = [NodeWithScore(node=TextNode(text="Evidence 1", score=0.9), score=0.9)]
    return mock

@pytest.fixture
def mock_evidence():
    """Fixture for mock evidence nodes."""
    def create_node(text: str, score: float) -> list[NodeWithScore]:
        node = NodeWithScore(node=TextNode(text=text), score=score)
        return node
    return [create_node("Evidence 1", 0.9), create_node("Evidence 2", 0.8)]

def test_advocate_initialization(mock_llm: MagicMock, mock_retriever: MagicMock) -> None:
    """Test advocate initialization with different options."""
    evidence_options = {
        'min_score': 0.75
    }

    options = {
        'system_prompt': "Systemprompt",
        'label_options': ['correct', 'incorrect', 'not_enough_information'],
    }
    
    advocate = AdvocateStep(
        retriever=mock_retriever,
        llm=mock_llm, 
        options=options, 
        evidence_options=evidence_options
    )
    assert advocate.evidence_step.min_score == 0.75
    assert advocate.system_prompt == "Systemprompt"
    assert advocate.label_options == ['correct', 'incorrect', 'not_enough_information']

def test_default_options(mock_llm: MagicMock, mock_retriever: MagicMock) -> None:
    """Test default options when none provided."""
    evidence_options = {}
    options = {}
    advocate = AdvocateStep(
        retriever=mock_retriever,
        llm=mock_llm, 
        options=options, 
        evidence_options=evidence_options
    )
    assert advocate.system_prompt is not None
    assert advocate.label_options is not None
    assert advocate.max_retries is not None
    assert advocate.evidence_step is not None

def test_evaluate_claim(mock_llm: MagicMock, mock_retriever: MagicMock) -> None:
    """Test evidence evaluation process."""
    evidence_options = {}
    options = {}
    advocate = AdvocateStep(
        retriever=mock_retriever,
        llm=mock_llm, 
        options=options, 
        evidence_options=evidence_options
    )
    verdict, reasoning = advocate.evaluate_claim("Test claim")
    
    assert mock_llm.chat.called
    assert verdict == "CORRECT"
    assert isinstance(reasoning, str)

def test_llm_error_handling(mock_llm: MagicMock, mock_retriever: MagicMock) -> None:
    """Test handling of LLM errors."""
    mock_llm.chat.return_value = MagicMock(message=MagicMock(content="Invalid response"))
    advocate = AdvocateStep(
        retriever=mock_retriever,
        llm=mock_llm,
    )
    verdict, reasoning = advocate.evaluate_claim("Test claim")
    
    assert verdict == "ERROR_PARSING_RESPONSE"
    assert reasoning == "No reasoning available"

def test_retry_mechanism(mock_llm: MagicMock, mock_retriever: MagicMock) -> None:
    """Test retry mechanism for invalid responses."""
    responses = [
        MagicMock(message=MagicMock(content="Invalid 1")),
        MagicMock(message=MagicMock(content="Invalid 2")),
        MagicMock(message=MagicMock(content="((correct)): Valid response"))
    ]
    mock_llm.chat.side_effect = responses

    advocate = AdvocateStep(
        retriever=mock_retriever,
        llm=mock_llm,
    )
    
    verdict, reasoning = advocate.evaluate_claim("Test claim")
    assert verdict == "CORRECT"
    assert mock_llm.chat.call_count == 3 