import pytest
from unittest.mock import Mock, MagicMock, patch
from factchecker.steps.mediator import MediatorStep
from llama_index.core.llms import ChatMessage

@pytest.fixture
def mock_llm():
    """Fixture for mocked LLM"""
    mock = MagicMock()
    mock.chat.return_value = MagicMock(message=MagicMock(content="((correct)): Final verdict based on all evidence."))
    return mock

def test_mediator_initialization(mock_llm):
    """Test mediator initialization with different options"""
    options = {
        'arbitrator_primer': "You are a mediator synthesizing verdicts."
    }
    
    mediator = MediatorStep(llm=mock_llm, options=options.copy())
    assert mediator.prompt == "You are a mediator synthesizing verdicts."

def test_default_options(mock_llm):
    """Test default options when none provided"""
    mediator = MediatorStep(llm=mock_llm)
    assert mediator.prompt == ""

def test_synthesize_verdicts(mock_llm):
    """Test verdict synthesis process"""
    mediator = MediatorStep(llm=mock_llm)
    verdicts_and_reasonings = [
        ("CORRECT", "First evaluation"),
        ("INCORRECT", "Second evaluation")
    ]
    result = mediator.synthesize_verdicts(verdicts_and_reasonings, "Test claim")
    
    assert mock_llm.chat.called
    assert result == "CORRECT"

def test_llm_error_handling(mock_llm):
    """Test handling of LLM errors"""
    mock_llm.chat.return_value = MagicMock(message=MagicMock(content="Invalid response"))
    mediator = MediatorStep(llm=mock_llm)
    result = mediator.synthesize_verdicts([("CORRECT", "Test")], "Test claim")
    
    assert result == "ERROR_PARSING_RESPONSE"

def test_retry_mechanism(mock_llm):
    """Test retry mechanism for invalid responses"""
    responses = [
        MagicMock(message=MagicMock(content="Invalid 1")),
        MagicMock(message=MagicMock(content="Invalid 2")),
        MagicMock(message=MagicMock(content="((correct)): Valid response"))
    ]
    mock_llm.chat.side_effect = responses
    mediator = MediatorStep(llm=mock_llm)
    
    result = mediator.synthesize_verdicts([("CORRECT", "Test")], "Test claim")
    assert result == "CORRECT"
    assert mock_llm.chat.call_count == 3

def test_max_retries_exceeded(mock_llm):
    """Test behavior when max retries are exceeded"""
    mock_llm.chat.return_value = MagicMock(message=MagicMock(content="Invalid format"))
    mediator = MediatorStep(llm=mock_llm)
    
    result = mediator.synthesize_verdicts([("CORRECT", "Test")], "Test claim")
    assert result == "ERROR_PARSING_RESPONSE"
    assert mock_llm.chat.call_count == mediator.max_retries

def test_empty_verdicts(mock_llm):
    """Test handling of empty verdicts list"""
    mediator = MediatorStep(llm=mock_llm)
    result = mediator.synthesize_verdicts([], "Test claim")
    
    # The actual implementation returns the LLM response even for empty verdicts
    assert result == "CORRECT" 