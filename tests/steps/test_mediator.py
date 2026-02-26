import pytest
from unittest.mock import Mock, MagicMock, patch
from llama_index.core.llms import ChatMessage

from factchecker.steps.mediator import MediatorStep, MEDIATOR_VERDICT_FORMAT

@pytest.fixture
def mock_llm():
    """Fixture for mocked LLM"""
    mock = MagicMock()
    mock.chat.return_value = MagicMock(message=MagicMock(content="((correct)): Final verdict based on all evidence."))
    return mock

def test_mediator_initialization(mock_llm: MagicMock) -> None:
    """Test mediator initialization with different options."""
    options = {
        'system_prompt': "You are a mediator synthesizing verdicts."
    }
    
    mediator = MediatorStep(llm=mock_llm, options=options.copy())
    assert mediator.system_prompt == "You are a mediator synthesizing verdicts."

def test_default_options(mock_llm):
    """Test default options when none provided"""
    mediator = MediatorStep(llm=mock_llm)
    assert mediator.system_prompt == ""

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


def test_mediator_retry_includes_format_feedback(mock_llm):
    """On retry after wrong format, second chat call receives messages with last answer and format instruction."""
    wrong = '{ "verdict": ("correct", "medium") }'
    right = "((correct))"
    mock_llm.chat.side_effect = [
        MagicMock(message=MagicMock(content=wrong)),
        MagicMock(message=MagicMock(content=right)),
    ]
    mediator = MediatorStep(llm=mock_llm)
    result = mediator.synthesize_verdicts([("CORRECT", "r1")], "Claim")
    assert result == "CORRECT"
    assert mock_llm.chat.call_count == 2
    # Second call: messages include assistant (wrong) + user retry with format feedback
    messages_second = mock_llm.chat.call_args[0][0]
    assert len(messages_second) >= 4
    retry_user = next(m for m in messages_second if m.role == "user" and "wrong format" in m.content)
    assert "Your previous response was in the wrong format" in retry_user.content
    assert "This was your last answer" in retry_user.content
    assert "wrong" in retry_user.content.lower()
    assert MEDIATOR_VERDICT_FORMAT in retry_user.content or "((correct))" in retry_user.content


def test_mediator_custom_verdict_parser(mock_llm):
    """When verdict_parser is provided, it is used and can parse alternative formats."""
    def parse_json_style(content):
        if '"verdict":' in content and "correct" in content.lower():
            return "CORRECT"
        if '"verdict":' in content and "incorrect" in content.lower():
            return "INCORRECT"
        return None
    mock_llm.chat.return_value = MagicMock(
        message=MagicMock(content='{ "verdict": ("correct", "medium") }\nSome reasoning.')
    )
    mediator = MediatorStep(llm=mock_llm, options={"verdict_parser": parse_json_style})
    result = mediator.synthesize_verdicts([("CORRECT", "r1")], "Claim")
    assert result == "CORRECT"
    assert mock_llm.chat.call_count == 1


def test_mediator_max_retries_from_options(mock_llm):
    """max_retries can be overridden via options."""
    mediator = MediatorStep(llm=mock_llm, options={"max_retries": 5})
    assert mediator.max_retries == 5 