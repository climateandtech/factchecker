import pytest
from unittest.mock import Mock, MagicMock, patch
from factchecker.steps.evaluate import EvaluateStep
from llama_index.core.llms import ChatMessage

@pytest.fixture
def mock_llm():
    """Fixture for mocked LLM"""
    mock = MagicMock()
    mock.chat.return_value = MagicMock(message=MagicMock(content='{"label": "correct"}'))
    return mock

def test_evaluate_initialization(mock_llm):
    """Test evaluate initialization with different options"""
    options = {
        'pro_prompt_template': "Pro evidence: {evidence}",
        'con_prompt_template': "Con evidence: {evidence}",
        'system_prompt_template': "System: {claim}",
        'format_prompt': "Answer with TRUE or FALSE"
    }
    
    evaluator = EvaluateStep(llm=mock_llm, options=options.copy())
    assert evaluator.pro_prompt_template == "Pro evidence: {evidence}"
    assert evaluator.con_prompt_template == "Con evidence: {evidence}"
    assert evaluator.system_prompt_template == "System: {claim}"
    assert evaluator.format_prompt == "Answer with TRUE or FALSE"

def test_default_options(mock_llm):
    """Test default options when none provided"""
    evaluator = EvaluateStep(llm=mock_llm)
    assert evaluator.pro_prompt_template is not None
    assert evaluator.con_prompt_template is not None
    assert evaluator.system_prompt_template is not None
    assert evaluator.format_prompt is not None

def test_evaluate_claim(mock_llm):
    """Test evidence evaluation process"""
    evaluator = EvaluateStep(llm=mock_llm)
    result = evaluator.evaluate_claim("Test claim", "Pro evidence", "Con evidence")
    
    assert mock_llm.chat.called
    assert result == "CORRECT"

def test_llm_error_handling(mock_llm):
    """Test handling of LLM errors"""
    mock_llm.chat.return_value = MagicMock(message=MagicMock(content="Invalid JSON"))
    evaluator = EvaluateStep(llm=mock_llm)
    result = evaluator.evaluate_claim("Test claim", "Pro evidence", "Con evidence")
    
    assert result == "ERROR_PARSING_RESPONSE"

def test_json_parsing(mock_llm):
    """Test parsing of JSON responses"""
    responses = [
        '{"label": "correct"}',
        '{"label": "CORRECT"}',
        '{"label": "Correct"}'
    ]
    
    evaluator = EvaluateStep(llm=mock_llm)
    for response in responses:
        mock_llm.chat.return_value = MagicMock(message=MagicMock(content=response))
        result = evaluator.evaluate_claim("Test claim", "Pro evidence", "Con evidence")
        assert result == "CORRECT"

def test_missing_label(mock_llm):
    """Test handling of JSON response without label"""
    mock_llm.chat.return_value = MagicMock(message=MagicMock(content='{"other": "value"}'))
    evaluator = EvaluateStep(llm=mock_llm)
    result = evaluator.evaluate_claim("Test claim", "Pro evidence", "Con evidence")
    
    assert result == "ERROR_PARSING_RESPONSE"

def test_message_generation(mock_llm):
    """Test message generation for LLM"""
    evaluator = EvaluateStep(llm=mock_llm)
    claim = "Test claim"
    pro_evidence = "Pro evidence"
    con_evidence = "Con evidence"
    
    messages = [
        ChatMessage(role="system", content=evaluator.system_prompt_template.format(claim=claim)),
        ChatMessage(role="user", content=evaluator.pro_prompt_template.format(evidence=pro_evidence)),
        ChatMessage(role="user", content=evaluator.con_prompt_template.format(evidence=con_evidence)),
        ChatMessage(role="user", content=evaluator.format_prompt)
    ]
    
    assert len(messages) == 4  # System + pro + con + format prompts
    assert all(isinstance(m, ChatMessage) for m in messages)
    assert all(m.role in ["system", "user"] for m in messages) 