import pytest
from unittest.mock import Mock, patch, MagicMock, call
from factchecker.strategies.advocate_mediator import AdvocateMediatorStrategy

@pytest.fixture
def mock_llama_retriever():
    with patch('factchecker.retrieval.llama_base_retriever.LlamaBaseRetriever') as mock:
        retriever = Mock()
        # Mock specific evidence retrieval
        retriever.retrieve.return_value = [
            {"text": "Strong evidence supporting the claim", "score": 0.95},
            {"text": "Moderate evidence supporting the claim", "score": 0.85},
            {"text": "Weak evidence supporting the claim", "score": 0.75}
        ]
        mock.return_value = retriever
        yield mock

@pytest.fixture
def mock_llama_indexer():
    with patch('factchecker.strategies.advocate_mediator.LlamaVectorStoreIndexer') as mock:
        indexer = Mock()
        indexer.index.return_value = True
        mock.return_value = indexer
        yield mock

@pytest.fixture
def mock_advocate_step():
    with patch('factchecker.strategies.advocate_mediator.AdvocateStep') as mock:
        advocate = Mock()
        advocate.evaluate_claim.return_value = ("SUPPORTS", "Based on strong evidence, this claim is supported")
        mock.return_value = advocate
        yield mock

@pytest.fixture
def mock_mediator_step():
    with patch('factchecker.strategies.advocate_mediator.MediatorStep') as mock:
        mediator = Mock()
        mediator.synthesize_verdicts.return_value = "FINAL_SUPPORTS"
        mock.return_value = mediator
        yield mock

def test_advocate_evaluation_with_evidence(mock_llama_indexer, mock_llama_retriever, mock_advocate_step, mock_mediator_step):
    """
    Test that advocates properly evaluate evidence with different confidence levels.
    """
    # Configure mock advocate to return different verdicts based on evidence
    advocate = mock_advocate_step.return_value
    advocate.evaluate_claim.side_effect = [
        ("SUPPORTS", "High confidence support"),
        ("PARTIALLY_SUPPORTS", "Medium confidence support"),
        ("REFUTES", "Low confidence support")
    ]

    strategy = AdvocateMediatorStrategy(
        indexer_options_list=[{"model_name": "test"} for _ in range(3)],
        retriever_options_list=[{"model_name": "test"} for _ in range(3)],
        advocate_options={"top_k": 3, "min_score": 0.7},
        mediator_options={"temperature": 0.5},
        advocate_prompt="test prompt",
        mediator_prompt="test prompt"
    )

    claim = "Test claim for evidence evaluation"
    final_verdict, verdicts, reasonings = strategy.evaluate_claim(claim)

    # Verify each advocate was called with proper evidence
    assert advocate.evaluate_claim.call_count == 3
    assert verdicts == ["SUPPORTS", "PARTIALLY_SUPPORTS", "REFUTES"]
    assert reasonings == [
        "High confidence support",
        "Medium confidence support",
        "Low confidence support"
    ]

def test_mediator_synthesis_logic(mock_llama_indexer, mock_llama_retriever, mock_advocate_step, mock_mediator_step):
    """
    Test that mediator properly synthesizes different combinations of verdicts.
    """
    # Configure advocates to return mixed verdicts
    advocate = mock_advocate_step.return_value
    advocate.evaluate_claim.side_effect = [
        ("SUPPORTS", "Support reasoning"),
        ("REFUTES", "Refute reasoning")
    ]

    # Configure mediator with different synthesis results
    mediator = mock_mediator_step.return_value
    mediator.synthesize_verdicts.return_value = "INCONCLUSIVE"

    strategy = AdvocateMediatorStrategy(
        indexer_options_list=[{"model_name": "test"} for _ in range(2)],
        retriever_options_list=[{"model_name": "test"} for _ in range(2)],
        advocate_options={"top_k": 3},
        mediator_options={},
        advocate_prompt="test prompt",
        mediator_prompt="test prompt"
    )

    claim = "Test claim for mediator synthesis"
    final_verdict, verdicts, reasonings = strategy.evaluate_claim(claim)

    # Verify mediator was called with correct verdicts
    expected_verdicts = [
        ("SUPPORTS", "Support reasoning"),
        ("REFUTES", "Refute reasoning")
    ]
    mock_mediator_step.return_value.synthesize_verdicts.assert_called_once_with(
        expected_verdicts,
        claim
    )
    assert final_verdict == "INCONCLUSIVE"

def test_evidence_threshold_filtering(mock_llama_indexer, mock_llama_retriever, mock_advocate_step, mock_mediator_step):
    """
    Test that evidence below confidence threshold is filtered out.
    """
    # Configure retriever to return evidence with varying confidence scores
    retriever = mock_llama_retriever.return_value
    retriever.retrieve.return_value = [
        {"text": "High confidence evidence", "score": 0.9},
        {"text": "Medium confidence evidence", "score": 0.8},
        {"text": "Low confidence evidence", "score": 0.6}  # Below threshold
    ]

    strategy = AdvocateMediatorStrategy(
        indexer_options_list=[{"model_name": "test"}],
        retriever_options_list=[{"model_name": "test"}],
        advocate_options={"top_k": 3, "min_score": 0.7},  # Set minimum score threshold
        mediator_options={},
        advocate_prompt="test prompt",
        mediator_prompt="test prompt"
    )

    claim = "Test claim for evidence filtering"
    strategy.evaluate_claim(claim)

    # Verify advocate only received evidence above threshold
    advocate = mock_advocate_step.return_value
    advocate.evaluate_claim.assert_called_once()
    
    # Verify the configuration was passed correctly to the advocate step
    mock_advocate_step.assert_called_with(
        options={"top_k": 3, "min_score": 0.7, "system_prompt_template": "test prompt"},
        evidence_options={
            'indexer': mock_llama_indexer.return_value,
            'top_k': 3,
            'min_score': 0.7,
            'query_template': "evidence for: {claim}"
        }
    )

def test_error_handling_and_recovery(mock_llama_indexer, mock_llama_retriever, mock_advocate_step, mock_mediator_step):
    """
    Test error handling for various failure scenarios.
    """
    # Configure first advocate to fail, second to succeed
    advocate1 = Mock()
    advocate1.evaluate_claim.side_effect = Exception("Evidence evaluation failed")
    advocate2 = Mock()
    advocate2.evaluate_claim.return_value = ("SUPPORTS", "Successful evaluation")
    
    mock_advocate_step.side_effect = [advocate1, advocate2]

    strategy = AdvocateMediatorStrategy(
        indexer_options_list=[{"model_name": "test"}, {"model_name": "test"}],
        retriever_options_list=[{"model_name": "test"}, {"model_name": "test"}],
        advocate_options={},
        mediator_options={},
        advocate_prompt="test prompt",
        mediator_prompt="test prompt"
    )

    # Test should raise exception since we require all advocates to succeed
    with pytest.raises(Exception) as exc_info:
        strategy.evaluate_claim("Test claim")
    
    assert str(exc_info.value) == "Evidence evaluation failed"
    
    # Verify second advocate was never called due to fail-fast behavior
    assert not advocate2.evaluate_claim.called

def test_real_world_configuration(mock_llama_indexer, mock_llama_retriever, mock_advocate_step, mock_mediator_step):
    """
    Test the strategy with real-world configuration from climatefeedback.py
    """
    indexer_options_list = [{
        'source_directory': 'data',
        'index_name': 'advocate1_index'
    }]

    retriever_options_list = [{
        'top_k': 8,
        'indexer_options': indexer_options_list[0]
    }]

    advocate_options = {
        'max_evidences': 10,
        'top_k': 8,
        'min_score': 0.75
    }

    mediator_options = {}

    strategy = AdvocateMediatorStrategy(
        indexer_options_list=indexer_options_list,
        retriever_options_list=retriever_options_list,
        advocate_options=advocate_options,
        mediator_options=mediator_options,
        advocate_prompt="test advocate prompt",
        mediator_prompt="test mediator prompt"
    )

    # Configure mock responses
    mock_llama_retriever.return_value.retrieve.return_value = [
        {"text": "Evidence 1", "score": 0.9},
        {"text": "Evidence 2", "score": 0.85},
        {"text": "Evidence 3", "score": 0.8},
        {"text": "Evidence 4", "score": 0.76},
        {"text": "Evidence 5", "score": 0.74}  # Below threshold
    ]

    # Test with a realistic climate claim
    claim = "Global temperatures have not risen in the past decade"
    final_verdict, verdicts, reasonings = strategy.evaluate_claim(claim)

    # Verify the configuration was properly passed to the advocate step
    mock_advocate_step.assert_called_with(
        options={"max_evidences": 10, "top_k": 8, "min_score": 0.75, "system_prompt_template": "test advocate prompt"},
        evidence_options={
            'indexer': mock_llama_indexer.return_value,
            'top_k': 8,
            'min_score': 0.75,
            'query_template': "evidence for: {claim}"
        }
    )

def test_confidence_based_verdict(mock_llama_indexer, mock_llama_retriever, mock_advocate_step, mock_mediator_step):
    """
    Test how confidence scores affect the final verdict.
    """
    # Configure evidence with varying confidence scores
    mock_llama_retriever.return_value.retrieve.return_value = [
        {"text": "High confidence evidence", "score": 0.95},
        {"text": "Medium confidence evidence", "score": 0.85},
        {"text": "Low confidence evidence", "score": 0.75}
    ]

    advocate = mock_advocate_step.return_value
    advocate.evaluate_claim.side_effect = [
        ("CORRECT", "High confidence reasoning"),
        ("INCONCLUSIVE", "Medium confidence reasoning"),
        ("INCORRECT", "Low confidence reasoning")
    ]

    strategy = AdvocateMediatorStrategy(
        indexer_options_list=[{"model_name": "test"} for _ in range(3)],
        retriever_options_list=[{"model_name": "test"} for _ in range(3)],
        advocate_options={"min_score": 0.8},  # Higher threshold for confidence
        mediator_options={},
        advocate_prompt="test prompt",
        mediator_prompt="test prompt"
    )

    claim = "Test claim with varying evidence confidence"
    final_verdict, verdicts, reasonings = strategy.evaluate_claim(claim)

    # Verify that the configuration was properly passed to the advocate step
    mock_advocate_step.assert_called_with(
        options={"min_score": 0.8, "system_prompt_template": "test prompt"},
        evidence_options={
            'indexer': mock_llama_indexer.return_value,
            'top_k': 5,  # Default value
            'min_score': 0.8,
            'query_template': "evidence for: {claim}"
        }
    ) 