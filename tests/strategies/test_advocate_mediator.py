from unittest.mock import Mock, ANY

import pytest

from factchecker.strategies.advocate_mediator import AdvocateMediatorStrategy


def test_advocate_evaluation_with_evidence(
        advocate_mediator_strategy: AdvocateMediatorStrategy, 
        mock_advocate_step: Mock
    ) -> None:
    """Test that advocates properly evaluate evidence with different confidence levels."""
    # Configure the dummy advocate (returned by the patched AdvocateStep) to return different values.
    dummy_adv = mock_advocate_step.return_value
    dummy_adv.evaluate_claim.side_effect = [
        ("SUPPORTS", "High confidence support"),
        ("PARTIALLY_SUPPORTS", "Medium confidence support"),
        ("REFUTES", "Low confidence support")
    ]

    claim = "Test claim for evidence evaluation"
    final_verdict, verdicts, reasonings = advocate_mediator_strategy.evaluate_claim(claim)

    # Verify that evaluate_claim was called three times (one per advocate).
    assert dummy_adv.evaluate_claim.call_count == 3
    assert verdicts == ["SUPPORTS", "PARTIALLY_SUPPORTS", "REFUTES"]
    assert reasonings == [
        "High confidence support",
        "Medium confidence support",
        "Low confidence support"
    ]


def test_mediator_synthesis_logic(
    advocate_mediator_strategy_factory: AdvocateMediatorStrategy,
    mock_advocate_step: Mock,
    mock_mediator_step: Mock,
) -> None:
    """Test that mediator properly synthesizes different combinations of verdicts."""
    dummy_adv = mock_advocate_step.return_value
    dummy_adv.evaluate_claim.side_effect = [
        ("SUPPORTS", "Support reasoning"),
        ("REFUTES", "Refute reasoning")
    ]

    dummy_med = mock_mediator_step.return_value
    dummy_med.synthesize_verdicts.return_value = "INCONCLUSIVE"

    strategy = advocate_mediator_strategy_factory(2)

    claim = "Test claim for mediator synthesis"
    final_verdict, verdicts, reasonings = strategy.evaluate_claim(claim)

    expected_verdicts = [
        ("SUPPORTS", "Support reasoning"),
        ("REFUTES", "Refute reasoning")
    ]
    dummy_med.synthesize_verdicts.assert_called_once_with(expected_verdicts, claim)
    assert final_verdict == "INCONCLUSIVE"

def test_advocate_step_receives_correct_evidence_options(
    advocate_mediator_strategy_factory: AdvocateMediatorStrategy,
    mock_llama_retriever: Mock,
    mock_advocate_step: Mock,
) -> None:
    """
    Test that the AdvocateStep is constructed with the correct evidence options, including a minimum score threshold. 
    
    (The actual filtering logic should be tested in the AdvocateStep or EvidenceStep tests.).
    """
    # Configure the dummy retriever to return evidence with varying confidence scores.
    retriever = mock_llama_retriever.return_value
    retriever.retrieve.return_value = [
        {"text": "High confidence evidence", "score": 0.9},
        {"text": "Medium confidence evidence", "score": 0.8},
        {"text": "Low confidence evidence", "score": 0.6}  # below threshold 0.7
    ]
    
    # Create a strategy with 1 advocate using the factory fixture.
    strategy = advocate_mediator_strategy_factory(1)
    claim = "Test claim for evidence filtering"
    strategy.evaluate_claim(claim)
    
    # Verify that AdvocateStep was constructed with the expected options.
    # According to the strategy implementation, AdvocateStep is instantiated as:
    #   AdvocateStep(retriever=retriever, options=advocate_options, evidence_options=evidence_options)
    expected_advocate_options = {
        "system_prompt": "test_system_prompt",
        "label_options": ["SUPPORTS", "PARTIALLY_SUPPORTS", "REFUTES"]
    }
    expected_evidence_options = {"min_score": 0.7}
    
    # We use mock.ANY for the retriever because we may not have direct control over it.
    from unittest.mock import ANY
    mock_advocate_step.assert_called_with(
        retriever=ANY,
        options=expected_advocate_options,
        evidence_options=expected_evidence_options
    )
    
    # Optionally, verify that the dummy advocate's evaluate_claim was called.
    dummy_adv = mock_advocate_step.return_value
    dummy_adv.evaluate_claim.assert_called_once_with(claim)


def test_fail_fast_on_advocate_error(
    mock_llama_indexer, 
    mock_llama_retriever, 
    mock_advocate_step, 
    mock_mediator_step
):
    """Test that the strategy raises an exception immediately if an advocate fails."""
    # Configure first advocate to raise an exception and second to succeed.
    advocate1 = Mock()
    advocate1.evaluate_claim.side_effect = Exception("Evidence evaluation failed")
    advocate2 = Mock()
    advocate2.evaluate_claim.return_value = ("SUPPORTS", "Successful evaluation")
    
    # Set the side effect on the patched AdvocateStep: first call fails, second would succeed.
    mock_advocate_step.side_effect = [advocate1, advocate2]

    # Instantiate the strategy without extra (unsupported) keyword arguments.
    strategy = AdvocateMediatorStrategy(
        indexer_options_list=[{"model_name": "test"}, {"model_name": "test"}],
        retriever_options_list=[{"top_k": 3}, {"top_k": 3}],
        advocate_options={},
        evidence_options={},
        mediator_options={}
    )

    # Expect the evaluation to raise an exception due to the failing advocate.
    with pytest.raises(Exception) as exc_info:
        strategy.evaluate_claim("Test claim")
    
    # Check that the exception message contains the expected text.
    assert "Evidence evaluation failed" in str(exc_info.value)
    
    # Verify that the second advocate's evaluate_claim was never called (fail-fast behavior).
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
    }]

    advocate_options = {
        'system_prompt': 'test advocate prompt',
        'label_options': ['CORRECT', 'INCORRECT', 'NOT_ENOUGH_INFO'],
    }

    evidence_options = {
        'min_score': 0.75
    }

    mediator_options = {}

    strategy = AdvocateMediatorStrategy(
        indexer_options_list=indexer_options_list,
        retriever_options_list=retriever_options_list,
        advocate_options=advocate_options,
        evidence_options=evidence_options,
        mediator_options=mediator_options,
    )

    # Configure mock responses
    mock_llama_retriever.return_value.retrieve.return_value = [
        {"text": "Evidence 1", "score": 0.9},
        {"text": "Evidence 2", "score": 0.85},
        {"text": "Evidence 3", "score": 0.8},
        {"text": "Evidence 4", "score": 0.76},
        {"text": "Evidence 5", "score": 0.74},  # Below threshold
    ]

    # Test with a realistic climate claim
    claim = "Global temperatures have not risen in the past decade"
    final_verdict, verdicts, reasonings = strategy.evaluate_claim(claim)

    # Verify the configuration was properly passed to the advocate step
    mock_advocate_step.assert_called_with(
        retriever=ANY,
        options=advocate_options,
        evidence_options=evidence_options,
    )