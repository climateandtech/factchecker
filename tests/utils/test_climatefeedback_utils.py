import pandas as pd
import pytest

from factchecker.utils.climatefeedback_utils import (
    evaluate_climatefeedback_claims,
    map_verdict,
    VALID_LEVELS,
)

def test_map_verdict_level_7():
    # Test level 7 mapping (most granular)
    assert map_verdict("correct", level=7) == "correct"
    assert map_verdict("inaccurate", level=7) == "inaccurate"
    assert map_verdict("imprecise", level=7) == "imprecise"
    assert map_verdict("misleading", level=7) == "misleading"
    assert map_verdict("flawed reasoning", level=7) == "flawed_reasoning"
    assert map_verdict("lacks context", level=7) == "lacks_context"
    assert map_verdict("unsupported", level=7) == "unsupported"
    assert map_verdict("correct but", level=7) == "correct_but"
    assert map_verdict("mostly correct", level=7) == "mostly_correct"
    assert map_verdict("mostly accurate", level=7) == "mostly_accurate"
    assert map_verdict("accurate", level=7) == "accurate"
    assert map_verdict(" Mostly Correct ", level=7) == "mostly_correct"  # Test stripping and case handling

def test_map_verdict_level_5():
    # Test level 5 mapping (medium granularity)
    assert map_verdict("incorrect", level=5) == "incorrect"
    assert map_verdict("inaccurate", level=5) == "incorrect"  # Maps to incorrect
    assert map_verdict("imprecise", level=5) == "imprecise"
    assert map_verdict("misleading", level=5) == "misleading"
    assert map_verdict("flawed reasoning", level=5) == "flawed_reasoning"
    assert map_verdict("lacks context", level=5) == "unsupported"  # Maps to unsupported
    assert map_verdict("unsupported", level=5) == "unsupported"
    assert map_verdict("correct but", level=5) == "mostly_correct"  # Maps to mostly_correct
    assert map_verdict("mostly correct", level=5) == "mostly_correct"
    assert map_verdict("mostly accurate", level=5) == "correct"  # Maps to correct
    assert map_verdict("accurate", level=5) == "correct"  # Maps to correct
    assert map_verdict("correct", level=5) == "correct"

def test_map_verdict_level_2():
    # Test level 2 mapping (binary classification)
    # Test all correct labels
    assert map_verdict("correct", level=2) == "correct"
    assert map_verdict("mostly correct", level=2) == "correct"
    assert map_verdict("correct but", level=2) == "correct"
    assert map_verdict("mostly accurate", level=2) == "correct"
    assert map_verdict("accurate", level=2) == "correct"
    
    # Test all incorrect labels
    assert map_verdict("incorrect", level=2) == "incorrect"
    assert map_verdict("inaccurate", level=2) == "incorrect"
    assert map_verdict("unsupported", level=2) == "incorrect"
    assert map_verdict("misleading", level=2) == "incorrect"
    assert map_verdict("flawed reasoning", level=2) == "incorrect"
    assert map_verdict("lacks context", level=2) == "incorrect"
    assert map_verdict("mostly inaccurate", level=2) == "incorrect"
    assert map_verdict("imprecise", level=2) == "incorrect"

def test_map_verdict_unknown():
    # Test handling of unknown verdicts
    assert map_verdict("nonexistent_verdict", level=7) == "unknown"
    assert map_verdict("nonexistent_verdict", level=5) == "unknown"
    assert map_verdict("nonexistent_verdict", level=2) == "unknown"
    assert map_verdict("", level=2) == "unknown"  # Test empty string

def test_map_verdict_case_and_spacing():
    # Test case insensitivity and spacing handling
    assert map_verdict(" Correct ", level=2) == "correct"
    assert map_verdict("MOSTLY CORRECT", level=2) == "correct"
    assert map_verdict("Lacks Context", level=5) == "unsupported"
    assert map_verdict("FLAWED REASONING", level=7) == "flawed_reasoning"
    assert map_verdict("  Accurate  ", level=7) == "accurate"

def test_map_verdict_invalid_level():
    # Test invalid level handling raises ValueError
    with pytest.raises(ValueError, match=f"Level must be one of {VALID_LEVELS}"):
        map_verdict("correct", level=1)
    with pytest.raises(ValueError, match=f"Level must be one of {VALID_LEVELS}"):
        map_verdict("correct", level=6)
    with pytest.raises(ValueError, match=f"Level must be one of {VALID_LEVELS}"):
        map_verdict("correct", level=0)


# --- evaluate_climatefeedback_claims: return (collectors, errors), error tracking, claim_indices ---


def test_evaluate_climatefeedback_claims_returns_collectors_and_errors():
    """evaluate_climatefeedback_claims returns (collectors, errors); no errors when all succeed."""
    strategy = _make_mock_strategy(
            [
                ("correct", ["SUPPORTS"], ["r1"], None, None),
                ("incorrect", ["REFUTES"], ["r2"], None, None),
            ]
        )
    claims_df = pd.DataFrame({
        "Claim": ["First claim.", "Second claim."],
        "Climate Feedback": ["correct", "incorrect"],
    })
    collectors, errors = evaluate_climatefeedback_claims(strategy, claims_df, num_advocates=1)
    assert isinstance(collectors, dict)
    assert isinstance(errors, list)
    assert len(errors) == 0
    assert len(collectors["true_labels"]) == 2
    assert "claim_indices" in collectors
    assert len(collectors["claim_indices"]) == 2


def test_evaluate_climatefeedback_claims_tracks_errors_on_failure():
    """When strategy raises on some claims, errors list is populated and collectors only have successes."""
    strategy = _make_mock_strategy(
        [
            ("correct", ["SUPPORTS"], ["r1"], None, None),
            None,  # second call raises
            ("correct", ["SUPPORTS"], ["r3"], None, None),
        ],
        raise_on_none=True,
    )
    claims_df = pd.DataFrame({
        "Claim": ["Claim one.", "Claim two fail.", "Claim three."],
        "Climate Feedback": ["correct", "incorrect", "correct"],
    })
    collectors, errors = evaluate_climatefeedback_claims(strategy, claims_df, num_advocates=1)
    assert len(collectors["true_labels"]) == 2
    assert len(errors) == 1
    assert errors[0]["error_type"] == "ValueError"
    assert "Intentional failure" in errors[0]["error_message"]
    assert errors[0]["claim_index"] == 1
    assert "Claim two" in errors[0]["claim_preview"]


def test_evaluate_climatefeedback_claims_claim_indices_only_successes():
    """claim_indices contains only indices of successfully evaluated claims."""
    strategy = _make_mock_strategy(
        [
            ("correct", ["SUPPORTS"], ["r1"], None, None),
            None,  # raise on index 1
            ("incorrect", ["REFUTES"], ["r3"], None, None),
        ],
        raise_on_none=True,
    )
    claims_df = pd.DataFrame({
        "Claim": ["A", "B", "C"],
        "Climate Feedback": ["correct", "incorrect", "incorrect"],
    })
    collectors, errors = evaluate_climatefeedback_claims(strategy, claims_df, num_advocates=1)
    # Indices from iterrows() are 0, 1, 2 (default RangeIndex)
    assert collectors["claim_indices"] == [0, 2]
    assert len(errors) == 1 and errors[0]["claim_index"] == 1


def _make_mock_strategy(returns, raise_on_none=False):
    """Strategy whose evaluate_claim returns the next item from returns; None means raise if raise_on_none."""

    class MockStrategy:
        def __init__(self):
            self.returns = returns
            self.raise_on_none = raise_on_none
            self.call_count = 0

        def evaluate_claim(self, claim):
            i = self.call_count
            self.call_count += 1
            val = self.returns[i] if i < len(self.returns) else self.returns[-1]
            if val is None and self.raise_on_none:
                raise ValueError("Intentional failure for testing")
            return val

    return MockStrategy() 