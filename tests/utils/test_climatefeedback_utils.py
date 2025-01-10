import pytest
from factchecker.utils.climatefeedback_utils import map_verdict, VALID_LEVELS

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