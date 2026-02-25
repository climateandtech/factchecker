"""Tests for experiment_utils: show_evaluation_errors, save_evaluation_errors, create_results_dataframe (partial results)."""
import os
import pandas as pd
import pytest

from factchecker.utils.experiment_utils import (
    create_results_dataframe,
    initialize_results_collectors,
    save_evaluation_errors,
    show_evaluation_errors,
)


def test_show_evaluation_errors_empty(capfd):
    """Empty errors list should print nothing."""
    show_evaluation_errors([])
    out, _ = capfd.readouterr()
    assert "Evaluation errors" not in out
    assert out.strip() == ""


def test_show_evaluation_errors_prints_summary_and_details(capfd):
    """Errors are printed: summary (count, by type) and per-error details."""
    errors = [
        {
            "claim_index": 0,
            "error_type": "ValueError",
            "error_message": "Parse failed",
            "claim_preview": "Some claim text...",
        },
        {
            "claim_index": 2,
            "error_type": "TimeoutError",
            "error_message": "Took too long",
            "claim_preview": "Another claim...",
        },
    ]
    show_evaluation_errors(errors, total_claims=5)
    out, _ = capfd.readouterr()
    assert "--- Evaluation errors ---" in out
    assert "Failed claims: 2 / 5" in out
    assert "By error type:" in out
    assert "ValueError" in out
    assert "TimeoutError" in out
    assert "Details:" in out
    assert "[0] ValueError: Parse failed" in out
    assert "Some claim text" in out
    assert "[2] TimeoutError: Took too long" in out
    assert "---" in out


def test_show_evaluation_errors_no_total_claims(capfd):
    """total_claims can be omitted (shows '?' for total)."""
    show_evaluation_errors([{"claim_index": 0, "error_type": "X", "error_message": "m"}], total_claims=None)
    out, _ = capfd.readouterr()
    assert "Failed claims: 1 / ?" in out


def test_create_results_dataframe_partial_claims_claim_indices():
    """When claim_indices is set, results DataFrame contains only those claims (partial results)."""
    claims = pd.DataFrame({
        "Claim": ["claim A", "claim B", "claim C"],
        "Climate Feedback": ["correct", "incorrect", "correct"],
    })
    # Simulate: only claims at index 0 and 2 were successfully evaluated
    collectors = initialize_results_collectors(1)
    collectors["true_labels"] = ["correct", "correct"]
    collectors["predicted_results"] = ["correct", "incorrect"]
    collectors["mediator_reasonings"] = ["r1", "r2"]
    collectors["advocate_evidences"] = [["e1", "e2"]]
    collectors["advocate_verdicts"] = [["v1", "v2"]]
    collectors["advocate_reasonings"] = [["x1", "x2"]]
    collectors["claim_indices"] = [0, 2]

    result = create_results_dataframe(claims, collectors)

    assert len(result) == 2
    assert list(result["Claim"].values) == ["claim A", "claim C"]
    assert list(result["True Label"].values) == ["correct", "correct"]


def test_create_results_dataframe_full_claims_no_claim_indices():
    """Without claim_indices, full length must match (classic behavior)."""
    claims = pd.DataFrame({
        "Claim": ["claim A", "claim B"],
        "Climate Feedback": ["correct", "incorrect"],
    })
    collectors = initialize_results_collectors(1)
    collectors["true_labels"] = ["correct", "incorrect"]
    collectors["predicted_results"] = ["correct", "incorrect"]
    collectors["mediator_reasonings"] = ["r1", "r2"]
    collectors["advocate_evidences"] = [["e1", "e2"]]
    collectors["advocate_verdicts"] = [["v1", "v2"]]
    collectors["advocate_reasonings"] = [["x1", "x2"]]
    # no claim_indices

    result = create_results_dataframe(claims, collectors)

    assert len(result) == 2
    assert list(result["Claim"].values) == ["claim A", "claim B"]


def test_create_results_dataframe_length_mismatch_raises():
    """When not using claim_indices, length mismatch raises ValueError."""
    claims = pd.DataFrame({"Claim": ["a", "b"], "Climate Feedback": ["c", "i"]})
    collectors = initialize_results_collectors(1)
    collectors["true_labels"] = ["correct", "incorrect", "correct"]  # 3
    collectors["predicted_results"] = ["c", "i", "c"]
    collectors["mediator_reasonings"] = ["r1", "r2", "r3"]
    collectors["advocate_evidences"] = [["e1", "e2", "e3"]]
    collectors["advocate_verdicts"] = [["v1", "v2", "v3"]]
    collectors["advocate_reasonings"] = [["x1", "x2", "x3"]]

    with pytest.raises(ValueError, match="Number of claims doesn't match"):
        create_results_dataframe(claims, collectors)


def test_create_results_dataframe_claim_indices_length_mismatch_raises():
    """claim_indices length must match collected results length."""
    claims = pd.DataFrame({"Claim": ["a", "b", "c"], "Climate Feedback": ["c", "i", "c"]})
    collectors = initialize_results_collectors(1)
    collectors["true_labels"] = ["correct", "correct"]
    collectors["predicted_results"] = ["c", "c"]
    collectors["mediator_reasonings"] = ["r1", "r2"]
    collectors["advocate_evidences"] = [["e1", "e2"]]
    collectors["advocate_verdicts"] = [["v1", "v2"]]
    collectors["advocate_reasonings"] = [["x1", "x2"]]
    collectors["claim_indices"] = [0, 2, 1]  # 3 indices but only 2 results

    with pytest.raises(ValueError, match="claim_indices length must match"):
        create_results_dataframe(claims, collectors)


def test_save_evaluation_errors_empty_returns_none(tmp_path):
    """Empty errors list returns None and does not write a file."""
    path = save_evaluation_errors([], base_path=str(tmp_path))
    assert path is None
    assert list(tmp_path.iterdir()) == []


def test_save_evaluation_errors_writes_csv(tmp_path):
    """Errors are written to a timestamped CSV; returns path."""
    errors = [
        {"claim_index": 0, "error_type": "ValueError", "error_message": "Bad", "claim_preview": "x"},
    ]
    path = save_evaluation_errors(errors, base_path=str(tmp_path), prefix="err")
    assert path is not None
    assert os.path.basename(path).startswith("err_")
    assert path.endswith(".csv")
    df = pd.read_csv(path)
    assert len(df) == 1
    assert df["error_type"].iloc[0] == "ValueError"
    assert df["claim_preview"].iloc[0] == "x"
