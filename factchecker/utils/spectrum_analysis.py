"""
Analyze Spectrum experiment results (CSV).

Loads a spectrum_claims_results_*.csv, resolves true labels to the Science Feedback
verdict scale, and computes classification metrics (primary verdict) and weighted
distribution metrics (log loss, top-k accuracy). No LLM or model calls.

Used by factchecker.tools.analyze_spectrum_results and the Spectrum experiment script.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from factchecker.utils.climatefeedback_utils import map_verdict
from factchecker.utils.metrics import calculate_classification_metrics

from factchecker.experiments.advocate_mediator_climatefeedback_spectrum.weighted_metrics import (
    format_weighted_metrics,
    log_loss,
    top_k_accuracy,
)


def _cluster_verdict_for_metrics(label: str, level: int) -> str:
    """Map a single verdict (our format, e.g. mostly_accurate) to clustered label at given level (2 or 5)."""
    if not label or not isinstance(label, str):
        return "unknown"
    normalized_input = label.strip().lower().replace("_", " ")
    return map_verdict(normalized_input, level=level)


@dataclass
class SpectrumAnalysisResult:
    """Result of running spectrum analysis on a results CSV."""

    n_rows: int
    classification_report: Optional[str] = None
    weighted_metrics_text: Optional[str] = None
    log_loss_value: Optional[float] = None
    top_2_accuracy: Optional[float] = None
    top_3_accuracy: Optional[float] = None
    error: Optional[str] = None
    # Normalized (clustered) metrics: key (level, role) -> report string
    normalized_reports: Optional[dict] = None


def _normalize_predicted(p: str) -> str:
    if not p or not isinstance(p, str):
        return "unknown"
    s = p.strip().lower().replace(" ", "_")
    if s == "not_enough_info":
        return "not_enough_information"
    return s


def _resolve_true_labels(df: pd.DataFrame) -> Optional[List[str]]:
    """
    Resolve true labels to Science Feedback verdict scale (level-7).
    Prefers 'True Label (SF verdict)', then 'Mapped True Label', then maps 'True Label' with level=7.
    """
    if "True Label (SF verdict)" in df.columns:
        s = df["True Label (SF verdict)"].astype(str).str.strip().str.lower().str.replace(" ", "_")
        return s.tolist()
    if "Mapped True Label" in df.columns:
        s = df["Mapped True Label"].astype(str).str.strip().str.lower().str.replace(" ", "_")
        return s.tolist()
    if "True Label" in df.columns:
        return [map_verdict(str(x), level=7) for x in df["True Label"]]
    return None


def run_spectrum_analysis(csv_path: str) -> SpectrumAnalysisResult:
    """
    Load a Spectrum results CSV and compute classification + weighted distribution metrics.

    Args:
        csv_path: Path to spectrum_claims_results_*.csv (must have 'Weighted verdict'
                  and one of 'True Label (SF verdict)', 'Mapped True Label', or 'True Label').

    Returns:
        SpectrumAnalysisResult with n_rows, classification_report, weighted_metrics_text,
        log_loss_value, top_2_accuracy, top_3_accuracy, or error if invalid.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return SpectrumAnalysisResult(n_rows=0, error=f"Failed to load CSV: {e}")

    if "Weighted verdict" not in df.columns:
        return SpectrumAnalysisResult(
            n_rows=len(df),
            error=f"CSV must have 'Weighted verdict' column. Found: {list(df.columns)}",
        )

    true_list = _resolve_true_labels(df)
    if true_list is None:
        return SpectrumAnalysisResult(
            n_rows=len(df),
            error="CSV must have one of 'True Label (SF verdict)', 'Mapped True Label', or 'True Label'.",
        )

    weighted_verdicts = df["Weighted verdict"].astype(str).tolist()
    true_list = [t if t and t != "nan" else "unknown" for t in true_list]

    # Classification metrics (primary verdict vs true)
    classification_report: Optional[str] = None
    pred_list: Optional[List[str]] = None
    advocate_list: Optional[List[str]] = None
    if "Predicted Verdict" in df.columns:
        pred_list = [_normalize_predicted(str(p)) for p in df["Predicted Verdict"]]
        try:
            classification_report = calculate_classification_metrics(
                true_list, pred_list, verdict_mapper=None
            )
        except Exception:
            pass
    if "Advocate Primary Verdict" in df.columns:
        advocate_list = [_normalize_predicted(str(p)) for p in df["Advocate Primary Verdict"]]

    # Normalized (clustered) metrics: level 2 and 5 for advocate and mediator
    normalized_reports: Optional[dict] = None
    if true_list and (pred_list is not None or advocate_list is not None):
        normalized_reports = {}
        for level in (2, 5):
            true_clustered = [_cluster_verdict_for_metrics(t, level) for t in true_list]
            level_name = "correct/incorrect" if level == 2 else "level-5"
            if advocate_list is not None and len(advocate_list) == len(true_list):
                adv_clustered = [_cluster_verdict_for_metrics(a, level) for a in advocate_list]
                try:
                    normalized_reports[(level, "advocate")] = (
                        f"--- Advocate primary vs true (normalized, level={level}: {level_name}) ---\n"
                        + calculate_classification_metrics(
                            true_clustered, adv_clustered, verdict_mapper=None
                        )
                    )
                except Exception:
                    pass
            if pred_list is not None and len(pred_list) == len(true_list):
                med_clustered = [_cluster_verdict_for_metrics(p, level) for p in pred_list]
                try:
                    normalized_reports[(level, "mediator")] = (
                        f"--- Mediator predicted vs true (normalized, level={level}: {level_name}) ---\n"
                        + calculate_classification_metrics(
                            true_clustered, med_clustered, verdict_mapper=None
                        )
                    )
                except Exception:
                    pass

    # Weighted distribution metrics
    weighted_metrics_text = format_weighted_metrics(true_list, weighted_verdicts)
    log_loss_value = log_loss(true_list, weighted_verdicts)
    top_2 = top_k_accuracy(true_list, weighted_verdicts, 2)
    top_3 = top_k_accuracy(true_list, weighted_verdicts, 3)

    return SpectrumAnalysisResult(
        n_rows=len(df),
        classification_report=classification_report,
        weighted_metrics_text=weighted_metrics_text,
        log_loss_value=log_loss_value,
        top_2_accuracy=top_2,
        top_3_accuracy=top_3,
        normalized_reports=normalized_reports or None,
    )


def find_latest_spectrum_results_csv(
    search_dirs: Optional[List[Path]] = None,
) -> Optional[Path]:
    """
    Find the most recent spectrum_claims_results_*.csv under the given directories.
    Defaults to experiments/results and results relative to cwd and package root.
    """
    if search_dirs is None:
        base = Path(__file__).resolve().parent.parent
        search_dirs = [
            Path("experiments/results"),
            Path("results"),
            base / "experiments" / "results",
        ]
    for d in search_dirs:
        if not d.exists():
            continue
        files = sorted(
            d.glob("spectrum_claims_results_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if files:
            return files[0]
    return None
