"""
Metrics for weighted verdict distributions (Spectrum experiment).

- Log loss (cross-entropy): uses full distribution; lower is better.
- Top-k accuracy: true verdict in model's top-k by weight.
"""
import json
import logging
import math
from typing import List, Optional

logger = logging.getLogger(__name__)

from factchecker.experiments.advocate_mediator_climatefeedback_spectrum.advocate_mediator_climatefeedback_spectrum_prompts import (
    SPECTRUM_CATEGORIES,
)

# Avoid log(0)
_EPS = 1e-10


def _parse_weights(weighted_verdict: str) -> Optional[dict]:
    """Parse weighted_verdict JSON to dict category -> weight. Returns None if invalid."""
    if not weighted_verdict or not weighted_verdict.strip():
        return None
    s = weighted_verdict.strip()
    if not s.startswith("{"):
        return None
    try:
        d = json.loads(s)
        if not isinstance(d, dict):
            return None
        out = {}
        for c in SPECTRUM_CATEGORIES:
            out[c] = float(d.get(c, 0.0))
        return out
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def log_loss(
    true_labels: List[str],
    weighted_verdicts: List[str],
) -> Optional[float]:
    """
    Mean cross-entropy loss: -log(p_true) per sample.
    True label is one-hot; predicted is the weight distribution.
    Lower is better. Returns None if no valid pairs.
    """
    if len(true_labels) != len(weighted_verdicts) or not true_labels:
        return None
    total = 0.0
    count = 0
    for true, wv in zip(true_labels, weighted_verdicts, strict=True):
        weights = _parse_weights(wv)
        if weights is None:
            # Fallback: treat as one-hot on a single verdict (wv is the verdict string)
            p_true = 1.0 if true.strip().lower().replace(" ", "_") == wv.strip().lower().replace(" ", "_") else 0.0
        else:
            true_norm = true.strip().lower().replace(" ", "_")
            if true_norm == "not_enough_info":
                true_norm = "not_enough_information"
            p_true = weights.get(true_norm, 0.0)
        p_true = max(p_true, _EPS)
        total += -math.log(p_true)
        count += 1
    return total / count if count else None


def top_k_accuracy(
    true_labels: List[str],
    weighted_verdicts: List[str],
    k: int,
) -> Optional[float]:
    """
    Fraction of samples where the true label is in the model's top-k by weight.
    Returns value in [0, 1] or None if no valid pairs.
    """
    if len(true_labels) != len(weighted_verdicts) or not true_labels or k < 1:
        return None
    hits = 0
    for true, wv in zip(true_labels, weighted_verdicts, strict=True):
        true_norm = true.strip().lower().replace(" ", "_")
        if true_norm == "not_enough_info":
            true_norm = "not_enough_information"
        weights = _parse_weights(wv)
        if weights is None:
            # Fallback: single verdict string
            pred_norm = wv.strip().lower().replace(" ", "_") if wv else ""
            top_k = [pred_norm] if pred_norm else []
        else:
            sorted_cats = sorted(weights.keys(), key=lambda c: weights[c], reverse=True)
            top_k = sorted_cats[:k]
        if true_norm in top_k:
            hits += 1
    return hits / len(true_labels)


def format_weighted_metrics(
    true_labels: List[str],
    weighted_verdicts: List[str],
) -> str:
    """Compute log loss and top-2/top-3 accuracy and return a formatted string for logging/print."""
    lines = ["\n--- Weighted distribution metrics ---"]
    ll = log_loss(true_labels, weighted_verdicts)
    if ll is not None:
        lines.append(f"Log loss (cross-entropy): {ll:.4f} (lower is better)")
    t2 = top_k_accuracy(true_labels, weighted_verdicts, 2)
    if t2 is not None:
        lines.append(f"Top-2 accuracy:           {t2:.2%} (true verdict in top-2 by weight)")
    t3 = top_k_accuracy(true_labels, weighted_verdicts, 3)
    if t3 is not None:
        lines.append(f"Top-3 accuracy:           {t3:.2%} (true verdict in top-3 by weight)")
    lines.append("---")
    return "\n".join(lines)
