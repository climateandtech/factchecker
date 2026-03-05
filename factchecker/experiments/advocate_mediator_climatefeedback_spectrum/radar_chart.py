"""
Radar chart visualization for weighted verdict distribution (Spectrum experiment).
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from factchecker.experiments.advocate_mediator_climatefeedback_spectrum.advocate_mediator_climatefeedback_spectrum_prompts import (
    SPECTRUM_CATEGORIES,
)


def _get_weights_dict(weighted_verdict: str) -> Optional[Dict[str, float]]:
    """Parse weighted_verdict (JSON string or raw) to dict of category -> weight. Returns None if invalid."""
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


def plot_radar(
    weights: Dict[str, float],
    output_path: str,
    title: str = "Verdict spectrum",
) -> bool:
    """
    Plot a single radar chart for one weight distribution.
    Returns True if plotted, False if matplotlib unavailable or weights invalid.
    """
    try:
        import math
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available; skipping radar chart.")
        return False

    categories = [c.replace("_", " ").title() for c in SPECTRUM_CATEGORIES]
    values = [weights.get(c_key, 0.0) for c_key in SPECTRUM_CATEGORIES]
    n = len(categories)
    angles = [2 * math.pi * i / n for i in range(n)]
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    ax.plot(angles, values, "o-", linewidth=2, label="Weights")
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=12, pad=20)
    ax.grid(True)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Radar chart saved to %s", output_path)
    return True


def plot_radar_summary(
    weighted_verdicts: List[str],
    output_path: str,
    title: str = "Mean verdict spectrum (all claims)",
) -> bool:
    """
    Plot a single radar chart of mean weights across all claims.
    Returns True if plotted, False otherwise.
    """
    try:
        import math
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available; skipping radar summary chart.")
        return False

    dicts = [_get_weights_dict(w) for w in weighted_verdicts]
    dicts = [d for d in dicts if d is not None]
    if not dicts:
        logger.warning("No valid weighted verdicts for radar summary.")
        return False

    mean_weights = {}
    for c in SPECTRUM_CATEGORIES:
        mean_weights[c] = sum(d.get(c, 0.0) for d in dicts) / len(dicts)

    return plot_radar(mean_weights, output_path, title=title)


def generate_radar_charts(
    weighted_verdicts: List[str],
    output_dir: str = "experiments/results/spectrum_radar",
    per_claim: bool = True,
    summary: bool = True,
) -> List[str]:
    """
    Generate radar charts for weighted verdicts.
    If per_claim, one chart per claim (claim index in filename).
    If summary, one chart of mean weights.
    Returns list of saved file paths.
    """
    saved: List[str] = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if summary and weighted_verdicts:
        path = str(Path(output_dir) / "spectrum_summary.png")
        if plot_radar_summary(weighted_verdicts, path, title="Mean verdict spectrum (all claims)"):
            saved.append(path)

    if per_claim:
        for i, wv in enumerate(weighted_verdicts):
            d = _get_weights_dict(wv)
            if d is None:
                continue
            path = str(Path(output_dir) / f"spectrum_claim_{i}.png")
            if plot_radar(d, path, title=f"Verdict spectrum (claim {i})"):
                saved.append(path)

    return saved
