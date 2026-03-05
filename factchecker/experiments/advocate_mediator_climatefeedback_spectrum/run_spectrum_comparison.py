"""
Compare two mediator configurations on the same claims:

1. **Formula run**: Advocate outputs weights only → mediator combines by formula (average + argmax), no LLM.
2. **LLM + evidence run**: Advocate outputs label + weights → mediator is LLM and also sees the most
   relevant supporting/refuting chunks.

Both runs report normalized metrics (level 2: correct/incorrect; level 5: five categories) for comparison.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

from llama_index.core import Settings
from tqdm import tqdm

from factchecker.experiments.advocate_mediator_climatefeedback.advocate_mediator_climatefeedback import (
    setup_sources,
)
from factchecker.experiments.advocate_mediator_climatefeedback_spectrum.advocate_mediator_climatefeedback_spectrum import (
    EXPERIMENT_PARAMS,
    _cluster_verdict_for_metrics,
    _primary_verdict_from_final,
    _spectrum_verdict_mapper_predicted,
    _spectrum_verdict_mapper_true,
    setup_strategy,
)
from factchecker.utils.experiment_utils import collect_evaluation_results
from factchecker.utils.climatefeedback_utils import (
    sample_climatefeedback_claims,
)
from factchecker.utils.experiment_utils import (
    configure_logging,
    initialize_results_collectors,
    save_results,
    show_evaluation_errors,
    verify_environment,
)
from factchecker.utils.metrics import calculate_classification_metrics

logger = logging.getLogger(__name__)


def _run_one(
    run_name: str,
    params: Dict[str, Any],
    sampled_claims,
    downloaded_files: Optional[List[str]] = None,
):
    """Run one configuration; return collectors and errors."""
    strategy = setup_strategy(params=params, downloaded_files=downloaded_files)
    num_advocates = len(strategy.advocate_steps)
    collectors = initialize_results_collectors(num_advocates)
    collectors["weighted_verdicts"] = []
    collectors["advocate_evidence_metadata"] = [[] for _ in range(num_advocates)]
    errors: List[Dict[str, Any]] = []

    for idx, row in tqdm(
        sampled_claims.iterrows(),
        total=len(sampled_claims),
        desc=f"{run_name}: Evaluating claims",
    ):
        claim_text = row["Claim"]
        true_label = row["Climate Feedback"]
        try:
            (
                final_verdict,
                verdicts,
                reasonings,
                evidence_metadata_per_advocate,
                advocate_weights_per_advocate,
            ) = strategy.evaluate_claim(claim_text)
            primary_verdict = _primary_verdict_from_final(final_verdict)
            collectors = collect_evaluation_results(
                collectors,
                (true_label, primary_verdict, verdicts, reasonings),
                num_advocates=len(verdicts) if not collectors["advocate_evidences"] else None,
                claim_index=idx,
            )
            collectors["weighted_verdicts"].append(final_verdict)
            for i in range(num_advocates):
                meta = (
                    evidence_metadata_per_advocate[i]
                    if evidence_metadata_per_advocate and i < len(evidence_metadata_per_advocate)
                    else None
                )
                collectors["advocate_evidence_metadata"][i].append(meta if meta else [])
            if "advocate_weighted_verdicts" in collectors:
                for i in range(num_advocates):
                    w = (
                        advocate_weights_per_advocate[i]
                        if advocate_weights_per_advocate and i < len(advocate_weights_per_advocate)
                        else None
                    )
                    collectors["advocate_weighted_verdicts"][i].append(
                        json.dumps(w) if isinstance(w, dict) else w
                    )
        except Exception as e:
            logger.error("Error processing claim %s (%s): %s", idx, run_name, e)
            errors.append({
                "claim_index": idx,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "claim_preview": (claim_text[:80] + "...") if len(claim_text) > 80 else claim_text,
            })
    return collectors, errors


def main(experiment_options: Optional[Dict[str, Any]] = None):
    """Run both configurations on the same claims and compare normalized metrics."""
    configure_logging()
    params = {**EXPERIMENT_PARAMS, **(experiment_options or {})}
    params["total_samples"] = params.get("total_samples", 10)

    verify_environment()
    downloaded_files = setup_sources(params=params)
    Settings.chunk_size = params["chunk_size"]
    Settings.chunk_overlap = params["chunk_overlap"]

    logger.info("Sampling claims once for both runs...")
    sampled_claims = sample_climatefeedback_claims(
        csv_path=params["dataset_path"],
        total_samples=params["total_samples"],
        correct_ratio=params["correct_ratio"],
    )

    # Run 1: weights from advocate → mediator = formula (no LLM)
    run1_params = {
        **params,
        "advocate_output_mode": "weights_only",
        "mediator_mode": "formula",
    }
    logger.info("Run 1: advocate weights_only + mediator formula")
    collectors1, errors1 = _run_one("formula", run1_params, sampled_claims, downloaded_files)
    show_evaluation_errors(errors1, total_claims=len(sampled_claims))

    # Run 2: label + weights from advocate → mediator LLM with evidence summary
    run2_params = {
        **params,
        "advocate_output_mode": "label_and_weights",
        "mediator_mode": "llm_with_evidence",
    }
    logger.info("Run 2: advocate label_and_weights + mediator llm_with_evidence")
    collectors2, errors2 = _run_one("llm_with_evidence", run2_params, sampled_claims, downloaded_files)
    show_evaluation_errors(errors2, total_claims=len(sampled_claims))

    # Align results by claim index (only claims that succeeded in both runs)
    indices1 = set(collectors1.get("claim_indices", []))
    indices2 = set(collectors2.get("claim_indices", []))
    common_indices = sorted(indices1 & indices2)
    if not common_indices:
        logger.warning("No claims succeeded in both runs; cannot compare.")
        return

    # Build aligned lists for common claims
    pos1 = {idx: i for i, idx in enumerate(collectors1.get("claim_indices", []))}
    pos2 = {idx: i for i, idx in enumerate(collectors2.get("claim_indices", []))}
    true_labels = []
    run1_primaries = []
    run2_primaries = []
    claims_subset = []
    for idx in common_indices:
        i1, i2 = pos1[idx], pos2[idx]
        true_labels.append(collectors1["true_labels"][i1])
        run1_primaries.append(collectors1["predicted_results"][i1])
        run2_primaries.append(collectors2["predicted_results"][i2])
        claims_subset.append(sampled_claims.loc[idx, "Claim"])

    true_mapped = [_spectrum_verdict_mapper_true(l) for l in true_labels]
    run1_mapped = [_spectrum_verdict_mapper_predicted(p) for p in run1_primaries]
    run2_mapped = [_spectrum_verdict_mapper_predicted(p) for p in run2_primaries]

    # Normalized (level 2 and level 5) metrics for each run
    print("\n" + "=" * 60)
    print("COMPARISON: Formula vs LLM+evidence (normalized labels)")
    print("=" * 60)

    for level in (2, 5):
        level_name = "correct/incorrect (level 2)" if level == 2 else "level-5 categories"
        true_clustered = [_cluster_verdict_for_metrics(l, level) for l in true_mapped]
        run1_clustered = [_cluster_verdict_for_metrics(l, level) for l in run1_mapped]
        run2_clustered = [_cluster_verdict_for_metrics(l, level) for l in run2_mapped]

        m1 = calculate_classification_metrics(true_clustered, run1_clustered, verdict_mapper=None)
        m2 = calculate_classification_metrics(true_clustered, run2_clustered, verdict_mapper=None)

        print(f"\n--- {level_name} ---")
        print("Run 1 (formula: advocate weights → average + argmax):")
        print(m1)
        print("\nRun 2 (LLM+evidence: advocate label+weights → mediator sees supporting/refuting chunks):")
        print(m2)

    # Comparison CSV: Claim, True Label, Run1_mediator_primary, Run2_mediator_primary, True_L2, Run1_L2, Run2_L2, True_L5, Run1_L5, Run2_L5
    import pandas as pd

    true_l2 = [_cluster_verdict_for_metrics(l, 2) for l in true_mapped]
    true_l5 = [_cluster_verdict_for_metrics(l, 5) for l in true_mapped]
    run1_l2 = [_cluster_verdict_for_metrics(l, 2) for l in run1_mapped]
    run1_l5 = [_cluster_verdict_for_metrics(l, 5) for l in run1_mapped]
    run2_l2 = [_cluster_verdict_for_metrics(l, 2) for l in run2_mapped]
    run2_l5 = [_cluster_verdict_for_metrics(l, 5) for l in run2_mapped]

    comparison_df = pd.DataFrame({
        "Claim": claims_subset,
        "True Label": true_labels,
        "Run1_mediator_primary": run1_primaries,
        "Run2_mediator_primary": run2_primaries,
        "True_L2": true_l2,
        "Run1_L2": run1_l2,
        "Run2_L2": run2_l2,
        "True_L5": true_l5,
        "Run1_L5": run1_l5,
        "Run2_L5": run2_l5,
    })
    out_path = save_results(
        comparison_df,
        base_path="experiments/results",
        prefix="spectrum_comparison_formula_vs_llm_evidence",
    )
    logger.info("Comparison CSV saved to: %s", out_path)
    print(f"\nComparison saved to: {out_path}")
    print(f"Compared {len(common_indices)} claims (successful in both runs).")


if __name__ == "__main__":
    main()
