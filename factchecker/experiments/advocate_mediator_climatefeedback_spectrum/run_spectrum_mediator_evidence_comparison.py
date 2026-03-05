"""
Compare mediator with no evidence vs mediator with 3 advocate-selected supporting chunks.

(a) Mediator sees only verdicts and reasonings (no evidence).
(b) Mediator sees verdicts, reasonings, and up to 3 chunks the advocate used to support its verdict
    (supporting chunks only, no contradicting evidence).

Both runs use the same advocate setup and chunk labelling; only what the mediator sees differs.
"""
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from llama_index.core import Settings
from tqdm import tqdm

from factchecker.experiments.advocate_mediator_climatefeedback.advocate_mediator_climatefeedback import (
    setup_sources,
)
from factchecker.experiments.advocate_mediator_climatefeedback_spectrum.advocate_mediator_climatefeedback_spectrum import (
    EXPERIMENT_PARAMS,
    _cluster_verdict_for_metrics,
    _spectrum_verdict_mapper_predicted,
    _spectrum_verdict_mapper_true,
    evaluate_spectrum_claims,
    setup_strategy,
)
from factchecker.utils.climatefeedback_utils import sample_climatefeedback_claims
from factchecker.utils.experiment_utils import (
    configure_logging,
    save_results,
    show_evaluation_errors,
    verify_environment,
)
from factchecker.utils.metrics import calculate_classification_metrics
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


def main(experiment_options: Optional[Dict[str, Any]] = None):
    """Run (a) mediator no evidence and (b) mediator with 3 supporting chunks; compare on same claims."""
    configure_logging()
    params = {**EXPERIMENT_PARAMS, **(experiment_options or {})}
    params["ablation_compare_chunk_labelling"] = False
    params["compare_mediator_modes"] = False
    params["total_samples"] = params.get("total_samples", 10)
    params["use_evidence_classifier"] = True  # need classified chunks to build advocate proof

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
    total_claims = len(sampled_claims)
    parallel_claims = params.get("parallel_claims", 0) or 0

    # Run (a): mediator sees no evidence
    params_no_evidence = {**params, "mediator_mode": "llm"}
    strategy_a = setup_strategy(params=params_no_evidence, downloaded_files=downloaded_files)
    num_advocates = len(strategy_a.advocate_steps)
    collectors_a, errors_a = evaluate_spectrum_claims(
        strategy_a, sampled_claims, num_advocates, parallel_claims=parallel_claims
    )
    show_evaluation_errors(errors_a, total_claims=total_claims)

    # Run (b): mediator sees only 3 supporting chunks (advocate's proof)
    params_with_proof = {**params, "mediator_mode": "llm_with_advocate_proof", "mediator_proof_max_chunks": 3}
    strategy_b = setup_strategy(params=params_with_proof, downloaded_files=downloaded_files)
    collectors_b, errors_b = evaluate_spectrum_claims(
        strategy_b, sampled_claims, num_advocates, parallel_claims=parallel_claims
    )
    show_evaluation_errors(errors_b, total_claims=total_claims)

    # Align by claim index (only claims that succeeded in both runs)
    indices_a = set(collectors_a.get("claim_indices", []))
    indices_b = set(collectors_b.get("claim_indices", []))
    common_indices = sorted(indices_a & indices_b)
    if not common_indices:
        logger.warning("No claims succeeded in both runs; cannot compare.")
        print("No claims succeeded in both runs; cannot compare.")
        return

    pos_a = {idx: i for i, idx in enumerate(collectors_a.get("claim_indices", []))}
    pos_b = {idx: i for i, idx in enumerate(collectors_b.get("claim_indices", []))}
    true_labels = [collectors_a["true_labels"][pos_a[idx]] for idx in common_indices]
    run_a_primaries = [collectors_a["predicted_results"][pos_a[idx]] for idx in common_indices]
    run_b_primaries = [collectors_b["predicted_results"][pos_b[idx]] for idx in common_indices]
    claims_subset = [sampled_claims.loc[idx, "Claim"] for idx in common_indices]

    true_mapped = [_spectrum_verdict_mapper_true(l) for l in true_labels]
    run_a_mapped = [_spectrum_verdict_mapper_predicted(p) for p in run_a_primaries]
    run_b_mapped = [_spectrum_verdict_mapper_predicted(p) for p in run_b_primaries]

    print("\n" + "=" * 60)
    print("COMPARISON: Mediator (a) no evidence vs (b) 3 advocate-supporting chunks")
    print("=" * 60)

    for level in (2, 5):
        level_name = "correct/incorrect (L2)" if level == 2 else "level-5 (L5)"
        true_c = [_cluster_verdict_for_metrics(l, level) for l in true_mapped]
        run_a_c = [_cluster_verdict_for_metrics(l, level) for l in run_a_mapped]
        run_b_c = [_cluster_verdict_for_metrics(l, level) for l in run_b_mapped]
        acc_a = accuracy_score(true_c, run_a_c)
        acc_b = accuracy_score(true_c, run_b_c)
        print(f"\n--- {level_name} ---")
        print(f"(a) Mediator no evidence:  {acc_a:.2%}")
        print(f"(b) Mediator 3 supporting chunks: {acc_b:.2%}")
        m_a = calculate_classification_metrics(true_c, run_a_c, verdict_mapper=None)
        m_b = calculate_classification_metrics(true_c, run_b_c, verdict_mapper=None)
        print("\n(a) Full report:")
        print(m_a)
        print("\n(b) Full report:")
        print(m_b)

    true_l2 = [_cluster_verdict_for_metrics(l, 2) for l in true_mapped]
    true_l5 = [_cluster_verdict_for_metrics(l, 5) for l in true_mapped]
    run_a_l2 = [_cluster_verdict_for_metrics(l, 2) for l in run_a_mapped]
    run_a_l5 = [_cluster_verdict_for_metrics(l, 5) for l in run_a_mapped]
    run_b_l2 = [_cluster_verdict_for_metrics(l, 2) for l in run_b_mapped]
    run_b_l5 = [_cluster_verdict_for_metrics(l, 5) for l in run_b_mapped]

    comparison_df = pd.DataFrame({
        "Claim": claims_subset,
        "True Label": true_labels,
        "Run_a_mediator_primary_no_evidence": run_a_primaries,
        "Run_b_mediator_primary_3_supporting_chunks": run_b_primaries,
        "True_L2": true_l2,
        "Run_a_L2": run_a_l2,
        "Run_b_L2": run_b_l2,
        "True_L5": true_l5,
        "Run_a_L5": run_a_l5,
        "Run_b_L5": run_b_l5,
    })
    out_path = save_results(
        comparison_df,
        base_path="experiments/results",
        prefix="spectrum_comparison_mediator_no_evidence_vs_3_supporting_chunks",
    )
    logger.info("Comparison CSV saved to: %s", out_path)
    print(f"\nComparison saved to: {out_path}")
    print(f"Compared {len(common_indices)} claims (successful in both runs).")


if __name__ == "__main__":
    main()
