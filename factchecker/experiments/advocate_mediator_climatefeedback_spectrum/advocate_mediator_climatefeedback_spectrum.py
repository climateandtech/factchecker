"""
Spectrum experiment: advocate-mediator with weighted verdict distribution
over the full Science Feedback claim review verdict scale.

Each retrieved evidence chunk is first classified (supports / refutes / nei)
by an EvidenceClassifierStep, then the advocate evaluates the claim using the
classified evidence and the full SF label options.
"""
import json
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from llama_index.core import Settings
from tqdm import tqdm

from factchecker.experiments.advocate_mediator_climatefeedback.advocate_mediator_climatefeedback import (
    setup_sources,
)
from factchecker.experiments.advocate_mediator_climatefeedback_spectrum.advocate_mediator_climatefeedback_spectrum_prompts import (
    EVIDENCE_CLASSIFIER_REASONS,
    SPECTRUM_ADVOCATE_SELECT_CHUNKS_INSTRUCTION,
    SPECTRUM_ADVOCATE_VERDICT_FORMAT,
    SPECTRUM_ADVOCATE_VERDICT_FORMAT_LABEL_AND_WEIGHTS,
    SPECTRUM_ADVOCATE_VERDICT_FORMAT_WEIGHTS_ONLY,
    SPECTRUM_CATEGORIES,
    SPECTRUM_LABEL_OPTIONS,
    SPECTRUM_MEDIATOR_USER_MESSAGE_SUFFIX,
    SPECTRUM_MEDIATOR_VERDICT_FORMAT_INSTRUCTION,
    spectrum_advocate_primer,
    spectrum_evidence_classifier_system_prompt,
    spectrum_evidence_classifier_user_prompt,
    spectrum_mediator_primer,
)
from factchecker.experiments.advocate_mediator_climatefeedback_spectrum.weighted_parser import (
    combine_advocate_weights_formula,
    parse_advocate_response_label_only_with_chunk_selection,
    parse_advocate_response_label_and_weights,
    parse_advocate_response_weights_only,
    parse_mediator_single_label,
    parse_weighted_response,
)
from factchecker.steps.advocate import ADVOCATE_OUTPUT_MODES
from factchecker.steps.evidence_classifier import EvidenceClassifierStep
from factchecker.strategies.advocate_mediator import AdvocateMediatorStrategy
from factchecker.utils.climatefeedback_utils import (
    map_verdict,
    sample_climatefeedback_claims,
)

from factchecker.prompts.advocate_prompts import chunk_stats_from_classified
from factchecker.utils.experiment_utils import (
    collect_evaluation_results,
    configure_logging,
    initialize_results_collectors,
    save_evaluation_errors,
    save_results,
    show_evaluation_errors,
    verify_environment,
)
from sklearn.metrics import accuracy_score

from factchecker.utils.metrics import calculate_classification_metrics

from factchecker.experiments.advocate_mediator_climatefeedback_spectrum.radar_chart import (
    generate_radar_charts,
)
from factchecker.experiments.advocate_mediator_climatefeedback_spectrum.weighted_metrics import (
    format_weighted_metrics,
)

logger = logging.getLogger(__name__)

EXPERIMENT_PARAMS = {
    "dataset_path": "datasets/Combined_Overview_Climate_Feedback_Claims.csv",
    "total_samples": 10,
    "correct_ratio": 0.3,
    "chunk_size": 150,
    "chunk_overlap": 20,
    "main_source_directory": "data/sources",
    "index_path": "data/indices/advocate_mediator_climatefeedback_spectrum",
    "sources_csv": "factchecker/experiments/advocate_mediator_climatefeedback/advocate_mediator_climatefeedback_sources.csv",
    "sources_output_folder": "data/sources",
    "sources_skip_existing": True,
    "sources_max_sources": 1,
    "sources_url_column": "url",
    "sources_output_filename_column": "output_filename",
    "sources_output_subfolder_column": "output_subfolder",
    "top_k": 8,
    "advocate_output_mode": "label_only",  # or "label_and_weights", "weights_only"
    "advocate_evidence_display": "full",  # or "relevant_only" — advocate sees only relevant_phrase per chunk (+ chunk_summary stats)
    "mediator_evidence_relevant_only": False,  # when mediator sees evidence, pass only relevant_phrase (+ stats line)
    "use_evidence_classifier": True,
    "ablation_compare_chunk_labelling": False,  # run with and without chunk labelling and save comparison data
    "compare_mediator_modes": False,  # report advocate_only vs formula vs LLM mediator (uses label_and_weights for formula)
    "run_all_comparisons": True,  # one run: ablation + mediator evidence (no vs 3 chunks) + three-way (advocate/formula/LLM)
    "comparison_runs": None,  # None = run all 6; or list of ints in [1,2,3,4,5,6] to run only those (e.g. [1,2,5,6] for relevant_only ablations only)
    "parallel_claims": 0,  # 0 or 1 = sequential; >1 = max workers for parallel claim verification (e.g. 4 to max GPU)
}


def _advocate_primary_verdict(advocate_verdicts: List[List[str]]) -> List[str]:
    """Per claim, return the consensus verdict across advocates (mode). Single advocate → that verdict."""
    if not advocate_verdicts or not advocate_verdicts[0]:
        return []
    n_claims = len(advocate_verdicts[0])
    out = []
    for j in range(n_claims):
        verdicts = [advocate_verdicts[i][j] for i in range(len(advocate_verdicts))]
        mode, _ = Counter(verdicts).most_common(1)[0]
        out.append(mode)
    return out


def _primary_verdict_from_final(final_verdict: str) -> str:
    """Compute primary verdict (argmax of weights) from final_verdict; or return as-is if not JSON weights."""
    if not final_verdict or not final_verdict.strip().startswith("{"):
        return final_verdict
    try:
        weights = json.loads(final_verdict)
        if isinstance(weights, dict) and weights:
            primary = max(weights, key=lambda k: weights[k])
            return primary
    except (json.JSONDecodeError, TypeError):
        pass
    return final_verdict


def setup_strategy(
    params: Optional[Dict[str, Any]] = None,
    downloaded_files: Optional[List[str]] = None,
) -> AdvocateMediatorStrategy:
    """Set up advocate-mediator strategy with spectrum (weighted) mediator."""
    verify_environment()
    p = {**EXPERIMENT_PARAMS, **(params or {})}
    Settings.chunk_size = p["chunk_size"]
    Settings.chunk_overlap = p["chunk_overlap"]
    main_source_directory = p["main_source_directory"]
    max_sources = p.get("sources_max_sources")

    if downloaded_files:
        indexer_options_list = [{"files": downloaded_files, "index_name": "ipcc"}]
    elif max_sources is not None and max_sources <= 1:
        indexer_options_list = [
            {"source_directory": os.path.join(main_source_directory, "ipcc"), "index_name": "ipcc"}
        ]
    else:
        indexer_options_list = [
            {"source_directory": os.path.join(main_source_directory, "ipcc"), "index_name": "ipcc"},
            {"source_directory": os.path.join(main_source_directory, "wmo"), "index_name": "wmo"},
            {"source_directory": os.path.join(main_source_directory, "nipcc"), "index_name": "nipcc"},
        ]

    indexer_overrides = {
        k: v
        for k, v in p.items()
        if k in ("chunk_size", "chunk_overlap", "embed_batch_size", "index_path", "embedding_type", "embedding_model", "show_progress")
    }
    if indexer_overrides:
        base_index_path = indexer_overrides.pop("index_path", None)
        for opts in indexer_options_list:
            opts.update(indexer_overrides)
            if base_index_path:
                opts["index_path"] = os.path.join(base_index_path, opts["index_name"])

    retriever_options_list = [{"top_k": p["top_k"]} for _ in indexer_options_list]

    use_evidence_classifier = p.get("use_evidence_classifier", True)
    evidence_classifier = None
    if use_evidence_classifier:
        evidence_classifier = EvidenceClassifierStep(
            options={
                "system_prompt": spectrum_evidence_classifier_system_prompt,
                "user_prompt_fn": spectrum_evidence_classifier_user_prompt,
                "include_reason": True,
                "allowed_reasons": EVIDENCE_CLASSIFIER_REASONS,
            }
        )

    mediator_mode = str(p.get("mediator_mode", "llm")).strip().lower()
    advocate_output_mode = str(p.get("advocate_output_mode", "label_only")).strip().lower()
    if advocate_output_mode not in ADVOCATE_OUTPUT_MODES:
        advocate_output_mode = "label_only"
    if mediator_mode == "llm_with_advocate_proof" and advocate_output_mode == "label_only":
        # Advocate must pick chunks that support its verdict; mediator will see only those
        verdict_format = SPECTRUM_ADVOCATE_VERDICT_FORMAT + " " + SPECTRUM_ADVOCATE_SELECT_CHUNKS_INSTRUCTION
        verdict_parser = parse_advocate_response_label_only_with_chunk_selection
    elif advocate_output_mode == "label_and_weights":
        verdict_format = SPECTRUM_ADVOCATE_VERDICT_FORMAT_LABEL_AND_WEIGHTS
        verdict_parser = parse_advocate_response_label_and_weights
    elif advocate_output_mode == "weights_only":
        verdict_format = SPECTRUM_ADVOCATE_VERDICT_FORMAT_WEIGHTS_ONLY
        verdict_parser = parse_advocate_response_weights_only
    else:
        verdict_format = SPECTRUM_ADVOCATE_VERDICT_FORMAT
        verdict_parser = None  # use default ((label)) parser
    advocate_options = {
        "system_prompt": spectrum_advocate_primer,
        "label_options": SPECTRUM_LABEL_OPTIONS,
        "evidence_classifier": evidence_classifier,
        "verdict_format": verdict_format,
        "verdict_parser": verdict_parser,
        "advocate_output_mode": advocate_output_mode,
        "advocate_evidence_display": str(p.get("advocate_evidence_display", "full")).strip().lower(),
    }
    if advocate_options["advocate_evidence_display"] not in ("full", "relevant_only"):
        advocate_options["advocate_evidence_display"] = "full"
    evidence_options = {}
    mediator_options = {
        "system_prompt": spectrum_mediator_primer,
        "mediator_mode": mediator_mode,
        "mediator_evidence_relevant_only": bool(p.get("mediator_evidence_relevant_only", False)),
    }
    if mediator_mode in ("llm", "llm_with_evidence", "llm_with_advocate_proof"):
        mediator_options["verdict_parser"] = parse_mediator_single_label
        mediator_options["verdict_format_instruction"] = SPECTRUM_MEDIATOR_VERDICT_FORMAT_INSTRUCTION
        mediator_options["user_message_suffix"] = SPECTRUM_MEDIATOR_USER_MESSAGE_SUFFIX
    else:
        mediator_options["verdict_parser"] = parse_weighted_response
    if mediator_mode == "formula":
        mediator_options["formula_fn"] = combine_advocate_weights_formula
    if mediator_mode == "llm_with_advocate_proof":
        mediator_options["mediator_proof_max_chunks"] = int(p.get("mediator_proof_max_chunks", 3))

    strategy = AdvocateMediatorStrategy(
        indexer_options_list,
        retriever_options_list,
        advocate_options,
        evidence_options,
        mediator_options,
    )
    logger.info("Spectrum strategy initialized successfully")
    return strategy


def _evaluate_one_claim(
    idx: Any,
    row: pd.Series,
    strategy: AdvocateMediatorStrategy,
) -> Tuple[Any, pd.Series, bool, Union[tuple, Dict[str, Any]]]:
    """Evaluate a single claim; return (idx, row, success, payload). Payload is 5-tuple on success, error dict on failure."""
    claim_text = row["Claim"]
    try:
        result = strategy.evaluate_claim(claim_text)
        return (idx, row, True, result)
    except Exception as e:
        logger.error("Error processing claim %s: %s", idx, e)
        return (
            idx,
            row,
            False,
            {
                "claim_index": idx,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "claim_preview": (claim_text[:80] + "...") if len(claim_text) > 80 else claim_text,
            },
        )


def evaluate_spectrum_claims(
    strategy: AdvocateMediatorStrategy,
    sampled_claims: pd.DataFrame,
    num_advocates: int,
    parallel_claims: int = 0,
):
    """Evaluate claims with spectrum (weighted) mediator; collect primary verdict, weighted_verdicts, and evidence metadata.
    If parallel_claims > 1, run claim verification in parallel with that many workers to utilize GPU."""
    collectors = initialize_results_collectors(num_advocates)
    collectors["weighted_verdicts"] = []
    collectors["advocate_evidence_metadata"] = [[] for _ in range(num_advocates)]
    collectors["chunk_stats"] = []  # per-claim stats: n_supports, n_refutes, n_nin, n_with/without relevant_phrase
    errors: List[Dict[str, Any]] = []

    items = list(sampled_claims.iterrows())
    max_workers = max(1, int(parallel_claims)) if parallel_claims else 0
    use_parallel = max_workers > 1

    if use_parallel:
        results_by_idx: Dict[Any, Tuple[Any, pd.Series, bool, Union[tuple, Dict[str, Any]]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_evaluate_one_claim, idx, row, strategy): idx
                for idx, row in items
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Evaluating claims (parallel)",
            ):
                idx, row, success, payload = future.result()
                results_by_idx[idx] = (idx, row, success, payload)
        # Reorder to match original sampled_claims order
        results = [results_by_idx[idx] for idx, _ in items]
    else:
        results = []
        for idx, row in tqdm(items, total=len(items), desc="Evaluating claims"):
            results.append(_evaluate_one_claim(idx, row, strategy))

    for idx, row, success, payload in results:
        claim_text = row["Claim"]
        true_label = row["Climate Feedback"]
        if not success:
            errors.append(payload)
            continue
        (
            final_verdict,
            verdicts,
            reasonings,
            evidence_metadata_per_advocate,
            advocate_weights_per_advocate,
        ) = payload
        primary_verdict = _primary_verdict_from_final(final_verdict)
        collectors = collect_evaluation_results(
            collectors,
            (true_label, primary_verdict, verdicts, reasonings),
            num_advocates=len(verdicts) if not collectors["advocate_evidences"] else None,
            claim_index=idx,
        )
        if final_verdict and not final_verdict.strip().startswith("{"):
            one_hot = {c: 1.0 if c == primary_verdict else 0.0 for c in SPECTRUM_CATEGORIES}
            collectors["weighted_verdicts"].append(json.dumps(one_hot))
        else:
            collectors["weighted_verdicts"].append(final_verdict)
        for i in range(num_advocates):
            meta = (
                evidence_metadata_per_advocate[i]
                if evidence_metadata_per_advocate and i < len(evidence_metadata_per_advocate)
                else None
            )
            collectors["advocate_evidence_metadata"][i].append(meta if meta else [])
            w = (
                advocate_weights_per_advocate[i]
                if advocate_weights_per_advocate and i < len(advocate_weights_per_advocate)
                else None
            )
            collectors["advocate_weighted_verdicts"][i].append(
                json.dumps(w) if isinstance(w, dict) else w
            )
        # Per-claim chunk statistics (from first advocate when classifier is used)
        meta0 = evidence_metadata_per_advocate[0] if evidence_metadata_per_advocate else []
        collectors["chunk_stats"].append(chunk_stats_from_classified(meta0))

    return collectors, errors


def _spectrum_verdict_mapper_true(label: str) -> str:
    """Map true label to Science Feedback claim verdict scale (level-7 granular)."""
    return map_verdict(label, level=7)


def _spectrum_verdict_mapper_predicted(label: str) -> str:
    """Normalize predicted (primary) verdict for metrics: lowercase only; no category mapping so the full spectrum of model outputs is preserved for analysis."""
    if not label:
        return "unknown"
    s = label.strip().lower().replace(" ", "_")
    if s == "not_enough_info":
        return "not_enough_information"
    return s


def _cluster_verdict_for_metrics(label: str, level: int) -> str:
    """Map a single verdict (our format, e.g. mostly_accurate) to clustered label at given level (2 or 5) via map_verdict."""
    if not label:
        return "unknown"
    # map_verdict expects space-separated keys
    normalized_input = (label or "").strip().lower().replace("_", " ")
    return map_verdict(normalized_input, level=level)


def _formula_primary_from_collectors(collectors: Dict[str, Any]) -> Optional[List[str]]:
    """
    Compute formula mediator primary verdict per claim from advocate_weighted_verdicts.
    Returns list of primary verdicts (one per claim) or None if weights are missing/invalid.
    """
    awv = collectors.get("advocate_weighted_verdicts")
    if not awv or not awv[0]:
        return None
    n_claims = len(awv[0])
    out = []
    for j in range(n_claims):
        weight_strs = [awv[i][j] if i < len(awv) else None for i in range(len(awv))]
        weight_dicts = []
        for s in weight_strs:
            if not s or not (isinstance(s, str) and s.strip().startswith("{")):
                continue
            try:
                d = json.loads(s)
                if isinstance(d, dict) and d:
                    weight_dicts.append(d)
            except (json.JSONDecodeError, TypeError):
                continue
        if not weight_dicts:
            out.append(None)
            continue
        primary, _ = combine_advocate_weights_formula(weight_dicts)
        out.append(primary)
    if all(p is None for p in out):
        return None
    return out


def _build_spectrum_results_df(collectors: Dict[str, Any], sampled_claims: pd.DataFrame) -> pd.DataFrame:
    """Build the full results DataFrame from collectors and sampled_claims (used in main and ablation)."""
    claim_indices = collectors.get("claim_indices")
    if claim_indices is not None and len(claim_indices) > 0:
        claims_subset = sampled_claims.loc[claim_indices].reset_index(drop=True)
    else:
        claims_subset = sampled_claims

    advocate_primary = _advocate_primary_verdict(collectors["advocate_verdicts"])
    mediator_predicted = collectors["predicted_results"]
    results_dict = {
        "Claim": claims_subset["Claim"].values,
        "True Label": collectors["true_labels"],
        "Advocate Primary Verdict": advocate_primary,
        "Mediator Predicted Verdict": mediator_predicted,
        "Predicted Verdict": mediator_predicted,
        "True Label (SF verdict)": [_spectrum_verdict_mapper_true(l) for l in collectors["true_labels"]],
        "Weighted verdict": collectors["weighted_verdicts"],
        "Mediator Reasoning": collectors["mediator_reasonings"],
    }
    chunk_stats_list = collectors.get("chunk_stats")
    if chunk_stats_list and len(chunk_stats_list) == len(collectors["true_labels"]):
        results_dict["Chunk stats"] = [
            " | ".join(f"{k}={v}" for k, v in s.items()) if isinstance(s, dict) else str(s)
            for s in chunk_stats_list
        ]
    meta_list = collectors.get("advocate_evidence_metadata")
    for i in range(len(collectors["advocate_evidences"])):
        if meta_list and i < len(meta_list):
            evidence_seen = []
            for meta in meta_list[i]:
                parts = []
                for m in meta:
                    stance = m.get("stance", "")
                    phrase = (m.get("relevant_phrase") or "").strip()
                    reason = (m.get("reason") or "").strip()
                    text = (m.get("text") or "").strip()
                    block = f"[stance: {stance}]"
                    if phrase:
                        block += f' relevant_phrase: "{phrase}"'
                    if reason:
                        block += f" reason: {reason}"
                    block += f"\n{text}"
                    parts.append(block)
                evidence_seen.append("\n---\n".join(parts) if parts else "")
            results_dict[f"Advocate {i+1} Evidence"] = evidence_seen
        else:
            results_dict[f"Advocate {i+1} Evidence"] = collectors["advocate_evidences"][i]
        results_dict[f"Advocate {i+1} Verdict"] = collectors["advocate_verdicts"][i]
        results_dict[f"Advocate {i+1} Reasoning"] = collectors["advocate_reasonings"][i]
        if meta_list and i < len(meta_list):
            chunk_ids_col = []
            relevant_phrases_col = []
            reasons_col = []
            for meta in meta_list[i]:
                chunk_ids_col.append(",".join(str(m.get("chunk_id", "")) for m in meta))
                relevant_phrases_col.append(" | ".join(str(m.get("relevant_phrase", "")) for m in meta))
                reasons_col.append(" | ".join(str(m.get("reason", "")) for m in meta))
            results_dict[f"Advocate {i+1} Chunk IDs"] = chunk_ids_col
            results_dict[f"Advocate {i+1} Relevant phrases"] = relevant_phrases_col
            results_dict[f"Advocate {i+1} Chunk reasons"] = reasons_col
        awv = collectors.get("advocate_weighted_verdicts")
        if awv and i < len(awv):
            results_dict[f"Advocate {i+1} Weighted verdict"] = awv[i]

    return pd.DataFrame(results_dict)


def _run_all_comparisons(params, sampled_claims, downloaded_files, total_claims: int):
    """
    Run comparisons on the same claims. By default runs 1--6; use comparison_runs to run only a subset (e.g. [1,2,5,6]).
    - Run 1: with chunk labelling, mediator no evidence (llm), advocate full chunks
    - Run 2: with chunk labelling, mediator 3 advocate-selected chunks (full snippets)
    - Run 3: without chunk labelling, mediator no evidence (llm)
    - Run 4: with chunk labelling, label_and_weights, mediator llm (three-way)
    - Run 5: like Run 1 but advocate_evidence_display=relevant_only
    - Run 6: like Run 2 but mediator_evidence_relevant_only=True

    Outputs only the comparison types whose required runs were executed (e.g. [1,2,5,6] -> mediator evidence, advocate display, mediator relevant_only).
    """
    parallel_claims = params.get("parallel_claims", 0) or 0
    which = params.get("comparison_runs")
    if which is None:
        run_set = {1, 2, 3, 4, 5, 6}
    else:
        run_set = {int(x) for x in which if 1 <= int(x) <= 6}

    c1 = c2 = c3 = c4 = c5 = c6 = None
    n_adv = None

    if 1 in run_set:
        p1 = {**params, "use_evidence_classifier": True, "advocate_output_mode": "label_only", "mediator_mode": "llm"}
        s1 = setup_strategy(params=p1, downloaded_files=downloaded_files)
        n_adv = len(s1.advocate_steps)
        c1, e1 = evaluate_spectrum_claims(s1, sampled_claims, n_adv, parallel_claims=parallel_claims)
        show_evaluation_errors(e1, total_claims=total_claims)
    if 2 in run_set:
        p2 = {**params, "use_evidence_classifier": True, "advocate_output_mode": "label_only", "mediator_mode": "llm_with_advocate_proof", "mediator_proof_max_chunks": 3}
        s2 = setup_strategy(params=p2, downloaded_files=downloaded_files)
        if n_adv is None:
            n_adv = len(s2.advocate_steps)
        c2, e2 = evaluate_spectrum_claims(s2, sampled_claims, n_adv, parallel_claims=parallel_claims)
        show_evaluation_errors(e2, total_claims=total_claims)
    if 3 in run_set:
        p3 = {**params, "use_evidence_classifier": False, "advocate_output_mode": "label_only", "mediator_mode": "llm"}
        s3 = setup_strategy(params=p3, downloaded_files=downloaded_files)
        if n_adv is None:
            n_adv = len(s3.advocate_steps)
        c3, e3 = evaluate_spectrum_claims(s3, sampled_claims, n_adv, parallel_claims=parallel_claims)
        show_evaluation_errors(e3, total_claims=total_claims)
    if 4 in run_set:
        p4 = {**params, "use_evidence_classifier": True, "advocate_output_mode": "label_and_weights", "mediator_mode": "llm"}
        s4 = setup_strategy(params=p4, downloaded_files=downloaded_files)
        if n_adv is None:
            n_adv = len(s4.advocate_steps)
        c4, e4 = evaluate_spectrum_claims(s4, sampled_claims, n_adv, parallel_claims=parallel_claims)
        show_evaluation_errors(e4, total_claims=total_claims)
    if 5 in run_set:
        p5 = {**params, "use_evidence_classifier": True, "advocate_output_mode": "label_only", "mediator_mode": "llm", "advocate_evidence_display": "relevant_only"}
        s5 = setup_strategy(params=p5, downloaded_files=downloaded_files)
        if n_adv is None:
            n_adv = len(s5.advocate_steps)
        c5, e5 = evaluate_spectrum_claims(s5, sampled_claims, n_adv, parallel_claims=parallel_claims)
        show_evaluation_errors(e5, total_claims=total_claims)
    if 6 in run_set:
        p6 = {**params, "use_evidence_classifier": True, "advocate_output_mode": "label_only", "mediator_mode": "llm_with_advocate_proof", "mediator_proof_max_chunks": 3, "mediator_evidence_relevant_only": True}
        s6 = setup_strategy(params=p6, downloaded_files=downloaded_files)
        if n_adv is None:
            n_adv = len(s6.advocate_steps)
        c6, e6 = evaluate_spectrum_claims(s6, sampled_claims, n_adv, parallel_claims=parallel_claims)
        show_evaluation_errors(e6, total_claims=total_claims)

    collectors = [c1, c2, c3, c4, c5, c6]
    sets_to_intersect = [set(c.get("claim_indices") or []) for c in collectors if c]
    common = sorted(set.intersection(*sets_to_intersect)) if sets_to_intersect else []
    if not common:
        logger.warning("No claims succeeded in all executed runs; cannot compare.")
        print("No claims succeeded in all executed runs; cannot compare.")
        return

    # True labels: use first available collector that has them (e.g. c1 or c4)
    ref_collector = c1 or c2 or c4 or c5 or c3 or c6
    def pos(c, ind): return c["claim_indices"].index(ind)
    true_labels = [ref_collector["true_labels"][pos(ref_collector, i)] for i in common]
    true_mapped = [_spectrum_verdict_mapper_true(l) for l in true_labels]

    def _section(title: str) -> None:
        print(f"\n{'='*60}\n{title}\n{'='*60}")

    paths = []

    # ----- 1) Ablation: with (Run 1) vs without (Run 3) chunk labelling -----
    if c1 is not None and c3 is not None:
        _section("1) Ablation: with vs without chunk labelling")
        adv1 = _advocate_primary_verdict(c1["advocate_verdicts"])
        med1 = [c1["predicted_results"][pos(c1, i)] for i in common]
        adv3 = _advocate_primary_verdict(c3["advocate_verdicts"])
        med3 = [c3["predicted_results"][pos(c3, i)] for i in common]
        a1_m = [_spectrum_verdict_mapper_predicted(l) for l in adv1]
        m1_m = [_spectrum_verdict_mapper_predicted(l) for l in med1]
        a3_m = [_spectrum_verdict_mapper_predicted(l) for l in adv3]
        m3_m = [_spectrum_verdict_mapper_predicted(l) for l in med3]
        for level in (2, 5):
            name = "L2 (correct/incorrect)" if level == 2 else "L5"
            tc = [_cluster_verdict_for_metrics(l, level) for l in true_mapped]
            print(f"\n--- {name} ---")
            print("With labelling:   Advocate %.2f%%, Mediator %.2f%%" % (accuracy_score(tc, [_cluster_verdict_for_metrics(l, level) for l in a1_m]) * 100, accuracy_score(tc, [_cluster_verdict_for_metrics(l, level) for l in m1_m]) * 100))
            print("Without labelling: Advocate %.2f%%, Mediator %.2f%%" % (accuracy_score(tc, [_cluster_verdict_for_metrics(l, level) for l in a3_m]) * 100, accuracy_score(tc, [_cluster_verdict_for_metrics(l, level) for l in m3_m]) * 100))
        df_ablation = pd.DataFrame({
            "Claim": [sampled_claims.loc[i, "Claim"] for i in common],
            "True Label": true_labels,
            "With_advocate": [adv1[pos(c1, i)] for i in common],
            "With_mediator": [med1[pos(c1, i)] for i in common],
            "Without_advocate": [adv3[pos(c3, i)] for i in common],
            "Without_mediator": [med3[pos(c3, i)] for i in common],
        })
        path_ablation = save_results(df_ablation, base_path="experiments/results", prefix="spectrum_all_ablation")
        paths.append(("Ablation", path_ablation))
        print(f"\nAblation CSV: {path_ablation}")

    # ----- 2) Mediator evidence: no evidence (Run 1) vs 3 chunks (Run 2) -----
    if c1 is not None and c2 is not None:
        _section("2) Mediator: no evidence vs 3 advocate-selected chunks")
        med1 = [c1["predicted_results"][pos(c1, i)] for i in common]
        med2 = [c2["predicted_results"][pos(c2, i)] for i in common]
        m1_m = [_spectrum_verdict_mapper_predicted(l) for l in med1]
        m2_m = [_spectrum_verdict_mapper_predicted(l) for l in med2]
        for level in (2, 5):
            name = "L2" if level == 2 else "L5"
            tc = [_cluster_verdict_for_metrics(l, level) for l in true_mapped]
            print(f"\n--- {name} --- No evidence: %.2f%% | 3 chunks: %.2f%%" % (accuracy_score(tc, [_cluster_verdict_for_metrics(l, level) for l in m1_m]) * 100, accuracy_score(tc, [_cluster_verdict_for_metrics(l, level) for l in m2_m]) * 100))
        df_med_ev = pd.DataFrame({
            "Claim": [sampled_claims.loc[i, "Claim"] for i in common],
            "True Label": true_labels,
            "Mediator_no_evidence": med1,
            "Mediator_3_chunks": med2,
        })
        path_med_ev = save_results(df_med_ev, base_path="experiments/results", prefix="spectrum_all_mediator_evidence")
        paths.append(("Mediator evidence", path_med_ev))
        print(f"\nMediator evidence CSV: {path_med_ev}")

    # ----- 3) Three-way: advocate_only, formula, LLM (from Run 4) -----
    if c4 is not None:
        _section("3) Three-way: Advocate-only vs Formula vs LLM mediator")
        adv4 = _advocate_primary_verdict(c4["advocate_verdicts"])
        med4 = [c4["predicted_results"][pos(c4, i)] for i in common]
        formula_primary = _formula_primary_from_collectors(c4)
        if formula_primary is not None:
            formula_list = [formula_primary[pos(c4, i)] for i in common]
            a4_m = [_spectrum_verdict_mapper_predicted(l) for l in adv4]
            f_m = [_spectrum_verdict_mapper_predicted(l) for l in formula_list]
            m4_m = [_spectrum_verdict_mapper_predicted(l) for l in med4]
            for level in (2, 5):
                name = "L2" if level == 2 else "L5"
                tc = [_cluster_verdict_for_metrics(l, level) for l in true_mapped]
                print(f"\n--- {name} --- Advocate-only: %.2f%% | Formula: %.2f%% | LLM: %.2f%%" % (
                    accuracy_score(tc, [_cluster_verdict_for_metrics(l, level) for l in a4_m]) * 100,
                    accuracy_score(tc, [_cluster_verdict_for_metrics(l, level) for l in f_m]) * 100,
                    accuracy_score(tc, [_cluster_verdict_for_metrics(l, level) for l in m4_m]) * 100,
                ))
            df_three = pd.DataFrame({
                "Claim": [sampled_claims.loc[i, "Claim"] for i in common],
                "True Label": true_labels,
                "Advocate_only": [adv4[pos(c4, i)] for i in common],
                "Formula": formula_list,
                "LLM_mediator": med4,
            })
            path_three = save_results(df_three, base_path="experiments/results", prefix="spectrum_all_threeway")
            paths.append(("Three-way", path_three))
            print(f"\nThree-way CSV: {path_three}")
        else:
            print("(Formula not available: no advocate weights in Run 4.)")

    # ----- 4) Advocate evidence: full chunks vs relevant_only (Run 1 vs Run 5) -----
    if c1 is not None and c5 is not None:
        _section("4) Advocate: full chunks vs relevant parts only (+ chunk stats)")
        adv1 = _advocate_primary_verdict(c1["advocate_verdicts"])
        med1 = [c1["predicted_results"][pos(c1, i)] for i in common]
        adv5 = _advocate_primary_verdict(c5["advocate_verdicts"])
        med5 = [c5["predicted_results"][pos(c5, i)] for i in common]
        a1_m = [_spectrum_verdict_mapper_predicted(l) for l in adv1]
        m1_m = [_spectrum_verdict_mapper_predicted(l) for l in med1]
        a5_m = [_spectrum_verdict_mapper_predicted(l) for l in adv5]
        m5_m = [_spectrum_verdict_mapper_predicted(l) for l in med5]
        for level in (2, 5):
            name = "L2" if level == 2 else "L5"
            tc = [_cluster_verdict_for_metrics(l, level) for l in true_mapped]
            print(f"\n--- {name} --- Full: Advocate %.2f%% / Mediator %.2f%% | Relevant-only: Advocate %.2f%% / Mediator %.2f%%" % (
                accuracy_score(tc, [_cluster_verdict_for_metrics(l, level) for l in a1_m]) * 100,
                accuracy_score(tc, [_cluster_verdict_for_metrics(l, level) for l in m1_m]) * 100,
                accuracy_score(tc, [_cluster_verdict_for_metrics(l, level) for l in a5_m]) * 100,
                accuracy_score(tc, [_cluster_verdict_for_metrics(l, level) for l in m5_m]) * 100,
            ))
        df_adv_ev = pd.DataFrame({
            "Claim": [sampled_claims.loc[i, "Claim"] for i in common],
            "True Label": true_labels,
            "Full_advocate": [adv1[pos(c1, i)] for i in common],
            "Full_mediator": [med1[pos(c1, i)] for i in common],
            "Relevant_only_advocate": [adv5[pos(c5, i)] for i in common],
            "Relevant_only_mediator": [med5[pos(c5, i)] for i in common],
        })
        path_adv_ev = save_results(df_adv_ev, base_path="experiments/results", prefix="spectrum_all_advocate_evidence_display")
        paths.append(("Advocate display", path_adv_ev))
        print(f"\nAdvocate evidence display CSV: {path_adv_ev}")

    # ----- 5) Mediator evidence: full snippets vs relevant_only (Run 2 vs Run 6) -----
    if c2 is not None and c6 is not None:
        _section("5) Mediator: full snippets vs relevant parts only (+ stats)")
        med2 = [c2["predicted_results"][pos(c2, i)] for i in common]
        med6 = [c6["predicted_results"][pos(c6, i)] for i in common]
        m2_m = [_spectrum_verdict_mapper_predicted(l) for l in med2]
        m6_m = [_spectrum_verdict_mapper_predicted(l) for l in med6]
        for level in (2, 5):
            name = "L2" if level == 2 else "L5"
            tc = [_cluster_verdict_for_metrics(l, level) for l in true_mapped]
            print(f"\n--- {name} --- Full snippets: %.2f%% | Relevant-only: %.2f%%" % (
                accuracy_score(tc, [_cluster_verdict_for_metrics(l, level) for l in m2_m]) * 100,
                accuracy_score(tc, [_cluster_verdict_for_metrics(l, level) for l in m6_m]) * 100,
            ))
        df_med_rel = pd.DataFrame({
            "Claim": [sampled_claims.loc[i, "Claim"] for i in common],
            "True Label": true_labels,
            "Mediator_full_snippets": med2,
            "Mediator_relevant_only": med6,
        })
        path_med_rel = save_results(df_med_rel, base_path="experiments/results", prefix="spectrum_all_mediator_relevant_only")
        paths.append(("Mediator relevant-only", path_med_rel))
        print(f"\nMediator relevant-only CSV: {path_med_rel}")

    summary = f"Comparisons: {len(common)} claims. " + " | ".join(f"{n}: {p}" for n, p in paths)
    logger.info(summary)
    print(f"\n{summary}")


def _run_ablation_chunk_labelling(params, sampled_claims, downloaded_files, total_claims: int):
    """Run experiment twice (with and without evidence classifier) on same claims; compare normalized metrics and save comparison CSV."""
    num_advocates = 1  # setup_strategy will set this
    # Run 1: with chunk labelling
    params_with = {**params, "use_evidence_classifier": True}
    strategy_with = setup_strategy(params=params_with, downloaded_files=downloaded_files)
    num_advocates = len(strategy_with.advocate_steps)
    parallel_claims = params.get("parallel_claims", 0) or 0
    collectors1, errors1 = evaluate_spectrum_claims(
        strategy_with, sampled_claims, num_advocates, parallel_claims=parallel_claims
    )
    show_evaluation_errors(errors1, total_claims=total_claims)
    # Run 2: without chunk labelling
    params_without = {**params, "use_evidence_classifier": False}
    strategy_without = setup_strategy(params=params_without, downloaded_files=downloaded_files)
    collectors2, errors2 = evaluate_spectrum_claims(
        strategy_without, sampled_claims, num_advocates, parallel_claims=parallel_claims
    )
    show_evaluation_errors(errors2, total_claims=total_claims)

    common_indices = sorted(
        set(collectors1.get("claim_indices") or []) & set(collectors2.get("claim_indices") or [])
    )
    if not common_indices:
        logger.warning("Ablation: no claims succeeded in both runs; cannot compare.")
        print("Ablation: no claims succeeded in both runs; cannot compare.")
        return

    # Align by claim index
    pos1_list = [collectors1["claim_indices"].index(idx) for idx in common_indices]
    pos2_list = [collectors2["claim_indices"].index(idx) for idx in common_indices]
    true_aligned = [collectors1["true_labels"][p] for p in pos1_list]
    advocate_primary_1 = _advocate_primary_verdict(collectors1["advocate_verdicts"])
    advocate_primary_2 = _advocate_primary_verdict(collectors2["advocate_verdicts"])
    with_advocate = [advocate_primary_1[p] for p in pos1_list]
    with_mediator = [collectors1["predicted_results"][p] for p in pos1_list]
    without_advocate = [advocate_primary_2[p] for p in pos2_list]
    without_mediator = [collectors2["predicted_results"][p] for p in pos2_list]

    true_mapped = [_spectrum_verdict_mapper_true(l) for l in true_aligned]
    with_adv_mapped = [_spectrum_verdict_mapper_predicted(l) for l in with_advocate]
    with_med_mapped = [_spectrum_verdict_mapper_predicted(l) for l in with_mediator]
    without_adv_mapped = [_spectrum_verdict_mapper_predicted(l) for l in without_advocate]
    without_med_mapped = [_spectrum_verdict_mapper_predicted(l) for l in without_mediator]

    # Normalized (level 2 and 5) metrics for comparison
    comparison_lines = [
        f"Ablation: compared {len(common_indices)} claims (successful in both runs).",
        "",
    ]
    for level in (2, 5):
        level_name = "correct/incorrect (L2)" if level == 2 else "level-5 (L5)"
        true_clustered = [_cluster_verdict_for_metrics(l, level) for l in true_mapped]
        with_adv_cluster = [_cluster_verdict_for_metrics(l, level) for l in with_adv_mapped]
        with_med_cluster = [_cluster_verdict_for_metrics(l, level) for l in with_med_mapped]
        without_adv_cluster = [_cluster_verdict_for_metrics(l, level) for l in without_adv_mapped]
        without_med_cluster = [_cluster_verdict_for_metrics(l, level) for l in without_med_mapped]
        acc_with_adv = accuracy_score(true_clustered, with_adv_cluster)
        acc_with_med = accuracy_score(true_clustered, with_med_cluster)
        acc_without_adv = accuracy_score(true_clustered, without_adv_cluster)
        acc_without_med = accuracy_score(true_clustered, without_med_cluster)
        comparison_lines.append(f"--- Normalized {level_name} ---")
        comparison_lines.append("With chunk labelling: Advocate accuracy=%.2f%%, Mediator accuracy=%.2f%%" % (acc_with_adv * 100, acc_with_med * 100))
        comparison_lines.append("Without chunk labelling: Advocate accuracy=%.2f%%, Mediator accuracy=%.2f%%" % (acc_without_adv * 100, acc_without_med * 100))
        comparison_lines.append("")
    comparison_str = "\n".join(comparison_lines)
    logger.info(comparison_str)
    print(comparison_str)

    # Full analysis and metrics for both runs (aligned on common_indices)
    weighted_with = [collectors1["weighted_verdicts"][p] for p in pos1_list]
    weighted_without = [collectors2["weighted_verdicts"][p] for p in pos2_list]

    def _print_section(title: str) -> None:
        print(f"\n{'='*60}\n{title}\n{'='*60}")

    # --- With chunk labelling ---
    _print_section("With chunk labelling — full metrics")
    print("\n--- Advocate primary vs true (14-way) ---")
    print(calculate_classification_metrics(true_mapped, with_adv_mapped, verdict_mapper=None))
    print("\n--- Mediator predicted vs true (14-way) ---")
    print(calculate_classification_metrics(true_mapped, with_med_mapped, verdict_mapper=None))
    for level in (2, 5):
        level_name = "correct/incorrect (L2)" if level == 2 else "level-5 (L5)"
        true_c = [_cluster_verdict_for_metrics(l, level) for l in true_mapped]
        with_adv_c = [_cluster_verdict_for_metrics(l, level) for l in with_adv_mapped]
        with_med_c = [_cluster_verdict_for_metrics(l, level) for l in with_med_mapped]
        print(f"\n--- Advocate vs true (normalized {level_name}) ---")
        print(calculate_classification_metrics(true_c, with_adv_c, verdict_mapper=None))
        print(f"\n--- Mediator vs true (normalized {level_name}) ---")
        print(calculate_classification_metrics(true_c, with_med_c, verdict_mapper=None))
    print(format_weighted_metrics(true_mapped, weighted_with))

    # --- Without chunk labelling ---
    _print_section("Without chunk labelling — full metrics")
    print("\n--- Advocate primary vs true (14-way) ---")
    print(calculate_classification_metrics(true_mapped, without_adv_mapped, verdict_mapper=None))
    print("\n--- Mediator predicted vs true (14-way) ---")
    print(calculate_classification_metrics(true_mapped, without_med_mapped, verdict_mapper=None))
    for level in (2, 5):
        level_name = "correct/incorrect (L2)" if level == 2 else "level-5 (L5)"
        true_c = [_cluster_verdict_for_metrics(l, level) for l in true_mapped]
        without_adv_c = [_cluster_verdict_for_metrics(l, level) for l in without_adv_mapped]
        without_med_c = [_cluster_verdict_for_metrics(l, level) for l in without_med_mapped]
        print(f"\n--- Advocate vs true (normalized {level_name}) ---")
        print(calculate_classification_metrics(true_c, without_adv_c, verdict_mapper=None))
        print(f"\n--- Mediator vs true (normalized {level_name}) ---")
        print(calculate_classification_metrics(true_c, without_med_c, verdict_mapper=None))
    print(format_weighted_metrics(true_mapped, weighted_without))

    # Build comparison CSV
    claims_subset = sampled_claims.loc[common_indices].reset_index(drop=True)
    comparison_df = pd.DataFrame({
        "claim_index": common_indices,
        "Claim": claims_subset["Claim"].values,
        "True Label": true_aligned,
        "With labelling Advocate Primary": with_advocate,
        "With labelling Mediator Primary": with_mediator,
        "Without labelling Advocate Primary": without_advocate,
        "Without labelling Mediator Primary": without_mediator,
        "True (mapped)": true_mapped,
        "With Advocate (mapped)": with_adv_mapped,
        "With Mediator (mapped)": with_med_mapped,
        "Without Advocate (mapped)": without_adv_mapped,
        "Without Mediator (mapped)": without_med_mapped,
    })
    comparison_path = save_results(
        comparison_df,
        base_path="experiments/results",
        prefix="spectrum_ablation_chunk_labelling",
    )
    logger.info("Ablation comparison CSV saved to: %s", comparison_path)
    print(f"\nComparison saved to: {comparison_path}")

    # Save full results for each run so we can compare all columns (evidence, reasoning, etc.)
    df_with = _build_spectrum_results_df(collectors1, sampled_claims)
    df_without = _build_spectrum_results_df(collectors2, sampled_claims)
    path_with = save_results(
        df_with,
        base_path="experiments/results",
        prefix="spectrum_ablation_with_labelling",
    )
    path_without = save_results(
        df_without,
        base_path="experiments/results",
        prefix="spectrum_ablation_without_labelling",
    )
    logger.info("Ablation full results: with labelling=%s, without labelling=%s", path_with, path_without)
    print(f"Full results: with labelling={path_with}, without labelling={path_without}")

    # Radar charts for both runs (subset aligned to common_indices)
    radar_dir = "experiments/results/spectrum_radar"
    radar_with = generate_radar_charts(
        weighted_with,
        output_dir=f"{radar_dir}/ablation_with_labelling",
        per_claim=False,
        summary=True,
    )
    radar_without = generate_radar_charts(
        weighted_without,
        output_dir=f"{radar_dir}/ablation_without_labelling",
        per_claim=False,
        summary=True,
    )
    if radar_with or radar_without:
        logger.info("Ablation radar charts: with=%s, without=%s", radar_with, radar_without)
        print(f"Ablation radar: with_labelling={radar_with}, without_labelling={radar_without}")

    n1 = len(collectors1.get("true_labels") or [])
    n2 = len(collectors2.get("true_labels") or [])
    summary = (
        f"Ablation summary: with labelling {n1}/{total_claims} claims ok ({len(errors1)} errors); "
        f"without labelling {n2}/{total_claims} claims ok ({len(errors2)} errors); "
        f"compared on {len(common_indices)} claims."
    )
    logger.info(summary)
    print(f"\n{summary}")


def main(experiment_options: Optional[Dict[str, Any]] = None):
    """Run the Spectrum (weighted verdicts) advocate-mediator Climate Feedback experiment."""
    configure_logging()
    params = {**EXPERIMENT_PARAMS, **(experiment_options or {})}
    verify_environment()
    downloaded_files = setup_sources(params=params)
    Settings.chunk_size = params.get("chunk_size", 150)
    Settings.chunk_overlap = params.get("chunk_overlap", 20)

    logger.info("Loading and sampling claims...")
    sampled_claims = sample_climatefeedback_claims(
        csv_path=params["dataset_path"],
        total_samples=params["total_samples"],
        correct_ratio=params["correct_ratio"],
    )
    total_claims = len(sampled_claims)

    # All comparisons in one go: ablation + mediator evidence + three-way
    if params.get("run_all_comparisons"):
        _run_all_comparisons(params, sampled_claims, downloaded_files, total_claims)
        return

    # Ablation: run with and without chunk labelling on same claims, then compare
    if params.get("ablation_compare_chunk_labelling"):
        _run_ablation_chunk_labelling(params, sampled_claims, downloaded_files, total_claims)
        return

    # Three-way mediator comparison: advocate_only vs formula vs LLM (needs advocate weights)
    if params.get("compare_mediator_modes"):
        params = {**params, "advocate_output_mode": "label_and_weights"}

    strategy = setup_strategy(params=params, downloaded_files=downloaded_files)
    num_advocates = len(strategy.advocate_steps)
    parallel_claims = params.get("parallel_claims", 0) or 0
    collectors, errors = evaluate_spectrum_claims(
        strategy, sampled_claims, num_advocates, parallel_claims=parallel_claims
    )
    show_evaluation_errors(errors, total_claims=total_claims)

    total_claims = len(sampled_claims)
    n_ok = len(collectors["true_labels"])
    n_err = len(errors)
    errors_file = None
    if errors:
        errors_file = save_evaluation_errors(
            errors, base_path="experiments/results", prefix="spectrum_evaluation_errors"
        )
        if errors_file:
            logger.info("Evaluation errors saved to: %s", errors_file)

    if n_ok == 0:
        logger.warning("No claims were successfully evaluated; skipping results and metrics.")
        print(f"Run summary: 0/{total_claims} claims evaluated successfully. {n_err} errors.")
        return

    results_df = _build_spectrum_results_df(collectors, sampled_claims)
    advocate_primary = _advocate_primary_verdict(collectors["advocate_verdicts"])
    mediator_predicted = collectors["predicted_results"]
    results_file = save_results(results_df, base_path="experiments/results", prefix="spectrum_claims_results")
    logger.info("Results saved to: %s", results_file)

    # Radar chart(s) for weighted verdict distribution
    radar_paths = generate_radar_charts(
        collectors["weighted_verdicts"],
        output_dir="experiments/results/spectrum_radar",
        per_claim=True,
        summary=True,
    )
    if radar_paths:
        logger.info("Radar charts saved: %s", radar_paths)

    # Metrics on Science Feedback claim verdict scale (granular, level-7)
    true_mapped = [_spectrum_verdict_mapper_true(l) for l in collectors["true_labels"]]
    advocate_primary_mapped = [_spectrum_verdict_mapper_predicted(p) for p in advocate_primary]
    mediator_pred_mapped = [_spectrum_verdict_mapper_predicted(p) for p in mediator_predicted]

    # Three-way mediator comparison: advocate_only vs formula vs LLM (when compare_mediator_modes and we have weights)
    if params.get("compare_mediator_modes"):
        formula_primary = _formula_primary_from_collectors(collectors)
        if formula_primary is not None:
            formula_mapped = [_spectrum_verdict_mapper_predicted(p) for p in formula_primary]
            print("\n" + "=" * 60)
            print("Mediator comparison: Advocate-only vs Formula vs LLM mediator")
            print("=" * 60)
            for name, pred_mapped in [
                ("Advocate-only (normalized)", advocate_primary_mapped),
                ("Formula (avg weights + argmax)", formula_mapped),
                ("LLM mediator", mediator_pred_mapped),
            ]:
                print(f"\n--- {name} vs true (14-way) ---")
                print(calculate_classification_metrics(true_mapped, pred_mapped, verdict_mapper=None))
            for level in (2, 5):
                level_name = "correct/incorrect (L2)" if level == 2 else "level-5 (L5)"
                true_c = [_cluster_verdict_for_metrics(l, level) for l in true_mapped]
                print(f"\n--- Normalized {level_name} ---")
                for name, pred_mapped in [
                    ("Advocate-only", advocate_primary_mapped),
                    ("Formula", formula_mapped),
                    ("LLM mediator", mediator_pred_mapped),
                ]:
                    pred_c = [_cluster_verdict_for_metrics(l, level) for l in pred_mapped]
                    acc = accuracy_score(true_c, pred_c)
                    print(f"  {name}: accuracy = {acc:.2%}")
            # Save mediator comparison CSV
            comparison_df = pd.DataFrame({
                "Claim": results_df["Claim"].values,
                "True Label": collectors["true_labels"],
                "Advocate-only Primary": advocate_primary,
                "Formula Primary": formula_primary,
                "LLM Mediator Primary": mediator_predicted,
            })
            comparison_path = save_results(
                comparison_df,
                base_path="experiments/results",
                prefix="spectrum_mediator_comparison",
            )
            logger.info("Mediator comparison CSV saved to: %s", comparison_path)
            print(f"\nMediator comparison saved to: {comparison_path}")
        else:
            print("\nMediator comparison skipped: no advocate weights (need advocate_output_mode=label_and_weights).")

    metrics_advocate = calculate_classification_metrics(
        true_mapped,
        advocate_primary_mapped,
        verdict_mapper=None,
    )
    logger.info("\n--- Advocate primary verdict (consensus) vs true ---")
    logger.info("Classification Metrics (Science Feedback claim verdict scale):")
    print("\n--- Advocate primary verdict (consensus) vs true ---")
    print(metrics_advocate)

    metrics_mediator = calculate_classification_metrics(
        true_mapped,
        mediator_pred_mapped,
        verdict_mapper=None,
    )
    logger.info("\n--- Mediator predicted verdict vs true ---")
    logger.info("Classification Metrics (Science Feedback claim verdict scale):")
    print("\n--- Mediator predicted verdict vs true ---")
    print(metrics_mediator)

    # Normalized (clustered) metrics: level 2 (correct/incorrect) and level 5
    for level in (2, 5):
        true_clustered = [_cluster_verdict_for_metrics(l, level) for l in true_mapped]
        advocate_clustered = [_cluster_verdict_for_metrics(l, level) for l in advocate_primary_mapped]
        mediator_clustered = [_cluster_verdict_for_metrics(l, level) for l in mediator_pred_mapped]
        level_name = "correct/incorrect" if level == 2 else "level-5 categories"
        metrics_advocate_cluster = calculate_classification_metrics(
            true_clustered,
            advocate_clustered,
            verdict_mapper=None,
        )
        metrics_mediator_cluster = calculate_classification_metrics(
            true_clustered,
            mediator_clustered,
            verdict_mapper=None,
        )
        logger.info("\n--- Advocate primary vs true (normalized, level=%s: %s) ---", level, level_name)
        print("\n--- Advocate primary vs true (normalized, level=%s: %s) ---" % (level, level_name))
        print(metrics_advocate_cluster)
        logger.info("\n--- Mediator predicted vs true (normalized, level=%s: %s) ---", level, level_name)
        print("\n--- Mediator predicted vs true (normalized, level=%s: %s) ---" % (level, level_name))
        print(metrics_mediator_cluster)

    # Weighted distribution metrics (log loss, top-2, top-3 accuracy)
    weighted_metrics_str = format_weighted_metrics(
        true_mapped,
        collectors["weighted_verdicts"],
    )
    logger.info(weighted_metrics_str)
    print(weighted_metrics_str)

    summary = f"Run summary: {n_ok}/{total_claims} claims evaluated successfully."
    if n_err:
        summary += f" {n_err} errors."
        if errors_file:
            summary += f" Details: {errors_file}"
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
