"""
Generic utilities for running and analyzing experiments.
"""
from typing import Dict, List, Any, Tuple, Optional, Callable
import pandas as pd
from datetime import datetime
import os
import logging
from llama_index.core import Settings

logger = logging.getLogger(__name__)


def show_evaluation_errors(errors: List[Dict[str, Any]], total_claims: Optional[int] = None) -> None:
    """
    Print evaluation errors to stdout (summary and per-error details), in addition to logging.
    """
    if not errors:
        return
    n = len(errors)
    total = total_claims if total_claims is not None else "?"
    print("\n--- Evaluation errors ---")
    print(f"Failed claims: {n} / {total}")
    by_type: Dict[str, int] = {}
    for e in errors:
        t = e.get("error_type", "Unknown")
        by_type[t] = by_type.get(t, 0) + 1
    print("By error type:", by_type)
    print("Details:")
    for e in errors:
        idx = e.get("claim_index", "?")
        typ = e.get("error_type", "?")
        msg = e.get("error_message", "?")
        preview = e.get("claim_preview", "")
        print(f"  [{idx}] {typ}: {msg}")
        if preview:
            print(f"       Claim: {preview}")
    print("---\n")


def configure_logging():
    """Configure basic logging for experiments. Ensures indexers and LlamaIndex show INFO logs."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Ensure factchecker and LlamaIndex loggers show INFO (indexing, retrieval, etc.)
    for name in ('factchecker', 'llama_index', 'root'):
        logging.getLogger(name).setLevel(logging.INFO)

def configure_llama_index():
    """Configure LlamaIndex settings."""
    Settings.chunk_size = 150
    Settings.chunk_overlap = 20

def verify_environment():
    """Verify required environment variables are set."""
    required_vars = ['OPENAI_API_MODEL', 'LLM_TYPE']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    logger.info("Environment configuration:")
    logger.info(f"OPENAI_API_MODEL = {os.getenv('OPENAI_API_MODEL')}")
    logger.info(f"LLM_TYPE = {os.getenv('LLM_TYPE')}")

def get_default_indexer_options(source_directory: str = 'data', index_name: str = 'advocate1_index'):
    """Get default indexer options."""
    return [{
        'source_directory': source_directory,
        'index_name': index_name
    }]

def get_default_retriever_options(indexer_options: list, top_k: int = 8):
    """Get default retriever options."""
    return [{
        'top_k': top_k,
        'indexer_options': indexer_options[0]
    }]

def get_default_advocate_options(max_evidences: int = 10, top_k: int = 8, min_score: float = 0.75):
    """Get default advocate options."""
    return {
        'max_evidences': max_evidences,
        'top_k': top_k,
        'min_score': min_score
    }

# Type aliases
ClaimData = Tuple[str, str, List[str], List[str]]  # (true_label, final_verdict, verdicts, reasonings)
ResultsDict = Dict[str, List]

def initialize_results_collectors(num_advocates: int) -> ResultsDict:
    """
    Initializes all the lists needed to collect results during an experiment.
    
    Args:
        num_advocates: Number of advocates in the experiment
        
    Returns:
        Dictionary containing empty lists for collecting various results
        
    Raises:
        ValueError: If num_advocates is negative
    """
    if num_advocates < 0:
        raise ValueError("num_advocates cannot be negative")
        
    return {
        'true_labels': [],
        'predicted_results': [],
        'mediator_reasonings': [],
        'advocate_evidences': [[] for _ in range(num_advocates)],
        'advocate_verdicts': [[] for _ in range(num_advocates)],
        'advocate_reasonings': [[] for _ in range(num_advocates)],
        'claim_indices': [],
    }

def collect_evaluation_results(
    collectors: ResultsDict,
    claim_data: ClaimData,
    num_advocates: Optional[int] = None,
    claim_index: Optional[Any] = None,
) -> ResultsDict:
    """
    Collects results from a single claim evaluation.

    Args:
        collectors: Dictionary of result collectors
        claim_data: Tuple of (true_label, final_verdict, verdicts, reasonings)
        num_advocates: Number of advocates (optional, for initialization)
        claim_index: Index of the claim in the source DataFrame (for partial-results alignment)

    Returns:
        Updated collectors dictionary
    """
    if not collectors and num_advocates is None:
        raise ValueError("collectors cannot be empty when num_advocates is None")

    true_label, final_verdict, verdicts, reasonings = claim_data

    if not isinstance(verdicts, list) or not isinstance(reasonings, list):
        raise ValueError("verdicts and reasonings must be lists")

    # Initialize collectors if first run
    if num_advocates is not None and not collectors.get('advocate_evidences'):
        collectors = initialize_results_collectors(num_advocates)

    if "claim_indices" not in collectors:
        collectors["claim_indices"] = []

    collectors["true_labels"].append(true_label)
    collectors["predicted_results"].append(final_verdict)
    collectors["mediator_reasonings"].append(reasonings[-1])
    if claim_index is not None:
        collectors["claim_indices"].append(claim_index)

    for i in range(len(verdicts)):
        collectors["advocate_evidences"][i].append(verdicts[i])
        collectors["advocate_verdicts"][i].append(verdicts[i])
        collectors["advocate_reasonings"][i].append(reasonings[i])

    return collectors

def create_results_dataframe(
    claims: pd.DataFrame,
    collectors: ResultsDict,
    verdict_mapper: Optional[Callable[[str], str]] = None
) -> pd.DataFrame:
    """
    Creates a DataFrame from collected experiment results.
    
    Args:
        claims: Original claims DataFrame
        collectors: Dictionary of collected results
        verdict_mapper: Optional function to map verdicts to standardized categories
        
    Returns:
        DataFrame containing all results
        
    Raises:
        ValueError: If required columns are missing
        ValueError: If collectors and claims have different lengths
    """
    if 'Claim' not in claims.columns:
        raise ValueError("claims DataFrame must contain 'Claim' column")

    claim_indices = collectors.get('claim_indices')
    if claim_indices is not None and len(claim_indices) > 0:
        # Partial results: align claims to successful evaluations only
        if len(claim_indices) != len(collectors['true_labels']):
            raise ValueError("claim_indices length must match collected results length")
        claims_subset = claims.loc[claim_indices].reset_index(drop=True)
    else:
        if len(claims) != len(collectors['true_labels']):
            raise ValueError("Number of claims doesn't match number of collected results")
        claims_subset = claims

    results_dict = {
        'Claim': claims_subset['Claim'].values,
        'True Label': collectors['true_labels'],
        'Predicted Verdict': collectors['predicted_results'],
        'Mediator Reasoning': collectors['mediator_reasonings']
    }
    
    # Add mapped verdicts if mapper provided
    if verdict_mapper:
        results_dict['Mapped True Label'] = [
            verdict_mapper(label) for label in collectors['true_labels']
        ]
    
    # Add advocate results
    for i in range(len(collectors['advocate_evidences'])):
        results_dict.update({
            f'Advocate {i+1} Evidence': collectors['advocate_evidences'][i],
            f'Advocate {i+1} Verdict': collectors['advocate_verdicts'][i],
            f'Advocate {i+1} Reasoning': collectors['advocate_reasonings'][i]
        })
    
    return pd.DataFrame(results_dict)

def save_results(
    results_df: pd.DataFrame,
    base_path: str = "experiments/results",
    prefix: str = "claims_results"
) -> str:
    """
    Saves results DataFrame to a timestamped CSV file.
    
    Args:
        results_df: DataFrame containing results
        base_path: Directory to save results in
        prefix: Prefix for the filename
        
    Returns:
        Path to the saved file
        
    Raises:
        ValueError: If results_df is empty
    """
    if results_df.empty:
        raise ValueError("Cannot save empty results DataFrame")
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_path}/{prefix}_{timestamp}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    results_df.to_csv(filename, index=False)
    return filename


def save_evaluation_errors(
    errors: List[Dict[str, Any]],
    base_path: str = "experiments/results",
    prefix: str = "evaluation_errors",
) -> Optional[str]:
    """
    Save evaluation errors to a timestamped CSV for inspection.

    Args:
        errors: List of error dicts (claim_index, error_type, error_message, claim_preview)
        base_path: Directory to save in
        prefix: Filename prefix

    Returns:
        Path to the saved file, or None if errors is empty
    """
    if not errors:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_path}/{prefix}_{timestamp}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame(errors)
    df.to_csv(filename, index=False)
    return filename