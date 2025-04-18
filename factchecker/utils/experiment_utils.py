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

def configure_logging():
    """Configure basic logging for experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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
        'advocate_reasonings': [[] for _ in range(num_advocates)]
    }

def collect_evaluation_results(
    collectors: ResultsDict,
    claim_data: ClaimData,
    num_advocates: Optional[int] = None
) -> ResultsDict:
    """
    Collects results from a single claim evaluation.
    
    Args:
        collectors: Dictionary of result collectors
        claim_data: Tuple of (true_label, final_verdict, verdicts, reasonings)
        num_advocates: Number of advocates (optional, for initialization)
        
    Returns:
        Updated collectors dictionary
        
    Raises:
        ValueError: If collectors is None or empty when num_advocates is None
        ValueError: If claim_data components don't match expected structure
    """
    if not collectors and num_advocates is None:
        raise ValueError("collectors cannot be empty when num_advocates is None")
        
    true_label, final_verdict, verdicts, reasonings = claim_data
    
    if not isinstance(verdicts, list) or not isinstance(reasonings, list):
        raise ValueError("verdicts and reasonings must be lists")
    
    # Initialize collectors if first run
    if num_advocates is not None and not collectors.get('advocate_evidences'):
        collectors = initialize_results_collectors(num_advocates)

    collectors['true_labels'].append(true_label)
    collectors['predicted_results'].append(final_verdict)
    collectors['mediator_reasonings'].append(reasonings[-1])
    
    for i in range(len(verdicts)):
        collectors['advocate_evidences'][i].append(verdicts[i])
        collectors['advocate_verdicts'][i].append(verdicts[i])
        collectors['advocate_reasonings'][i].append(reasonings[i])
    
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
        
    if len(claims) != len(collectors['true_labels']):
        raise ValueError("Number of claims doesn't match number of collected results")
    
    results_dict = {
        'Claim': claims['Claim'],
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