"""
Utilities specific to the Climate Feedback dataset and experiments.
"""
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from factchecker.utils.experiment_utils import collect_evaluation_results, initialize_results_collectors
from tqdm import tqdm

logger = logging.getLogger(__name__)

VALID_LEVELS = {2, 5, 7}

def map_verdict(verdict: str, level: int = 2) -> str:
    """Maps Climate Feedback verdicts to standardized categories."""
    if not verdict or not isinstance(verdict, str):
        return "unknown"
        
    # Normalize the verdict by converting to lowercase and stripping whitespace
    verdict = verdict.strip().lower()
    
    if level not in VALID_LEVELS:
        raise ValueError(f"Level must be one of {VALID_LEVELS}")
    
    # Level 7 - Most granular mapping
    if level == 7:
        verdict_map = {
            "incorrect": "incorrect",
            "inaccurate": "inaccurate",
            "imprecise": "imprecise",
            "misleading": "misleading",
            "flawed reasoning": "flawed_reasoning",
            "lacks context": "lacks_context",
            "unsupported": "unsupported",
            "correct but": "correct_but",
            "correct": "correct",
            "mostly correct": "mostly_correct",
            "accurate": "accurate",
            "mostly accurate": "mostly_accurate"
        }
        return verdict_map.get(verdict, "unknown")
    
    # Level 5 - Medium granularity
    if level == 5:
        verdict_map = {
            "incorrect": "incorrect",
            "inaccurate": "incorrect",
            "imprecise": "imprecise",
            "misleading": "misleading",
            "flawed reasoning": "flawed_reasoning",
            "lacks context": "unsupported",
            "unsupported": "unsupported",
            "correct but": "mostly_correct",
            "correct": "correct",
            "mostly correct": "mostly_correct",
            "accurate": "correct",
            "mostly accurate": "correct"
        }
        return verdict_map.get(verdict, "unknown")
    
    # Level 2 - Binary classification
    if level == 2:
        correct_verdicts = [
            "correct", "mostly correct", "accurate", "mostly accurate",
            "correct but"
        ]
        incorrect_verdicts = [
            "incorrect", "inaccurate", "misleading", "flawed reasoning",
            "lacks context", "unsupported", "mostly inaccurate", "imprecise"
        ]
        
        if verdict in correct_verdicts:
            return "correct"
        elif verdict in incorrect_verdicts:
            return "incorrect"
        return "unknown"

def sample_climatefeedback_claims(
    csv_path: str,
    total_samples: int,
    correct_ratio: float = 0.3
) -> pd.DataFrame:
    """
    Load and sample claims from the Climate Feedback dataset ensuring a balanced ratio of correct claims.
    
    Args:
        csv_path: Path to the CSV file containing claims
        total_samples: Total number of samples to return
        correct_ratio: Desired ratio of correct claims (0-1)
        
    Returns:
        DataFrame containing sampled claims
        
    Raises:
        ValueError: If parameters are invalid or if not enough claims are available
    """
    if correct_ratio < 0 or correct_ratio > 1:
        raise ValueError("correct_ratio must be between 0 and 1")
        
    if total_samples < 1:
        raise ValueError("total_samples must be positive")
    
    # Load claims
    try:
        claims_df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {str(e)}")
    
    if 'Claim' not in claims_df.columns or 'Climate Feedback' not in claims_df.columns:
        raise ValueError("CSV must contain 'Claim' and 'Climate Feedback' columns")
    
    # Map verdicts to binary categories
    claims_df['verdict_binary'] = claims_df['Climate Feedback'].apply(lambda x: map_verdict(x, level=2))
    
    # Split into correct and incorrect claims
    correct_claims = claims_df[claims_df['verdict_binary'] == 'correct']
    other_claims = claims_df[claims_df['verdict_binary'] == 'incorrect']
    
    # Calculate required numbers
    num_correct = int(total_samples * correct_ratio)
    num_other = total_samples - num_correct
    
    logger.info(f"Total claims in dataset: {len(claims_df)}")
    logger.info(f"Number of correct claims available: {len(correct_claims)}")
    
    # Verify we have enough claims
    if len(correct_claims) < num_correct:
        raise ValueError(f"Not enough correct claims. Need {num_correct}, have {len(correct_claims)}")
    if len(other_claims) < num_other:
        raise ValueError(f"Not enough incorrect claims. Need {num_other}, have {len(other_claims)}")
    
    # Sample claims
    sampled_correct = correct_claims.sample(n=num_correct)
    sampled_other = other_claims.sample(n=num_other)
    
    logger.info(f"Sampled {num_correct} correct claims")
    logger.info(f"Sampled {num_other} additional claims")
    
    # Combine and shuffle
    sampled_claims = pd.concat([sampled_correct, sampled_other]).sample(frac=1).reset_index(drop=True)
    
    correct_count = len(sampled_claims[sampled_claims['verdict_binary'] == 'correct'])
    logger.info(f"Final sample - Total: {len(sampled_claims)}, Correct: {correct_count} ({correct_count/len(sampled_claims)*100:.1f}%)")
    
    return sampled_claims

def evaluate_climatefeedback_claim(
    strategy,
    claim: str,
    true_label: str,
    collectors: Dict,
    claim_index: int,
    total_claims: int
) -> Dict:
    """
    Evaluate a single Climate Feedback claim and collect results.
    
    Args:
        strategy: The evaluation strategy to use
        claim: The claim text to evaluate
        true_label: The true label from Climate Feedback
        collectors: Dictionary of result collectors
        claim_index: Index of current claim (for logging)
        total_claims: Total number of claims (for logging)
        
    Returns:
        Updated collectors dictionary
        
    Raises:
        Exception: If evaluation fails
    """
    try:
        final_verdict, verdicts, reasonings = strategy.evaluate_claim(claim)
        collectors = collect_evaluation_results(
            collectors,
            (true_label, final_verdict, verdicts, reasonings),
            num_advocates=len(verdicts) if not collectors['advocate_evidences'] else None
        )
        
        logger.debug(f"\nClaim {claim_index + 1}/{total_claims}:")
        logger.debug(f"Claim: {claim[:100]}...")
        logger.debug(f"True Label: {true_label} (Mapped: {map_verdict(true_label)})")
        logger.debug(f"Final Verdict: {final_verdict}")
        
    except Exception as e:
        logger.error(f"Error processing claim {claim_index + 1}: {str(e)}")
        logger.error(f"Claim text: {claim}")
        raise
        
    return collectors


def evaluate_climatefeedback_text(
    strategy,
    text: str,
    true_label: str,
    collectors: Dict,
    text_index: int,
    total_texts: int
) -> Dict:
    """
    Evaluate a single Climate Feedback text and collect results.
    
    Args:
        strategy: The evaluation strategy to use
        text: The text text to evaluate
        true_label: The true label from Climate Feedback
        collectors: Dictionary of result collectors
        text_index: Index of current text (for logging)
        total_texts: Total number of texts (for logging)
        
    Returns:
        Updated collectors dictionary
        
    Raises:
        Exception: If evaluation fails
    """
    try:
        final_verdict, verdicts, reasonings = strategy.evaluate_text(text)
        collectors = collect_evaluation_results(
            collectors,
            (true_label, final_verdict, verdicts, reasonings),
            num_advocates=len(verdicts) if not collectors['advocate_evidences'] else None
        )
        
        logger.debug(f"\nText {text_index + 1}/{total_texts}:")
        logger.debug(f"Text: {text[:100]}...")
        logger.debug(f"True Label: {true_label} (Mapped: {map_verdict(true_label)})")
        logger.debug(f"Final Verdict: {final_verdict}")
        
    except Exception as e:
        logger.error(f"Error processing text {text_index + 1}: {str(e)}")
        logger.error(f"Text: {text}")
        raise
        
    return collectors

def evaluate_climatefeedback_claims(strategy, sampled_claims, num_advocates: int = 1, text_col: str = "Claim", label_col:str = "Climate Feedback") -> Dict:
    """
    Evaluate a batch of Climate Feedback claims using the provided strategy.
    
    Args:
        strategy: The evaluation strategy to use
        sampled_claims: DataFrame containing claims to evaluate
        num_advocates: Number of advocates in the strategy
        
    Returns:
        Dictionary containing collected results
        
    Raises:
        ValueError: If sampled_claims is empty or missing required columns
    """
    if sampled_claims.empty:
        raise ValueError("sampled_claims cannot be empty")
        
    if text_col not in sampled_claims.columns or label_col not in sampled_claims.columns:
        raise ValueError(f"sampled_claims must contain {text_col} and {label_col} columns")
    
    collectors = initialize_results_collectors(num_advocates)
    
    logger.info("Starting claim evaluation...")
    for idx, row in tqdm(sampled_claims.iterrows(), total=len(sampled_claims), desc="Evaluating claims"):
        try:
            collectors = evaluate_climatefeedback_claim(
                strategy=strategy,
                claim=row[text_col],
                true_label=row[label_col],
                collectors=collectors,
                claim_index=idx,
                total_claims=len(sampled_claims)
            )
        except Exception as e:
            logger.error(f"Skipping claim {idx + 1} due to error {str(e)}")
            continue
            
    return collectors 


def evaluate_climatefeedback_texts(strategy, sampled_texts, num_advocates: int = 1, text_col: str = "Claim", label_col:str = "Climate Feedback") -> Dict:
    """
    Evaluate a batch of Climate Feedback claims using the provided strategy.
    
    Args:
        strategy: The evaluation strategy to use
        sampled_texts: DataFrame containing claims to evaluate
        num_advocates: Number of advocates in the strategy
        
    Returns:
        Dictionary containing collected results
        
    Raises:
        ValueError: If sampled_texts is empty or missing required columns
    """
    if sampled_texts.empty:
        raise ValueError("sampled_texts cannot be empty")
        
    if text_col not in sampled_texts.columns or label_col not in sampled_texts.columns:
        raise ValueError(f"sampled_texts must contain {text_col} and {label_col} columns")
    
    collectors = initialize_results_collectors(num_advocates)
    
    logger.info("Starting text evaluation...")
    for idx, row in tqdm(sampled_texts.iterrows(), total=len(sampled_texts), desc="Evaluating texts"):
        try:
            collectors = evaluate_climatefeedback_text(
                strategy=strategy,
                text=row[text_col],
                true_label=row[label_col],
                collectors=collectors,
                text_index=idx,
                total_texts=len(sampled_texts)
            )
        except Exception as e:
            logger.error(f"Skipping text {idx + 1} due to error")
            raise
            
    return collectors 