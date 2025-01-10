"""
Utilities for calculating and analyzing experiment metrics.
"""
from typing import List, Optional, Callable
from sklearn.metrics import classification_report

def calculate_classification_metrics(
    true_labels: List[str],
    predicted_results: List[str],
    verdict_mapper: Optional[Callable[[str], str]] = None
) -> str:
    """
    Calculates classification metrics for the experiment results.
    
    Args:
        true_labels: List of true labels
        predicted_results: List of predicted labels
        verdict_mapper: Optional function to map verdicts to standardized categories
        
    Returns:
        String containing the classification report
        
    Raises:
        ValueError: If input lists are empty or have different lengths
    """
    if not true_labels or not predicted_results:
        raise ValueError("Input lists cannot be empty")
        
    if len(true_labels) != len(predicted_results):
        raise ValueError("Input lists must have the same length")
    
    if verdict_mapper:
        true_labels = [verdict_mapper(label.lower()) for label in true_labels]
        predicted_results = [verdict_mapper(result.lower()) for result in predicted_results]
    else:
        true_labels = [label.lower() for label in true_labels]
        predicted_results = [result.lower() for result in predicted_results]
    
    return classification_report(true_labels, predicted_results) 