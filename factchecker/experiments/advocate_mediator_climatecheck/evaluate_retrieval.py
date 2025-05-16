"""Script to evaluate retrieval metrics on existing results files."""

import argparse
import logging
import os
import glob
from pathlib import Path

from factchecker.utils.experiment_utils import configure_logging
from factchecker.experiments.advocate_mediator_climatecheck.retrieval_metrics import RetrievalEvaluator

logger = logging.getLogger(__name__)

def find_results_files(directory: str = "results", pattern: str = "*.csv") -> list[str]:
    """Find result files matching the pattern in the given directory.
    
    Args:
        directory: Directory to search for result files
        pattern: File pattern to match
        
    Returns:
        List of matching file paths
    """
    results_pattern = os.path.join(directory, pattern)
    return glob.glob(results_pattern)

def main():
    """Main function to run retrieval evaluation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics on existing results")
    parser.add_argument(
        "--results-file", 
        type=str, 
        help="Path to the results file to evaluate"
    )
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="results",
        help="Directory containing results files (if --results-file not specified)"
    )
    parser.add_argument(
        "--pattern", 
        type=str, 
        default="*climatecheck*.csv",
        help="File pattern to match (if --results-file not specified)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results/metrics",
        help="Directory to save metrics results"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit the number of claims to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging()
    
    # Determine files to evaluate
    files_to_evaluate = []
    if args.results_file:
        if os.path.exists(args.results_file):
            files_to_evaluate = [args.results_file]
        else:
            logger.error(f"Results file not found: {args.results_file}")
            return
    else:
        files_to_evaluate = find_results_files(args.results_dir, args.pattern)
        
    if not files_to_evaluate:
        logger.error("No results files found to evaluate")
        return
        
    logger.info(f"Found {len(files_to_evaluate)} results files to evaluate")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each file
    for results_file in files_to_evaluate:
        file_name = os.path.basename(results_file)
        logger.info(f"\nEvaluating retrieval metrics for: {file_name}")
        
        # Set up output path
        metrics_file = os.path.join(
            args.output_dir, 
            f"metrics_{Path(file_name).stem}.json"
        )
        
        # Create evaluator
        evaluator = RetrievalEvaluator(results_path=metrics_file)
        
        try:
            # Evaluate retrieval
            metrics = evaluator.evaluate_from_results_file(
                results_file=results_file, 
                claims_limit=args.limit
            )
            
            # Display metrics
            logger.info("\nRetrieval Metrics:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{metric}: {value:.4f}")
                else:
                    logger.info(f"{metric}: {value}")
                    
            logger.info(f"Detailed metrics saved to: {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error evaluating {file_name}: {str(e)}")
    
    logger.info("\nRetrieval evaluation complete")

if __name__ == "__main__":
    main() 