import logging

from llama_index.core import Settings

from factchecker.experiments.advocate_mediator_climatefeedback.advocate_mediator_climatefeedback_prompts import (
    advocate_primer,
    arbitrator_primer,
)
from factchecker.strategies.advocate_mediator import AdvocateMediatorStrategy
from factchecker.utils.climatefeedback_utils import (
    evaluate_climatefeedback_claims,
    map_verdict,
    sample_climatefeedback_claims,
)
from factchecker.utils.experiment_utils import (
    configure_logging,
    create_results_dataframe,
    save_results,
    verify_environment,
)
from factchecker.utils.metrics import calculate_classification_metrics

logger = logging.getLogger(__name__)

# Experiment Parameters
EXPERIMENT_PARAMS = {
    # Dataset parameters
    'dataset_path': 'datasets/Combined_Overview_Climate_Feedback_Claims.csv',
    'total_samples': 10,  # Reduced from 100 to work with available data
    'correct_ratio': 0.3,  # Ratio of correct claims in sample
    
    # Document processing parameters
    'chunk_size': 150,  # Size of text chunks for indexing
    'chunk_overlap': 20,  # Overlap between chunks
    
    # Indexing parameters
    'source_directory': 'data',
    'index_name': 'advocate1_index',
    
    # Retrieval parameters
    'top_k': 8,  # Number of similar chunks to retrieve

    # Label options
    'label_options': ['correct', 'incorrect', 'not_enough_information'],
}

def setup_strategy() -> AdvocateMediatorStrategy:
    """Sets up the advocate-mediator strategy with experiment-specific options."""
    verify_environment()
    
    # Configure LlamaIndex parameters for this experiment
    Settings.chunk_size = EXPERIMENT_PARAMS['chunk_size']
    Settings.chunk_overlap = EXPERIMENT_PARAMS['chunk_overlap']
    
    indexer_options_list = [{
        'source_directory': EXPERIMENT_PARAMS['source_directory'],
        'index_name': EXPERIMENT_PARAMS['index_name']
    }]

    retriever_options_list = [{
        'top_k': EXPERIMENT_PARAMS['top_k'],
    }]

    advocate_options = {
        'system_prompt': advocate_primer,
        'label_options': EXPERIMENT_PARAMS['label_options'],
    }

    evidence_options = {}

    mediator_options = {
        'system_prompt': arbitrator_primer
    }

    strategy = AdvocateMediatorStrategy(
        indexer_options_list,
        retriever_options_list,
        advocate_options,
        evidence_options,
        mediator_options,
    )
    logger.info("Strategy initialized successfully")
    return strategy

def main():
    # Configure logging
    configure_logging()

    # Setup strategy
    strategy = setup_strategy()

    # Load and sample claims ensuring balanced dataset
    logger.info("Loading and sampling claims...")
    sampled_claims = sample_climatefeedback_claims(
        csv_path=EXPERIMENT_PARAMS['dataset_path'],
        total_samples=EXPERIMENT_PARAMS['total_samples'],
        correct_ratio=EXPERIMENT_PARAMS['correct_ratio']
    )

    # Evaluate claims
    collectors = evaluate_climatefeedback_claims(strategy, sampled_claims)

    # Create and save results DataFrame
    logger.info("Creating results DataFrame...")
    results_df = create_results_dataframe(
        sampled_claims,
        collectors,
        verdict_mapper=map_verdict
    )
    
    results_file = save_results(results_df)
    logger.info(f"Results saved to: {results_file}")

    # Calculate and print metrics
    logger.info("Calculating classification metrics...")
    metrics = calculate_classification_metrics(
        collectors['true_labels'],
        collectors['predicted_results'],
        verdict_mapper=map_verdict
    )
    logger.info("\nClassification Metrics:")
    print(metrics)

if __name__ == "__main__":
    main()