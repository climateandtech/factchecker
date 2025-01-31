from factchecker.strategies.advocate_mediator import AdvocateMediatorStrategy
from factchecker.experiments.advocate_mediator_climatefeedback.advocate_mediator_climatefeedback_prompts import advocate_primer, arbitrator_primer
from factchecker.utils.climatefeedback_utils import (
    map_verdict,
    sample_climatefeedback_claims,
    evaluate_climatefeedback_claims
)
from factchecker.utils.experiment_utils import (
    configure_logging,
    verify_environment,
    create_results_dataframe,
    save_results
)
from factchecker.utils.metrics import calculate_classification_metrics
from llama_index.core import Settings
import logging

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
    'similarity_top_k': 8,  # Number of similar chunks to retrieve
    
    # Advocate parameters
    'max_evidences': 10,  # Maximum pieces of evidence to consider
    'advocate_top_k': 8,  # Top k evidence pieces for advocate
    'min_score': 0.75,  # Minimum similarity score for evidence
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
        'similarity_top_k': EXPERIMENT_PARAMS['similarity_top_k'],
        'indexer_options': indexer_options_list[0]
    }]

    advocate_options = {
        'max_evidences': EXPERIMENT_PARAMS['max_evidences'],
        'top_k': EXPERIMENT_PARAMS['advocate_top_k'],
        'min_score': EXPERIMENT_PARAMS['min_score'],
        "thinking_llm": True,
        "thinking_token": "think",
    }

    mediator_options = {
        "thinking_llm": True,
        "thinking_token": "think",
    }

    strategy = AdvocateMediatorStrategy(
        indexer_options_list,
        retriever_options_list,
        advocate_options,
        mediator_options,
        advocate_primer,
        arbitrator_primer
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