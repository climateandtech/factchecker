import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from llama_index.core import Settings

from factchecker.experiments.advocate_mediator_climatefeedback.advocate_mediator_climatefeedback_prompts import (
    advocate_primer,
    arbitrator_primer,
)
from factchecker.strategies.advocate_mediator import AdvocateMediatorStrategy
from factchecker.tools.sources_downloader import SourcesDownloader
from factchecker.utils.climatefeedback_utils import (
    evaluate_climatefeedback_claims,
    map_verdict,
    sample_climatefeedback_claims,
)
from factchecker.utils.experiment_utils import (
    configure_logging,
    create_results_dataframe,
    save_evaluation_errors,
    save_results,
    show_evaluation_errors,
    verify_environment,
)
from factchecker.utils.metrics import calculate_classification_metrics

logger = logging.getLogger(__name__)

# Experiment Parameters (all controllable; overridable via main(experiment_options=...))
EXPERIMENT_PARAMS = {
    # Dataset parameters
    'dataset_path': 'datasets/Combined_Overview_Climate_Feedback_Claims.csv',
    'total_samples': 10,  # Reduced from 100 to work with available data
    'correct_ratio': 0.3,  # Ratio of correct claims in sample

    # Document processing parameters
    'chunk_size': 150,  # Size of text chunks for indexing
    'chunk_overlap': 20,  # Overlap between chunks

    # Indexing parameters
    'main_source_directory': 'data/sources',

    # Sources download (CSV -> PDFs)
    'sources_csv': 'factchecker/experiments/advocate_mediator_climatefeedback/advocate_mediator_climatefeedback_sources.csv',
    'sources_output_folder': 'data/sources',
    'sources_skip_existing': True,
    'sources_max_sources': 1,  # 1 for quick run; set None to download all sources
    'sources_url_column': 'url',
    'sources_output_filename_column': 'output_filename',
    'sources_output_subfolder_column': 'output_subfolder',

    # Retrieval parameters
    'top_k': 8,  # Number of similar chunks to retrieve

    # Label options
    'label_options': ['correct', 'incorrect', 'not_enough_information'],
}


def setup_sources(params: Optional[Dict[str, Any]] = None) -> list[str]:
    """Download the sources for the indices. Uses EXPERIMENT_PARAMS unless params override."""
    p = {**EXPERIMENT_PARAMS, **(params or {})}
    output_folder = p['sources_output_folder']
    downloader = SourcesDownloader(output_folder=output_folder)
    downloaded_files = downloader.download_pdfs_from_csv(
        p['sources_csv'],
        row_indices=None,
        url_column=p['sources_url_column'],
        output_filename_column=p['sources_output_filename_column'],
        output_subfolder_column=p['sources_output_subfolder_column'],
        skip_existing=p['sources_skip_existing'],
        max_sources=p.get('sources_max_sources'),
    )
    logging.info(f"Downloaded files: {downloaded_files}")
    return downloaded_files

def setup_strategy(params: Optional[Dict[str, Any]] = None, downloaded_files: Optional[list] = None) -> AdvocateMediatorStrategy:
    """Sets up the advocate-mediator strategy. Uses EXPERIMENT_PARAMS unless params override."""
    verify_environment()
    p = {**EXPERIMENT_PARAMS, **(params or {})}

    # Configure LlamaIndex parameters for this experiment
    Settings.chunk_size = p['chunk_size']
    Settings.chunk_overlap = p['chunk_overlap']

    main_source_directory = p['main_source_directory']
    max_sources = p.get('sources_max_sources')

    # When we have a list of specific files (e.g. from download limit or local folder): load only those
    if downloaded_files:
        indexer_options_list = [
            {
                'files': downloaded_files,
                'index_name': "ipcc",
            },
        ]
    elif max_sources is not None and max_sources <= 1:
        indexer_options_list = [
            {
                'source_directory': os.path.join(main_source_directory, "ipcc"),
                'index_name': "ipcc",
            },
        ]
    else:
        # Create an indexer for each subfolder in the main sources directory
        indexer_options_list = [
            {
                'source_directory': os.path.join(main_source_directory, "ipcc"),
                'index_name': "ipcc"
            },
            {
                'source_directory': os.path.join(main_source_directory, "wmo"),
                'index_name': "wmo"
            },
            {
                'source_directory': os.path.join(main_source_directory, "nipcc"),
                'index_name': "nipcc"
            },
        ]
    # Merge any indexer overrides (e.g. chunk_size, embed_batch_size from run_experiments)
    indexer_overrides = {k: v for k, v in p.items() if k in (
        'chunk_size', 'chunk_overlap', 'embed_batch_size', 'index_path',
        'embedding_type', 'embedding_model', 'show_progress'
    )}
    if indexer_overrides:
        for opts in indexer_options_list:
            opts.update(indexer_overrides)

    retriever_options_list = [{'top_k': p['top_k']} for _ in indexer_options_list]

    advocate_options = {
        'system_prompt': advocate_primer,
        'label_options': p['label_options'],
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

def main(experiment_options: Optional[Dict[str, Any]] = None):
    """
    Run the advocate-mediator climatefeedback experiment.

    All behaviour is controlled by EXPERIMENT_PARAMS; pass experiment_options to override
    (e.g. from run_experiments.py or CLI). Keys can include: sources_csv, sources_output_folder,
    sources_skip_existing, sources_max_sources, sources_url_column, sources_output_filename_column,
    sources_output_subfolder_column, chunk_size, chunk_overlap, main_source_directory, top_k,
    dataset_path, total_samples, correct_ratio, label_options, embed_batch_size, etc.
    """
    configure_logging()
    params = {**EXPERIMENT_PARAMS, **(experiment_options or {})}

    # Download sources (fully controlled by params)
    downloaded_files = setup_sources(params=params)

    # Setup strategy (uses params; when sources_max_sources<=1, pass files so we load only those)
    strategy = setup_strategy(params=params, downloaded_files=downloaded_files)

    # Load and sample claims
    logger.info("Loading and sampling claims...")
    sampled_claims = sample_climatefeedback_claims(
        csv_path=params['dataset_path'],
        total_samples=params['total_samples'],
        correct_ratio=params['correct_ratio']
    )

    # Evaluate claims
    collectors, errors = evaluate_climatefeedback_claims(strategy, sampled_claims)

    # Show errors to the user (summary + details), not only in logs
    show_evaluation_errors(errors, total_claims=len(sampled_claims))

    total_claims = len(sampled_claims)
    n_ok = len(collectors['true_labels'])
    n_err = len(errors)
    errors_file = None

    # Persist errors to CSV for inspection
    if errors:
        errors_file = save_evaluation_errors(errors, base_path="experiments/results", prefix="evaluation_errors")
        if errors_file:
            logger.info(f"Evaluation errors saved to: {errors_file}")
            print(f"Evaluation errors saved to: {errors_file}")

    if n_ok == 0:
        logger.warning("No claims were successfully evaluated; skipping results and metrics.")
        print(f"Run summary: 0/{total_claims} claims evaluated successfully. {n_err} errors.")
        return

    # Create and save results DataFrame (partial results when some claims failed)
    logger.info("Creating results DataFrame...")
    results_df = create_results_dataframe(
        sampled_claims,
        collectors,
        verdict_mapper=map_verdict
    )

    results_file = save_results(results_df)
    logger.info(f"Results saved to: {results_file}")

    # Calculate and print metrics (on successful claims only)
    logger.info("Calculating classification metrics...")
    metrics = calculate_classification_metrics(
        collectors['true_labels'],
        collectors['predicted_results'],
        verdict_mapper=map_verdict
    )
    logger.info("\nClassification Metrics:")
    print(metrics)

    # Run summary utilizing error output
    summary = f"Run summary: {n_ok}/{total_claims} claims evaluated successfully."
    if n_err:
        summary += f" {n_err} errors."
        if errors_file:
            summary += f" Details: {errors_file}"
    print(f"\n{summary}")


if __name__ == "__main__":
    main()