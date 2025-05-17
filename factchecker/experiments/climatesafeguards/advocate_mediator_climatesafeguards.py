import logging

from llama_index.core import Settings

from datasets import load_dataset
from factchecker.experiments.climatesafeguards.advocate_mediator_climatesafeguards_prompts import (
    advocate_primer,
    arbitrator_primer,
    extractor_context_prompt,
)
from factchecker.strategies.extractor_advocate_mediator import (
    ExtractorAdvocateMediatorStrategy,
)
from factchecker.utils.climatefeedback_utils import (
    evaluate_climatefeedback_texts,
    map_verdict,
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
    "dataset_path": "datasets/Combined_Overview_Climate_Feedback_Claims.csv",
    "total_samples": 10,  # Reduced from 100 to work with available data
    "correct_ratio": 0.3,  # Ratio of correct claims in sample
    # Document processing parameters
    "chunk_size": 150,  # Size of text chunks for indexing
    "chunk_overlap": 20,  # Overlap between chunks
    # Indexing parameters
    "source_directory": "data",
    "index_name": "advocate1_index",
    # Retrieval parameters
    "top_k": 8,  # Number of similar chunks to retrieve
    # Evidence parameters
    "max_evidences": 10,  # Maximum pieces of evidence to consider
    "min_score": 0.65,  # Minimum similarity score for evidence
    # Label options
    "label_options": ["correct", "incorrect", "not_enough_information"],
}


def setup_strategy() -> ExtractorAdvocateMediatorStrategy:
    """Sets up the advocate-mediator strategy with experiment-specific options."""
    verify_environment()

    # Configure LlamaIndex parameters for this experiment
    Settings.chunk_size = EXPERIMENT_PARAMS["chunk_size"]
    Settings.chunk_overlap = EXPERIMENT_PARAMS["chunk_overlap"]

    indexer_options_list = [
        {
            "source_directory": EXPERIMENT_PARAMS["source_directory"],
            "index_name": EXPERIMENT_PARAMS["index_name"],
        }
    ]

    retriever_options_list = [
        {
            "top_k": EXPERIMENT_PARAMS["top_k"],
        }
    ]

    advocate_options = {
        "system_prompt": advocate_primer,
        "label_options": EXPERIMENT_PARAMS["label_options"],
        "with_context": True,
    }

    evidence_options = {
        "min_score": EXPERIMENT_PARAMS["min_score"],
        "query_template": "{claim}",
    }

    mediator_options = {"system_prompt": arbitrator_primer}

    extractor_options = {
        "context_prompt_template": extractor_context_prompt,
    }

    strategy = ExtractorAdvocateMediatorStrategy(
        indexer_options_list=indexer_options_list,
        retriever_options_list=retriever_options_list,
        extractor_options=extractor_options,
        advocate_options=advocate_options,
        evidence_options=evidence_options,
        mediator_options=mediator_options,
    )
    logger.info("Strategy initialized successfully")
    return strategy


def download_claims():
    dataset_name = "charlotte-samson/climatesafeguards"
    dataset = load_dataset(dataset_name, split="train")
    dataset = (
        dataset.select_columns(
            [
                "whisper-largev3",
                "Misinfo",
            ]
        )
        .shuffle()
        .to_pandas()
        .dropna()
        .sample(n=100)
        .rename(columns={"whisper-largev3": "text"})
    )
    dataset["label"] = dataset.Misinfo.astype(int).map({0: "correct", 1: "incorrect"})
    return dataset


def main():
    from time import time

    t = time()
    # Configure logging
    configure_logging()

    # Setup strategy
    strategy = setup_strategy()

    # Load and sample claims ensuring balanced dataset
    logger.info("Loading and sampling texts...")
    texts = download_claims()
    logger.info(f"Found {len(texts)} texts")

    # Evaluate claims
    collectors = evaluate_climatefeedback_texts(
        strategy, texts, text_col="text", label_col="label"
    )

    # Create and save results DataFrame
    logger.info("Creating results DataFrame...")
    results_df = create_results_dataframe(
        texts,
        collectors,
        verdict_mapper=map_verdict,
        text_col="text",
    )

    results_file = save_results(results_df)
    logger.info(f"Results saved to: {results_file}")

    # Calculate and print metrics
    logger.info("Calculating classification metrics...")
    metrics = calculate_classification_metrics(
        collectors["true_labels"],
        collectors["predicted_results"],
        verdict_mapper=map_verdict,
    )
    logger.info("\nClassification Metrics:")
    print(metrics)

    print(texts.label.value_counts())

    print(f"Time Taken: {round(time() - t)} seconds!")


if __name__ == "__main__":
    main()
