#!/usr/bin/env python3
"""
Script to run fact-checking experiments with optimized settings.
Pass --experiment <module.path> to run any experiment that exposes a main().

Ollama (LLM and embeddings) is controlled via .env: set LLM_TYPE=ollama,
EMBEDDING_TYPE=ollama, OLLAMA_API_BASE_URL, OLLAMA_MODEL, etc. in .env.
"""

import argparse
import importlib
import logging
import os
import time
from datetime import datetime

# Load .env first so it controls LLM/embedding/Ollama settings
from dotenv import load_dotenv
load_dotenv(override=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Configure environment settings for the experiment. Ollama is controlled via .env."""
    # Only set OLLAMA_API_BASE_URL default if not already set (e.g. from .env)
    if not os.environ.get("OLLAMA_API_BASE_URL"):
        os.environ["OLLAMA_API_BASE_URL"] = "http://localhost:11434"
    llm_type = os.environ.get("LLM_TYPE", "openai")
    emb_type = os.environ.get("EMBEDDING_TYPE", "openai")
    logger.info(f"Using .env: LLM_TYPE={llm_type}, EMBEDDING_TYPE={emb_type}, OLLAMA_API_BASE_URL={os.environ['OLLAMA_API_BASE_URL']}")
    if emb_type == "ollama":
        emb_model = os.environ.get("OLLAMA_EMBEDDING_MODEL") or os.environ.get("OLLAMA_MODEL", "nomic-embed-text")
        logger.info(f"Ollama embedding model: {emb_model}")
    # Check if Ollama is available when using Ollama
    if llm_type == "ollama" or emb_type == "ollama":
        import httpx
        try:
            base = os.environ["OLLAMA_API_BASE_URL"]
            response = httpx.get(f"{base}/api/tags")
            if response.status_code == 200:
                logger.info(f"Ollama server is available at {base}")
                models = response.json().get("models", [])
                logger.info(f"Available models: {[m.get('name') for m in models]}")
            else:
                logger.warning(f"Ollama server at {base} returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama server: {str(e)}")
            logger.warning("Set LLM_TYPE=ollama and EMBEDDING_TYPE=ollama in .env and ensure Ollama is running.")

DEFAULT_EXPERIMENT = "factchecker.experiments.advocate_mediator_climatefeedback.advocate_mediator_climatefeedback"


def run_experiment(experiment_module: str, experiment_options=None):
    """Run an experiment by module path. Options are passed to main() when it accepts them."""
    if experiment_options is None:
        experiment_options = {}

    logger.info("Starting experiment: %s", experiment_module)

    try:
        mod = importlib.import_module(experiment_module)
        main = getattr(mod, "main", None)
        if main is None:
            logger.error("Module %s has no main() function.", experiment_module)
            return

        start_time = time.time()
        # Prefer passing experiment_options for experiments that support it
        try:
            main(experiment_options=experiment_options)
        except TypeError:
            try:
                main(indexer_options=experiment_options)
            except TypeError:
                main()
        elapsed = time.time() - start_time

        logger.info("Experiment completed successfully in %dm %ds", int(elapsed // 60), int(elapsed % 60))

    except ImportError as e:
        logger.error("Could not import experiment module: %s", e)
        logger.error("Use a full module path, e.g. factchecker.experiments.advocate_mediator_climatefeedback.advocate_mediator_climatefeedback")
        logger.error("Ensure the package is installed: pip install -e .")

    except Exception as e:
        logger.error("Error running experiment: %s", e)
        import traceback
        logger.error(traceback.format_exc())

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run fact-checking experiments. Use --experiment to choose which experiment module to run."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=DEFAULT_EXPERIMENT,
        metavar="MODULE",
        help="Experiment module path (default: %(default)s)",
    )

    parser.add_argument(
        "--node-batch-size", 
        type=int, 
        default=500,
        help="Batch size for node creation (default: 500)"
    )
    
    parser.add_argument(
        "--embedding-batch-size", 
        type=int, 
        default=32,
        help="Batch size for embedding (default: 32)"
    )
    
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=4,
        help="Number of worker threads (default: 4)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=150,
        help="Size of text chunks for indexing (default: 150)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=20,
        help="Overlap between chunks (default: 20)"
    )
    
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding the index even if it exists"
    )

    # Sources download (passed into experiment)
    parser.add_argument(
        "--sources-limit",
        type=int,
        default=None,
        metavar="N",
        help="Max number of sources to download (default: no limit)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download sources even if file already exists"
    )

    # Claim sampling (passed into experiment)
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        metavar="N",
        help="Number of claims to evaluate (default: use experiment default, e.g. 10)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    setup_environment()

    # Options passed into experiment when it accepts experiment_options
    experiment_options = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embed_batch_size": getattr(args, "embedding_batch_size", 32),
        "node_batch_size": getattr(args, "node_batch_size", None),
        "num_workers": getattr(args, "num_workers", None),
        "force_rebuild": args.force_rebuild,
        "sources_max_sources": args.sources_limit,
        "sources_skip_existing": not args.no_skip_existing,
        "total_samples": getattr(args, "samples", None),
    }
    experiment_options = {k: v for k, v in experiment_options.items() if v is not None}

    run_experiment(experiment_module=args.experiment, experiment_options=experiment_options) 