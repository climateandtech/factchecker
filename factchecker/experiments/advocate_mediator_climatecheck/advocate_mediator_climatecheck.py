import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from llama_index.core import Settings, Document
from datasets import load_dataset

from factchecker.strategies.advocate_mediator import AdvocateMediatorStrategy
from factchecker.utils.experiment_utils import (
    configure_logging,
    save_results,
    verify_environment,
)
from factchecker.experiments.advocate_mediator_climatecheck.climatecheck_sources import ClimateCheckSourcesManager
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.core.embeddings import load_embedding_model
from factchecker.core.llm import load_llm
from factchecker.experiments.advocate_mediator_climatecheck.specialized_advocate_configs import get_specialized_advocate_configs
from factchecker.experiments.advocate_mediator_climatecheck.climatecheck_parser import ClimateCheckParser
from factchecker.experiments.advocate_mediator_climatecheck.advocate_mediator_climatecheck_prompts import (
    claim_assessment_advocate_primer,
    claim_assessment_mediator_primer
)
from factchecker.steps.hyde_query_generator import HyDEQueryConfig
from factchecker.experiments.advocate_mediator_climatecheck.retrieval_metrics import RetrievalEvaluator

logger = logging.getLogger(__name__)

# Experiment Parameters
EXPERIMENT_PARAMS = {
    # Dataset parameters
    'num_samples': 1,  # Just test with one claim for now
    'num_papers': None,  # Only process 10 papers for testing
    
    # Document processing parameters
    'chunk_size': 200,  # Size of text chunks for indexing
    'chunk_overlap': 20,  # Overlap between chunks
    
    # Indexing parameters
    'main_source_directory': 'data/papers',  # Base directory for papers
    'index_name': 'climatecheck_index',  # Name for the persisted index
    'index_path': 'data/indices/climatecheck',  # Path to store the index
    'force_rebuild': False,  # Don't rebuild if index exists
    
    # Retrieval parameters
    'top_k': 8,  # Number of similar chunks to retrieve
    
    # LLM parameters
    'llm_type': 'ollama',  # Use Ollama
    'llm_model': 'qwen:14b',  # Use Qwen 2.5 14B
    'temperature': 0.1,  # Temperature for the LLM
    'context_window': 131072,  # Qwen 2.5 14B's actual context window size
    
    # Label options
    'label_options': ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO'],  # Updated to match dataset annotations
    
    # Metrics evaluation
    'compute_metrics': False,  # Disable metrics computation to respect num_papers setting
    'results_path': 'results/advocate_mediator_climatecheck.csv'  # Path to store the results data
}

# Setup signal handler for graceful termination
def handle_interrupt(signum, frame):
    """Handle interrupt signals gracefully."""
    print("\n\nReceived interrupt signal. Cleaning up and exiting...")
    # Log to both console and file
    logger.info("Process interrupted by user. Exiting gracefully.")
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, handle_interrupt)

def load_climatecheck_data(num_samples: int) -> List[Dict[str, Any]]:
    """Load and sample data from the ClimateCheck dataset."""
    logger.info("Loading claims from ClimateCheck dataset...")
    dataset = load_dataset("rabuahmad/climatecheck", split="train")
    logger.info(f"Found {len(dataset)} total claims in dataset")
    
    # Sample a subset of claims for testing
    logger.info(f"Sampling {num_samples} claims...")
    sampled_dataset = dataset.shuffle(seed=42).select(range(num_samples))
    
    # Convert to list of dicts with the fields we need
    claims = []
    for item in sampled_dataset:
        claims.append({
            'claim': item['claim'],
            'label': item['annotation'],  # Using annotation as the label
            'abstract_id': item['abstract_id'],
            'abstract': item['abstract']
        })
    
    # Log some stats about the claims
    label_counts = {}
    for claim in claims:
        label_counts[claim['label']] = label_counts.get(claim['label'], 0) + 1
    
    logger.info("Claim label distribution:")
    for label, count in label_counts.items():
        logger.info(f"  {label}: {count}")
    
    return claims

def setup_specialized_advocate_strategy(papers: Optional[List[Dict[str, Any]]] = None, 
                                        indexer_options: Optional[Dict[str, Any]] = None) -> AdvocateMediatorStrategy:
    """
    Set up the advocate strategy with specialized configurations for climate check claims.
    
    Args:
        papers: List of papers to use as sources (can be None if loading from disk)
        indexer_options: Optional customization for indexer settings
    
    Returns:
        AdvocateMediatorStrategy: Configured strategy with specialized advocates
    """
    # Configure the LLM
    Settings.llm = load_llm(
        llm_type=EXPERIMENT_PARAMS['llm_type'],
        model=EXPERIMENT_PARAMS['llm_model'],
        temperature=EXPERIMENT_PARAMS['temperature'],
        context_window=EXPERIMENT_PARAMS['context_window']
    )
    
    # Set index path
    index_path = EXPERIMENT_PARAMS['index_path']
    
    # Create base indexer options
    base_indexer_options = {
        'index_path': index_path,
        'show_progress': True,
    }
    
    # Merge with any custom indexer options
    if indexer_options:
        base_indexer_options.update(indexer_options)
    
    # Create shared indexer configuration
    shared_indexer = [{
        'type': 'llama_vector_store',
        'options': base_indexer_options
    }]
    
    # If papers are provided, convert them to documents (but don't build index yet)
    if papers:
        logger.info(f"Converting {len(papers)} papers to Documents...")
        # We'll create documents later in the main function
        logger.info("Documents will be created later")
    
    # Get advocate configs and add shared settings
    advocate_configs = []
    
    for config in get_specialized_advocate_configs(EXPERIMENT_PARAMS['label_options']):
        config['system_prompt'] = claim_assessment_advocate_primer
        
        # Add HyDE configuration with more aggressive settings
        config['hyde_config'] = HyDEQueryConfig(
            num_queries=2,  # Try up to 2 additional queries
            max_query_length=500,  # Allow detailed scientific queries
            temperature=0.8  # Slightly higher temperature for more diverse queries
        )
        
        # Add evidence options with high initial threshold
        config['evidence_options'] = {
            'min_score': 0.0,  # Set to 0.0 to disable filtering
            'min_evidence': 1  # Only require 1 piece since we have limited data
        }
        
        advocate_configs.append(config)
    
    # Create retriever options list - one per advocate
    retriever_options_list = [
        {'top_k': EXPERIMENT_PARAMS['top_k']} for _ in range(len(advocate_configs))
    ]
    
    # Create strategy with shared indexer and custom parser
    strategy = AdvocateMediatorStrategy(
        indexer_options_list=shared_indexer,  # Pass the shared indexer
        retriever_options_list=retriever_options_list,
        advocate_options_list=advocate_configs,  # Pass specialized configs
        evidence_options={
            'min_score': 0.0  # Set to 0.0 to disable filtering
        },
        mediator_options={
            'system_prompt': claim_assessment_mediator_primer,
            'parser': ClimateCheckParser()  # Use the custom XML parser
        }
    )
    
    return strategy

def main(indexer_options=None):
    """
    Run the ClimateCheck experiment with customizable indexer options.
    
    Args:
        indexer_options: Dictionary with custom options for the indexer configuration
    """
    # Configure logging
    configure_logging()

    # Create results directory if it doesn't exist
    results_path = EXPERIMENT_PARAMS.get('results_path', 'results/advocate_mediator_climatecheck.csv')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Determine if we should compute metrics
    compute_metrics = EXPERIMENT_PARAMS.get('compute_metrics', False)
    
    # If computing metrics, use all papers
    if compute_metrics:
        logger.info("Metrics computation enabled - using all available papers for indexing")
        original_num_papers = EXPERIMENT_PARAMS['num_papers'] 
        EXPERIMENT_PARAMS['num_papers'] = None  # Use all papers
    
    # Load paper abstracts
    logger.info("Loading paper abstracts...")
    sources_manager = ClimateCheckSourcesManager()
    try:
        # Create index directory if it doesn't exist
        os.makedirs(EXPERIMENT_PARAMS['index_path'], exist_ok=True)
        
        # Check if index exists
        index_exists = os.path.exists(os.path.join(EXPERIMENT_PARAMS['index_path'], 'docstore.json'))
        
        if index_exists and not EXPERIMENT_PARAMS['force_rebuild']:
            logger.info("Found existing index, loading...")
            strategy = setup_specialized_advocate_strategy(None, indexer_options)  # Pass None since we'll load from disk
            strategy.indexers[0].load_index()
            logger.info("Successfully loaded existing index")
        else:
            if index_exists:
                logger.info("Force rebuild enabled, creating new index...")
            else:
                logger.info("No existing index found, creating new one...")
                
            papers = sources_manager.get_paper_texts(num_samples=EXPERIMENT_PARAMS['num_papers'])
            
            if not papers:
                raise ValueError("No papers loaded from ClimateCheck sources")
            
            logger.info(f"Successfully loaded {len(papers)} papers")
            
            # Setup specialized advocate strategy
            logger.info("\nSetting up specialized advocate strategy...")
            strategy = setup_specialized_advocate_strategy(papers, indexer_options)
            
            # Save the index for future use
            logger.info("Saving index for future use...")
            # Build the index from the papers before saving it
            logger.info("Building index from papers...")
            
            # Create documents with explicitly generated unique IDs
            docs = []
            for i, paper in enumerate(papers):
                # Create a unique ID based on index to avoid any chance of duplicates
                doc_id = f"climate_paper_{i}"
                docs.append(Document(text=paper, id_=doc_id))
            
            logger.info(f"Created {len(docs)} documents with guaranteed unique IDs")
            
            # Show ways to monitor progress
            logger.info("TIP: To monitor embedding progress in real-time, run this in another terminal:")
            logger.info("  tail -f logs/embedding_progress.log")
            
            strategy.indexers[0].build_index(docs)
            strategy.indexers[0].save_index()  # Use the first indexer (there's only one in shared mode)
            logger.info("Index saved successfully")
        
        # Log the configuration
        logger.info("Specialized advocate strategy configuration:")
        logger.info(f"Chunk size: {Settings.chunk_size}")
        logger.info(f"Chunk overlap: {Settings.chunk_overlap}")
        logger.info(f"Top k: {EXPERIMENT_PARAMS['top_k']}")
        
    except Exception as e:
        logger.error(f"Error loading or indexing papers: {str(e)}")
        raise

    # Load sample claims
    logger.info("\nLoading claims from ClimateCheck dataset...")
    claims = load_climatecheck_data(EXPERIMENT_PARAMS['num_samples'])
    
    if not claims:
        raise ValueError("No claims loaded from ClimateCheck dataset")
    
    logger.info(f"Successfully loaded {len(claims)} claims")

    # Process claims
    results = []
    logger.info("\nProcessing claims...")
    for i, claim in enumerate(claims, 1):
        logger.info(f"\nProcessing claim {i}/{len(claims)}")
        logger.info(f"Claim text: {claim['claim'][:200]}...")
        
        try:
            # Evaluate claim using specialized advocates
            final_verdict, verdicts, reasonings = strategy.evaluate_claim(claim['claim'])
            logger.info(f"Final verdict: {final_verdict}")
            
            # Store results
            results.append({
                'claim': claim['claim'],
                'true_label': claim['label'],
                'predicted_label': final_verdict,
                'advocate_verdicts': verdicts,
                'advocate_reasonings': reasonings,
                'abstract_id': claim['abstract_id']  # Add abstract_id for retrieval metrics
            })
            
        except Exception as e:
            logger.error(f"Error processing claim: {str(e)}")
            continue

    # Save results
    if not results:
        logger.error("No results to save - all claims failed processing")
        return
        
    logger.info("\nSaving results...")
    results_df = pd.DataFrame(results)
    results_file = save_results(results_df, custom_path=EXPERIMENT_PARAMS.get('results_path'))
    logger.info(f"Results saved to: {results_file}")
    
    # Reset num_papers if it was changed
    if compute_metrics and 'original_num_papers' in locals():
        EXPERIMENT_PARAMS['num_papers'] = original_num_papers
    
    # Compute retrieval metrics if enabled
    if compute_metrics:
        logger.info("\nComputing retrieval metrics...")
        try:
            evaluator = RetrievalEvaluator()
            metrics = evaluator.evaluate_from_results_file(results_file)
            
            # Log the metrics
            logger.info("\nRetrieval Metrics:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{metric}: {value:.4f}")
                else:
                    logger.info(f"{metric}: {value}")
            
            logger.info(f"Detailed metrics saved to: {evaluator.results_path}")
        except Exception as e:
            logger.error(f"Error computing retrieval metrics: {str(e)}")
            logger.exception(e)

if __name__ == "__main__":
    main() 