import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
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
from factchecker.experiments.advocate_mediator_climatecheck.specialized_advocate_configs import get_specialized_advocate_configs
from factchecker.experiments.advocate_mediator_climatecheck.climatecheck_parser import ClimateCheckParser
from factchecker.experiments.advocate_mediator_climatecheck.advocate_mediator_climatecheck_prompts import (
    claim_assessment_advocate_primer,
    claim_assessment_mediator_primer
)
from factchecker.steps.hyde_query_generator import HyDEQueryConfig

logger = logging.getLogger(__name__)

# Experiment Parameters
EXPERIMENT_PARAMS = {
    # Dataset parameters
    'num_samples': 5,  # Small subset for initial testing
    'num_papers': 50,  # Start with 50 papers for testing persistence
    
    # Document processing parameters
    'chunk_size': 150,  # Size of text chunks for indexing
    'chunk_overlap': 20,  # Overlap between chunks
    
    # Indexing parameters
    'main_source_directory': 'data/papers',  # Base directory for papers
    'index_name': 'climatecheck_index',  # Name for the persisted index
    'index_path': 'data/indices/climatecheck',  # Path to store the index
    'force_rebuild': False,  # Whether to force rebuild the index even if it exists
    
    # Retrieval parameters
    'top_k': 8,  # Number of similar chunks to retrieve
    
    # Label options
    'label_options': ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO'],  # Updated to match dataset annotations
}

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

def setup_specialized_advocate_strategy(papers: Optional[List[Dict[str, Any]]] = None) -> AdvocateMediatorStrategy:
    """Sets up the advocate-mediator strategy with specialized climate experts.
    
    Args:
        papers: Optional list of papers to index. If None, assumes we're loading from disk.
    """
    # Create shared indexer instance with appropriate options
    indexer_options = {
        'index_name': EXPERIMENT_PARAMS['index_name'],
        'index_path': EXPERIMENT_PARAMS['index_path'],
        'chunk_size': EXPERIMENT_PARAMS['chunk_size'],
        'chunk_overlap': EXPERIMENT_PARAMS['chunk_overlap']
    }
    
    if papers:
        indexer_options['documents'] = papers
        
    shared_indexer = LlamaVectorStoreIndexer(indexer_options)
    
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
            'min_score': 0.7,  # High threshold for quality matches
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
            'min_score': 0.7  # Keep high threshold for quality matches
        },
        mediator_options={
            'system_prompt': claim_assessment_mediator_primer,
            'parser': ClimateCheckParser()  # Use the custom XML parser
        }
    )
    
    return strategy

def main():
    # Configure logging
    configure_logging()

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
            strategy = setup_specialized_advocate_strategy(None)  # Pass None since we'll load from disk
            strategy.indexer.load_index()
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
            strategy = setup_specialized_advocate_strategy(papers)
            
            # Save the index for future use
            logger.info("Saving index for future use...")
            strategy.indexer.save_index()
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
                'advocate_reasonings': reasonings
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
    save_results(results_df)
    logger.info("Results saved successfully")

if __name__ == "__main__":
    main() 