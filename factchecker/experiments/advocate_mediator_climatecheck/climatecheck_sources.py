"""Module for managing papers from the ClimateCheck dataset."""

import logging
import os
from typing import Dict, List, Optional
from datasets import load_dataset

logger = logging.getLogger(__name__)

class ClimateCheckSourcesManager:
    """Manager for organizing ClimateCheck papers and abstracts."""
    
    def __init__(self, base_output_folder: str = "data/papers") -> None:
        """Initialize the ClimateCheck sources manager.
        
        Args:
            base_output_folder (str): Base directory for storing paper information.
        """
        self.base_output_folder = base_output_folder
        os.makedirs(base_output_folder, exist_ok=True)
        logger.info(f"Initialized ClimateCheckSourcesManager with output folder: {base_output_folder}")
    
    def load_papers(self, num_samples: Optional[int] = None) -> List[Dict]:
        """Load papers from the ClimateCheck publications corpus.
        
        Args:
            num_samples (Optional[int]): Number of papers to load. If None, loads all.
            
        Returns:
            List[Dict]: List of paper information including abstracts.
        """
        logger.info("Loading papers from ClimateCheck publications corpus...")
        try:
            # Load papers from ClimateCheck dataset
            publications = load_dataset("rabuahmad/climatecheck_publications_corpus", split="train")
            logger.info(f"Successfully loaded dataset with {len(publications)} total papers")
            
            # Sample if requested
            if num_samples is not None:
                logger.info(f"Sampling {num_samples} papers from dataset...")
                publications = publications.shuffle(seed=42).select(range(num_samples))
                logger.info(f"Selected {len(publications)} papers after sampling")
            
            # Convert to list of dicts and filter for papers with abstracts
            papers = []
            papers_without_abstracts = 0
            total_processed = 0
            
            for pub in publications:
                total_processed += 1
                if pub['abstract'] and len(pub['abstract'].strip()) > 0:  # Only include papers with non-empty abstracts
                    papers.append({
                        'id': pub['s2orc_id'],
                        'title': pub['title'],
                        'abstract': pub['abstract']
                    })
                else:
                    papers_without_abstracts += 1
                
                if total_processed % 100 == 0:
                    logger.debug(f"Processed {total_processed} papers...")
            
            logger.info(f"Successfully loaded {len(papers)} papers with valid abstracts")
            logger.info(f"Skipped {papers_without_abstracts} papers without abstracts")
            
            # If we have more papers than requested, sample again
            if num_samples is not None and len(papers) > num_samples:
                logger.info(f"Sampling {num_samples} papers from {len(papers)} papers with abstracts...")
                import random
                random.seed(42)
                papers = random.sample(papers, num_samples)
                logger.info(f"Final number of papers after sampling: {len(papers)}")
            
            # Log some statistics about the papers
            if papers:
                abstract_lengths = [len(p['abstract']) for p in papers]
                avg_length = sum(abstract_lengths) / len(abstract_lengths)
                logger.info(f"Average abstract length: {avg_length:.1f} characters")
                logger.info(f"Sample paper title: {papers[0]['title']}")
                logger.info(f"Sample abstract preview: {papers[0]['abstract'][:200]}...")
            
            return papers
            
        except Exception as e:
            logger.error(f"Error loading papers: {str(e)}")
            raise
    
    def get_paper_texts(self, num_samples: Optional[int] = None) -> List[str]:
        """Get paper abstracts.
        
        Args:
            num_samples (Optional[int]): Number of papers to load. If None, loads all.
            
        Returns:
            List[str]: List of paper abstracts.
        """
        papers = self.load_papers(num_samples=num_samples)
        abstracts = [p['abstract'] for p in papers]
        
        if not abstracts:
            logger.error("No abstracts found in loaded papers")
            raise ValueError("No abstracts found in loaded papers")
            
        logger.info(f"Successfully extracted {len(abstracts)} paper abstracts")
        logger.info(f"Abstract lengths: min={min(len(a) for a in abstracts)}, "
                   f"max={max(len(a) for a in abstracts)}, "
                   f"avg={sum(len(a) for a in abstracts)/len(abstracts):.1f}")
        
        return abstracts 