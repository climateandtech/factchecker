import logging
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support

from factchecker.utils.experiment_utils import configure_logging
from factchecker.experiments.advocate_mediator_climatecheck.climatecheck_sources import ClimateCheckSourcesManager
from factchecker.experiments.advocate_mediator_climatecheck.climatecheck_parser import ClimateCheckParser

logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """
    Evaluator for ClimateCheck paper retrieval metrics based on advocate XML outputs.
    Computes standard IR metrics (Precision, Recall, NDCG, MRR).
    """
    
    def __init__(self, results_path: str = None):
        """
        Initialize retrieval evaluator.
        
        Args:
            results_path: Path to save results (if None, uses default)
        """
        self.results_path = results_path or 'results/retrieval_metrics.json'
        self.parser = ClimateCheckParser()
        
    def calculate_precision_at_k(self, relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
        """
        Calculate precision at k.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs in ranked order
            k: Number of top documents to consider
            
        Returns:
            Precision at k
        """
        if not retrieved_docs or k <= 0:
            return 0.0
            
        top_k_docs = retrieved_docs[:min(k, len(retrieved_docs))]
        relevant_in_top_k = sum(1 for doc in top_k_docs if doc in relevant_docs)
        
        return relevant_in_top_k / len(top_k_docs)
    
    def calculate_recall_at_k(self, relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
        """
        Calculate recall at k.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs in ranked order
            k: Number of top documents to consider
            
        Returns:
            Recall at k
        """
        if not relevant_docs or not retrieved_docs or k <= 0:
            return 0.0
            
        top_k_docs = retrieved_docs[:min(k, len(retrieved_docs))]
        relevant_in_top_k = sum(1 for doc in top_k_docs if doc in relevant_docs)
        
        return relevant_in_top_k / len(relevant_docs)
    
    def calculate_average_precision(self, relevant_docs: List[str], retrieved_docs: List[str]) -> float:
        """
        Calculate average precision.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs in ranked order
            
        Returns:
            Average precision
        """
        if not relevant_docs or not retrieved_docs:
            return 0.0
            
        precisions = []
        relevant_found = 0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precisions.append(precision_at_i)
                
        if not precisions:
            return 0.0
            
        return sum(precisions) / len(relevant_docs)
    
    def calculate_ndcg_at_k(self, relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
        """
        Calculate normalized discounted cumulative gain at k.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs in ranked order
            k: Number of top documents to consider
            
        Returns:
            NDCG at k
        """
        if not relevant_docs or not retrieved_docs or k <= 0:
            return 0.0
            
        # Use binary relevance (1 for relevant, 0 for not relevant)
        relevance_scores = [1 if doc in relevant_docs else 0 for doc in retrieved_docs[:k]]
        
        # Calculate DCG
        dcg = relevance_scores[0] if relevance_scores else 0
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 1 + 1)  # +1 for 1-indexing
            
        # Calculate ideal DCG (IDCG)
        ideal_relevance = sorted([1 if doc in relevant_docs else 0 for doc in retrieved_docs[:k]], reverse=True)
        idcg = ideal_relevance[0] if ideal_relevance else 0
        for i in range(1, len(ideal_relevance)):
            idcg += ideal_relevance[i] / np.log2(i + 1 + 1)  # +1 for 1-indexing
            
        # Return NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_mrr(self, relevant_docs: List[str], retrieved_docs: List[str]) -> float:
        """
        Calculate mean reciprocal rank.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs in ranked order
            
        Returns:
            Mean reciprocal rank
        """
        if not relevant_docs or not retrieved_docs:
            return 0.0
            
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
                
        return 0.0
    
    def evaluate_retrieval(self, claim_responses: List[Dict[str, Any]], all_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate retrieval performance from claim responses.
        
        Args:
            claim_responses: List of dictionaries containing claim responses with paper analyses
            all_claims: List of all claims with ground truth data
            
        Returns:
            Dictionary of aggregated metrics
        """
        # Create lookup for ground truth data
        claim_abstract_lookup = {claim['claim']: claim['abstract_id'] for claim in all_claims}
        
        metrics_by_claim = []
        
        for response in claim_responses:
            claim_text = response['claim']
            paper_analysis = response['paper_analysis']
            
            if not paper_analysis:
                logger.warning(f"No paper analysis found for claim: {claim_text[:50]}...")
                continue
                
            # Extract ranked list of retrieved documents
            retrieved_docs = [paper['paper_id'] for paper in sorted(
                paper_analysis, 
                key=lambda x: x['relevance'], 
                reverse=True
            )]
            
            # Get ground truth relevant document(s)
            relevant_docs = []
            
            # First try to get abstract_id from the response if available
            if response.get('abstract_id'):
                relevant_docs = [response['abstract_id']]
            # Otherwise try to look it up from claim text
            else:
                relevant_doc = claim_abstract_lookup.get(claim_text)
                if relevant_doc:
                    relevant_docs = [relevant_doc]
            
            if not relevant_docs or not relevant_docs[0]:
                logger.warning(f"No ground truth abstract found for claim: {claim_text[:50]}...")
                continue
                
            # Calculate metrics
            claim_metrics = {
                'claim': claim_text,
                'retrieved_docs': retrieved_docs,
                'relevant_docs': relevant_docs,
                'precision@1': self.calculate_precision_at_k(relevant_docs, retrieved_docs, 1),
                'precision@3': self.calculate_precision_at_k(relevant_docs, retrieved_docs, 3),
                'precision@5': self.calculate_precision_at_k(relevant_docs, retrieved_docs, 5),
                'precision@10': self.calculate_precision_at_k(relevant_docs, retrieved_docs, 10),
                'recall@1': self.calculate_recall_at_k(relevant_docs, retrieved_docs, 1),
                'recall@3': self.calculate_recall_at_k(relevant_docs, retrieved_docs, 3),
                'recall@5': self.calculate_recall_at_k(relevant_docs, retrieved_docs, 5),
                'recall@10': self.calculate_recall_at_k(relevant_docs, retrieved_docs, 10),
                'average_precision': self.calculate_average_precision(relevant_docs, retrieved_docs),
                'ndcg@1': self.calculate_ndcg_at_k(relevant_docs, retrieved_docs, 1),
                'ndcg@3': self.calculate_ndcg_at_k(relevant_docs, retrieved_docs, 3),
                'ndcg@5': self.calculate_ndcg_at_k(relevant_docs, retrieved_docs, 5),
                'ndcg@10': self.calculate_ndcg_at_k(relevant_docs, retrieved_docs, 10),
                'mrr': self.calculate_mrr(relevant_docs, retrieved_docs)
            }
            
            metrics_by_claim.append(claim_metrics)
            
        # Calculate aggregate metrics
        if not metrics_by_claim:
            logger.error("No metrics calculated - no valid claims processed")
            return {}
            
        aggregate_metrics = {
            'num_claims': len(metrics_by_claim),
            'mean_precision@1': np.mean([m['precision@1'] for m in metrics_by_claim]),
            'mean_precision@3': np.mean([m['precision@3'] for m in metrics_by_claim]),
            'mean_precision@5': np.mean([m['precision@5'] for m in metrics_by_claim]),
            'mean_precision@10': np.mean([m['precision@10'] for m in metrics_by_claim]),
            'mean_recall@1': np.mean([m['recall@1'] for m in metrics_by_claim]),
            'mean_recall@3': np.mean([m['recall@3'] for m in metrics_by_claim]),
            'mean_recall@5': np.mean([m['recall@5'] for m in metrics_by_claim]),
            'mean_recall@10': np.mean([m['recall@10'] for m in metrics_by_claim]),
            'mean_average_precision': np.mean([m['average_precision'] for m in metrics_by_claim]),
            'mean_ndcg@1': np.mean([m['ndcg@1'] for m in metrics_by_claim]),
            'mean_ndcg@3': np.mean([m['ndcg@3'] for m in metrics_by_claim]),
            'mean_ndcg@5': np.mean([m['ndcg@5'] for m in metrics_by_claim]),
            'mean_ndcg@10': np.mean([m['ndcg@10'] for m in metrics_by_claim]),
            'mean_mrr': np.mean([m['mrr'] for m in metrics_by_claim]),
        }
        
        # Save detailed and aggregate metrics
        results = {
            'aggregate_metrics': aggregate_metrics,
            'metrics_by_claim': metrics_by_claim
        }
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        
        # Save results
        with open(self.results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Saved retrieval metrics to {self.results_path}")
        
        return aggregate_metrics
    
    def evaluate_from_results_file(self, results_file: str, claims_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate retrieval from a results file.
        
        Args:
            results_file: Path to results file with claims and parser outputs
            claims_limit: Limit number of claims to process (for testing)
            
        Returns:
            Dictionary of aggregated metrics
        """
        # Load results
        results_df = pd.read_csv(results_file)
        
        # Limit claims if specified
        if claims_limit:
            results_df = results_df.head(claims_limit)
        
        # Load all claims for ground truth - Import locally to avoid circular imports
        from datasets import load_dataset
        
        # Load ClimateCheck dataset
        logger.info("Loading claims from ClimateCheck dataset for ground truth...")
        dataset = load_dataset("rabuahmad/climatecheck", split="train")
        
        # Convert to list of dicts with the fields we need
        all_claims = []
        for item in dataset:
            all_claims.append({
                'claim': item['claim'],
                'label': item['annotation'],  # Using annotation as the label
                'abstract_id': item['abstract_id'],
                'abstract': item['abstract']
            })
        
        logger.info(f"Loaded {len(all_claims)} claims for ground truth")
        
        # Process each claim and collect paper analyses
        claim_responses = []
        
        for _, row in results_df.iterrows():
            # Extract the raw LLM response
            try:
                # For single string responses
                if isinstance(row.get('advocate_reasonings'), str):
                    response_content = row.get('advocate_reasonings', '')
                    if not response_content:
                        logger.warning(f"No advocate reasoning found for claim: {row['claim'][:50]}...")
                        continue
                        
                    # Parse the response to extract paper analysis
                    verdict = self.parser.parse_verdict(response_content)
                    paper_analysis = self.parser.get_last_paper_analysis()
                    
                # For list-type responses (stored as string representation)
                elif row.get('advocate_reasonings') and '[' in str(row.get('advocate_reasonings')):
                    import ast
                    # Try to safely evaluate the string as a Python list
                    try:
                        reasonings_list = ast.literal_eval(row.get('advocate_reasonings'))
                        for reasoning in reasonings_list:
                            if reasoning:
                                verdict = self.parser.parse_verdict(reasoning)
                                paper_analysis = self.parser.get_last_paper_analysis()
                                if paper_analysis:
                                    break
                    except:
                        logger.warning(f"Failed to parse advocate_reasonings as a list for claim: {row['claim'][:50]}...")
                        continue
                
                # Extract 'advocate_reasonings_1', 'advocate_reasonings_2', etc. if they exist
                else:
                    # Look for fields like advocate_reasonings_1, advocate_reasonings_2, etc.
                    paper_analysis = None
                    for col in row.index:
                        if col.startswith('advocate_reasonings_') and row[col]:
                            verdict = self.parser.parse_verdict(row[col])
                            paper_analysis = self.parser.get_last_paper_analysis()
                            if paper_analysis:
                                break
                
                if not paper_analysis:
                    logger.warning(f"No paper analysis extracted for claim: {row['claim'][:50]}...")
                    continue
                    
                claim_responses.append({
                    'claim': row['claim'],
                    'paper_analysis': paper_analysis,
                    'verdict': verdict,
                    'abstract_id': row.get('abstract_id')  # Include abstract_id if available
                })
            except Exception as e:
                logger.warning(f"Error processing claim: {str(e)}")
                continue
        
        # Evaluate retrieval
        return self.evaluate_retrieval(claim_responses, all_claims)
    
    def evaluate_interactive(self, num_claims: int = 10) -> Dict[str, Any]:
        """
        Run interactive evaluation of retrieval.
        
        Args:
            num_claims: Number of claims to evaluate
            
        Returns:
            Dictionary of aggregated metrics
        """
        # Import locally to avoid circular imports
        from datasets import load_dataset
        
        # Load ClimateCheck dataset
        logger.info("Loading claims from ClimateCheck dataset...")
        dataset = load_dataset("rabuahmad/climatecheck", split="train")
        
        # Convert to list of dicts with the fields we need
        all_claims = []
        for item in dataset:
            all_claims.append({
                'claim': item['claim'],
                'label': item['annotation'],  # Using annotation as the label
                'abstract_id': item['abstract_id'],
                'abstract': item['abstract']
            })
        
        claims_to_evaluate = all_claims[:num_claims]
        
        # Set up sources manager for paper retrieval
        sources_manager = ClimateCheckSourcesManager()
        
        # Load our current implementation
        from factchecker.experiments.advocate_mediator_climatecheck.advocate_mediator_climatecheck import (
            setup_specialized_advocate_strategy,
            EXPERIMENT_PARAMS
        )
        
        # Force rebuild the index with all papers, not just 50
        original_num_papers = EXPERIMENT_PARAMS['num_papers']
        EXPERIMENT_PARAMS['num_papers'] = None  # Use all papers
        EXPERIMENT_PARAMS['force_rebuild'] = True
        
        # Create the index and strategy
        os.makedirs(EXPERIMENT_PARAMS['index_path'], exist_ok=True)
        papers = sources_manager.get_paper_texts(num_samples=EXPERIMENT_PARAMS['num_papers'])
        strategy = setup_specialized_advocate_strategy(papers)
        
        # Process claims and collect responses
        claim_responses = []
        for claim_data in tqdm(claims_to_evaluate, desc="Evaluating claims"):
            try:
                # Evaluate claim using specialized advocates
                final_verdict, verdicts, reasonings = strategy.evaluate_claim(claim_data['claim'])
                
                # Parse the responses to extract paper analyses
                for i, reasoning in enumerate(reasonings):
                    self.parser.parse_verdict(reasoning)
                    paper_analysis = self.parser.get_last_paper_analysis()
                    
                    if paper_analysis:
                        claim_responses.append({
                            'claim': claim_data['claim'],
                            'paper_analysis': paper_analysis,
                            'verdict': verdicts[i] if i < len(verdicts) else None
                        })
                
            except Exception as e:
                logger.error(f"Error processing claim: {str(e)}")
                continue
        
        # Reset the experiment parameters
        EXPERIMENT_PARAMS['num_papers'] = original_num_papers
        EXPERIMENT_PARAMS['force_rebuild'] = False
        
        # Evaluate retrieval
        return self.evaluate_retrieval(claim_responses, all_claims)

def main():
    """Main function to run retrieval evaluation."""
    # Configure logging
    configure_logging()
    
    # Create evaluator
    evaluator = RetrievalEvaluator()
    
    # Run evaluation
    logger.info("Starting retrieval evaluation...")
    
    # Option 1: Evaluate from existing results file
    # metrics = evaluator.evaluate_from_results_file('results/advocate_mediator_climatecheck.csv')
    
    # Option 2: Run interactive evaluation
    metrics = evaluator.evaluate_interactive(num_claims=5)  # Start with small number for testing
    
    # Display metrics
    logger.info("\nRetrieval Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 