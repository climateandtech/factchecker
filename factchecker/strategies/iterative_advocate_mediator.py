"""Iterative advocate-mediator strategy with dynamic paper expansion and domain specialization."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from math import ceil

from factchecker.strategies.advocate_mediator import AdvocateMediatorStrategy
from factchecker.steps.specialized_advocate import SpecializedAdvocate, DomainExpertise
from factchecker.clustering.paper_clustering import PaperClusterer
from factchecker.core.llm import load_llm

logger = logging.getLogger(__name__)

@dataclass
class EvaluationRound:
    """Represents a round of claim evaluation."""
    papers: List[str]
    verdict: str
    confidence: float
    reasoning: str
    missing_aspects: List[str]
    domain_distribution: Optional[Dict[str, int]] = None

class IterativeAdvocateMediatorStrategy:
    """
    An iterative strategy that uses specialized advocates for different domains.
    
    This strategy:
    1. Clusters papers by domain expertise
    2. Assigns papers to domain-specific advocates
    3. Analyzes results for information gaps
    4. Generates new queries to find papers filling those gaps
    5. Repeats the process until sufficient information is found or max rounds reached
    """
    
    def __init__(
        self,
        domains: List[DomainExpertise],
        advocate_mediator_strategy: AdvocateMediatorStrategy,
        max_rounds: int = 2,
        min_confidence: float = 0.8,
    ):
        """Initialize the iterative strategy.
        
        Args:
            domains: List of domain expertise configurations
            advocate_mediator_strategy: Base strategy to use for each round
            max_rounds: Maximum number of paper search rounds
            min_confidence: Minimum confidence threshold to accept verdict
        """
        self.domains = {d.name: d for d in domains}
        self.base_strategy = advocate_mediator_strategy
        self.max_rounds = max_rounds
        self.min_confidence = min_confidence
        self.llm = load_llm()
        
        # Initialize paper clusterer
        self.clusterer = PaperClusterer()
        
        # Create specialized advocates
        self.advocates = {
            name: SpecializedAdvocate(
                retriever=self.base_strategy.retrievers[0],  # Use first retriever as base
                domain_expertise=domain,
                options=self.base_strategy.advocate_steps[0].options,  # Use first advocate's options as base
                evidence_options=self.base_strategy.advocate_steps[0].evidence_options
            )
            for name, domain in self.domains.items()
        }
    
    def distribute_papers(self, papers: List[str]) -> Dict[str, List[str]]:
        """Distribute papers to advocates based on domain expertise.
        
        Args:
            papers: List of papers to distribute
            
        Returns:
            Dictionary mapping domain names to lists of papers
        """
        # Create domain descriptions for clustering
        domain_descriptions = {
            name: f"{d.description}\nKeywords: {', '.join(d.keywords)}"
            for name, d in self.domains.items()
        }
        
        # Cluster papers by domain
        cluster_result = self.clusterer.cluster_papers(papers, domain_descriptions)
        
        return cluster_result.domain_papers
    
    def analyze_missing_information(self, round_result: EvaluationRound) -> List[str]:
        """Analyze what information is missing from current evaluation.
        
        Args:
            round_result: Results from the current evaluation round
            
        Returns:
            List of missing aspects that need more research
        """
        analysis_prompt = """Analyze this fact-checking result and identify what additional information would be helpful to increase confidence. Focus on:
1. Missing temporal aspects (e.g., more recent data)
2. Missing geographical regions
3. Missing methodological approaches
4. Missing counter-arguments or alternative explanations
5. Missing specific measurements or metrics

Current domain distribution:
{domain_distribution}

Verdict: {verdict}
Confidence: {confidence}
Reasoning: {reasoning}

List ONLY the specific types of information needed, one per line. If no additional information is needed, return "NONE".
"""
        
        try:
            domain_dist_str = "\n".join(
                f"- {domain}: {count} papers"
                for domain, count in round_result.domain_distribution.items()
            ) if round_result.domain_distribution else "No domain distribution available"
            
            response = self.llm.complete(
                analysis_prompt.format(
                    domain_distribution=domain_dist_str,
                    verdict=round_result.verdict,
                    confidence=round_result.confidence,
                    reasoning=round_result.reasoning
                )
            )
            
            missing_aspects = [
                aspect.strip()
                for aspect in response.text.split('\n')
                if aspect.strip() and aspect.strip() != 'NONE'
            ]
            
            return missing_aspects
        except Exception as e:
            logger.error(f"Error analyzing missing information: {e}")
            return []
    
    def generate_search_queries(self, claim: str, missing_aspects: List[str]) -> List[str]:
        """Generate new search queries based on missing information.
        
        Args:
            claim: The original claim being evaluated
            missing_aspects: List of missing information aspects
            
        Returns:
            List of search queries targeting the missing information
        """
        query_prompt = """Generate specific search queries to find scientific papers about missing aspects of this claim.
Each query should be focused and use technical/scientific terminology.

Claim: {claim}

Missing aspects:
{aspects}

Available domains:
{domains}

Generate one search query per aspect. Format each as a single line of text optimized for scientific paper search.
Consider the available domains when generating queries.
"""
        
        try:
            domains_str = "\n".join(
                f"- {name}: {d.description}"
                for name, d in self.domains.items()
            )
            
            response = self.llm.complete(
                query_prompt.format(
                    claim=claim,
                    aspects='\n'.join(f"- {aspect}" for aspect in missing_aspects),
                    domains=domains_str
                )
            )
            
            return [
                query.strip()
                for query in response.text.split('\n')
                if query.strip()
            ]
        except Exception as e:
            logger.error(f"Error generating search queries: {e}")
            return []
    
    def evaluate_claim(self, claim: str, initial_papers: List[str]) -> Dict[str, Any]:
        """Evaluate a claim through multiple rounds if needed.
        
        Args:
            claim: The claim to evaluate
            initial_papers: Initial set of papers to evaluate
            
        Returns:
            Final evaluation result
        """
        all_papers = initial_papers.copy()
        evaluation_rounds = []
        
        for round_num in range(self.max_rounds):
            logger.info(f"Starting evaluation round {round_num + 1}")
            
            # Distribute papers among domain-specific advocates
            domain_papers = self.distribute_papers(all_papers)
            
            # Run evaluation with each domain advocate
            advocate_results = []
            for domain_name, papers in domain_papers.items():
                if papers:  # Only evaluate if domain has papers
                    advocate = self.advocates[domain_name]
                    result = advocate.evaluate_claim(claim)
                    advocate_results.append(result)
            
            # Combine results through mediator
            round_result = self.base_strategy.combine_results(advocate_results)
            
            # Create round result object with domain distribution
            current_round = EvaluationRound(
                papers=all_papers,
                verdict=round_result['verdict'],
                confidence=round_result.get('confidence', 0.0),
                reasoning=round_result.get('reasoning', ''),
                missing_aspects=[],
                domain_distribution={name: len(papers) for name, papers in domain_papers.items()}
            )
            
            # Check if we need more information
            if current_round.confidence >= self.min_confidence:
                logger.info(f"Sufficient confidence ({current_round.confidence}) achieved in round {round_num + 1}")
                break
                
            # Analyze what information is missing
            current_round.missing_aspects = self.analyze_missing_information(current_round)
            if not current_round.missing_aspects:
                logger.info("No missing aspects identified")
                break
                
            # Generate new search queries
            new_queries = self.generate_search_queries(claim, current_round.missing_aspects)
            
            # Get new papers using the queries
            new_papers = []
            for query in new_queries:
                # Use the base strategy's paper selection mechanism
                selected_papers = self.base_strategy.select_papers(query)
                new_papers.extend(selected_papers)
            
            # Add new papers to the pool
            all_papers.extend(new_papers)
            evaluation_rounds.append(current_round)
            
            logger.info(f"Added {len(new_papers)} new papers for round {round_num + 2}")
        
        # Return final evaluation with history
        return {
            'verdict': evaluation_rounds[-1].verdict if evaluation_rounds else round_result['verdict'],
            'confidence': evaluation_rounds[-1].confidence if evaluation_rounds else round_result.get('confidence', 0.0),
            'reasoning': evaluation_rounds[-1].reasoning if evaluation_rounds else round_result.get('reasoning', ''),
            'papers': all_papers,
            'rounds': len(evaluation_rounds) + 1,
            'evaluation_history': [vars(round) for round in evaluation_rounds],
            'domain_distribution': evaluation_rounds[-1].domain_distribution if evaluation_rounds else None
        } 