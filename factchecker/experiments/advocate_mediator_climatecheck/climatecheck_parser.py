"""Parser for ClimateCheck experiment responses."""

import re
import logging
from typing import Optional, Dict, Any, List, Tuple
from factchecker.parsing.response_parsers import ResponseParser

logger = logging.getLogger(__name__)

class ClimateCheckParser(ResponseParser):
    """Parser for ClimateCheck experiment that extracts paper relevance and verdicts."""
    
    def __init__(self, min_paper_score: float = 8.0):
        """Initialize the parser.
        
        Args:
            min_paper_score: Minimum paper relevance score (0-10) to include in final evidence
        """
        self._last_papers = []
        self.min_paper_score = min_paper_score
    
    def parse_verdict(self, response_content: str) -> Optional[str]:
        """Parse verdict and paper analysis from response."""
        # Extract paper analysis
        papers = []
        paper_matches = re.finditer(
            r'<paper id="([^"]*)" relevance="([^"]*)">\s*([^<]*)\s*</paper>',
            response_content
        )
        
        # Log paper scores section header
        logger.info("\nPaper Relevance Scores:")
        
        for match in paper_matches:
            paper_id = match.group(1)
            relevance = float(match.group(2))
            explanation = match.group(3).strip()
            papers.append({
                'paper_id': paper_id,
                'relevance': relevance,
                'explanation': explanation
            })
            # Log each paper's score and explanation
            logger.info(f"Paper {paper_id}: Score={relevance:.1f} - {explanation}")
        
        self._last_papers = papers
        
        # Extract verdict
        verdict_match = re.search(r'<verdict>(.*?)</verdict>', response_content)
        if not verdict_match:
            logger.warning("No verdict XML tags found in response")
            return None
            
        verdict = verdict_match.group(1).strip().upper()
        logger.info(f"\nParsed verdict: {verdict}")
        
        return verdict
    
    def get_last_paper_analysis(self) -> List[Dict[str, Any]]:
        """Get the paper analysis from the last parsed response."""
        return self._last_papers

    def filter_evidence_by_scores(self, evidence_list: List[str]) -> List[str]:
        """Filter evidence chunks based on paper relevance scores.

        Args:
            evidence_list: List of evidence chunks

        Returns:
            List of evidence chunks from papers that meet the minimum score threshold
        """
        if not self._last_papers:
            logger.warning("No paper scores available for filtering")
            return evidence_list

        # Get IDs of papers that meet the threshold
        high_scoring_papers = {
            paper['paper_id'] 
            for paper in self._last_papers 
            if paper['relevance'] >= self.min_paper_score
        }

        if not high_scoring_papers:
            logger.warning(f"No papers met the minimum score threshold of {self.min_paper_score}")
            return []

        # Filter evidence chunks
        filtered_evidence = []
        for evidence in evidence_list:
            # Try to extract paper ID from the evidence metadata
            # This assumes the paper ID is somewhere in the evidence text or metadata
            for paper_id in high_scoring_papers:
                if paper_id in evidence:  # This is a simple check - adjust based on your actual structure
                    filtered_evidence.append(evidence)
                    break

        logger.info(f"Filtered evidence from {len(evidence_list)} chunks to {len(filtered_evidence)} chunks from high-scoring papers")
        return filtered_evidence

    def get_high_scoring_papers(self) -> List[Dict[str, Any]]:
        """Get papers that meet the minimum score threshold.

        Returns:
            List of paper dictionaries with scores >= min_paper_score
        """
        return [
            paper for paper in self._last_papers 
            if paper['relevance'] >= self.min_paper_score
        ] 