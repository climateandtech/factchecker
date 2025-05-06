import logging
from typing import List, Dict, Any, Optional

from factchecker.retrieval.abstract_retriever import AbstractRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore, TextNode
from factchecker.core.llm import load_llm

logger = logging.getLogger(__name__)

class EvidenceStep:
    """
    A step in the fact-checking process that gathers evidence for claims.
    
    This class handles the retrieval of relevant evidence from a document store,
    using similarity-based search to find potential matches.
    """

    def __init__(self, retriever: AbstractRetriever, options: dict = None) -> None:
        """
        Initialize an EvidenceStep instance.

        Args:
            retriever (AbstractRetriever): The retriever instance for finding relevant documents
            options (dict, optional): Configuration options including:
                - query_template: Template for formatting the search query
                - top_k: Number of top results to retrieve
                - min_score: Minimum similarity score for retriever (not for filtering)
        """
        self.retriever = retriever
        self.options = options if options is not None else {}
        self.query_template = self.options.pop('query_template', "{claim}")
        
        # Pass min_score to retriever if specified
        if 'min_score' in self.options:
            self.retriever.options['min_score'] = self.options.pop('min_score')

    def build_query(self, claim: str) -> str:
        """
        Build a search query from a claim using the configured template.

        Args:
            claim (str): The claim to build a query for

        Returns:
            str: The formatted search query
        """
        return self.query_template.format(claim=claim)

    def gather_evidence(self, claim: str) -> List[str]:
        """
        Gather evidence relevant to a given claim.

        Args:
            claim (str): The claim to gather evidence for

        Returns:
            list[str]: List of evidence texts found by the retriever
        """
        try:
            query = self.build_query(claim)
            evidence = self.retriever.retrieve(query)

            # Check if evidence is a list and has required attributes
            if not isinstance(evidence, list):
                logger.error("Expected evidence to be a list, but got %s. Returning empty list.", type(evidence))
                return []
            
            if not evidence:
                logger.info("No evidence found for claim")
                return []

            # Log evidence scores for debugging
            logger.info(f"Retrieved {len(evidence)} evidence nodes for claim: {claim}")
            for i, item in enumerate(evidence):
                score = getattr(item, 'score', None) or (
                    getattr(item.node, 'score', None) if hasattr(item, 'node') else None
                )
                logger.debug(f"Evidence {i+1} score: {score}")

            # Convert evidence to text list
            evidence_texts = []
            for item in evidence:
                if hasattr(item, 'text'):
                    evidence_texts.append(item.text)
                elif hasattr(item, 'node') and hasattr(item.node, 'text'):
                    evidence_texts.append(item.node.text)
                else:
                    logger.warning(f"Skipping evidence item without text attribute: {item}")

            return evidence_texts

        except Exception as e:
            logger.error(f"Error gathering evidence: {str(e)}")
            return []
