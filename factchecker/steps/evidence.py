import logging

from factchecker.retrieval.abstract_retriever import AbstractRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore
from factchecker.core.llm import load_llm

class EvidenceStep:
    """
    A step in the fact-checking process that gathers and classifies evidence for claims.
    
    This class handles the retrieval and filtering of relevant evidence from a document store,
    using similarity-based search and post-processing to ensure quality results.
    """

    def __init__(self, retriever: AbstractRetriever, options: dict =None) -> None:
        """
        Initialize an EvidenceStep instance.

        Args:
            retriever (AbstractRetriever): The retriever instance for finding relevant documents
            options (dict, optional): Configuration options including:
                - query_template: Template for formatting the search query
                - top_k: Number of top results to retrieve
                - min_score: Minimum similarity score threshold
        """
        self.retriever = retriever
        self.options = options if options is not None else {}
        # Extract specific options and remove them from the options hash
        # The query_template can be customized to target specific types of evidence,
        # e.g. "evidence for: {claim}" for supporting evidence or "evidence against: {claim}" 
        # for contradicting evidence
        self.query_template = self.options.pop('query_template', "{claim}")
        self.min_score = self.options.pop('min_score', 0.0)

    def build_query(self, claim: str) -> str:
        """
        Build a search query from a claim using the configured template.

        Args:
            claim (str): The claim to build a query for

        Returns:
            str: The formatted search query
        """
        return self.query_template.format(claim=claim)

    def gather_evidence(self, claim: str):
        """
        Gather and filter evidence relevant to a given claim.

        Args:
            claim (str): The claim to gather evidence for

        Returns:
            list: List of filtered evidence nodes that meet the similarity threshold
        """
        query = self.build_query(claim)
        evidence = self.retriever.retrieve(query)

        # Check if evidence is a list and if its elements are NodeWithScore objects
        if not isinstance(evidence, list):
            logging.error("Expected evidence to be a list, but got %s. Returning empty list.", type(evidence))
            return []
        if evidence and not isinstance(evidence[0], NodeWithScore):
            logging.error("Evidence items are not NodeWithScore objects (first item is %s). Returning empty list.", type(evidence[0]))
            return []
        logging.info(f"Retrieved {len(evidence)} evidence nodes for claim: {claim}")

        # Filter evidence based on similarity score
        filtered_evidence = self.classify_evidence(evidence)
        logging.info(f"Filtered to {len(filtered_evidence)} evidence nodes with similarity score > {self.min_score}")
        if not filtered_evidence:
            return []
        
        # Extract text from evidence nodes
        evidence_texts = self.extract_text_from_evidence(filtered_evidence)
        logging.info(f"Extracted text from {len(evidence_texts)} evidence nodes")

        return evidence_texts

    def gather_evidence_with_metadata(self, claim: str) -> list[dict]:
        """
        Gather evidence and return list of dicts with chunk_id, text, and score
        for tracking and for use with evidence classifier (relevant phrase, etc.).
        """
        query = self.build_query(claim)
        evidence = self.retriever.retrieve(query)
        if not isinstance(evidence, list) or not evidence:
            return []
        if not isinstance(evidence[0], NodeWithScore):
            logging.error("Evidence items are not NodeWithScore; returning empty list.")
            return []
        filtered_evidence = self.classify_evidence(evidence)
        if not filtered_evidence:
            return []
        result = []
        for i, item in enumerate(filtered_evidence):
            node = item.node
            chunk_id = getattr(node, "node_id", None) or getattr(node, "id", None)
            if chunk_id is None:
                chunk_id = f"chunk_{i}"
            if hasattr(chunk_id, "hex"):  # UUID
                chunk_id = str(chunk_id)
            result.append({
                "chunk_id": chunk_id,
                "text": node.text,
                "score": float(item.score) if item.score is not None else 0.0,
            })
        logging.info(f"Retrieved {len(result)} evidence nodes with metadata for claim: {claim}")
        return result

    def extract_text_from_evidence(self, evidence: list[NodeWithScore]) -> list[str]:
        """
        Extract text from evidence nodes.

        Args:
            evidence (list): List of evidence nodes to extract text from

        Returns:
            list: List of text strings extracted from evidence nodes

        """
        # check if evidence is a list
        if evidence is not None and isinstance(evidence, list):
            if evidence and isinstance(evidence[0], NodeWithScore):
                # For each NodeWithScore, extract the text from its node attribute.
                evidence_texts = [item.node.text for item in evidence]
                return evidence_texts
            else: 
                raise ValueError("Evidence must be a list of NodeWithScore objects")
        else: 
            logging.WARNING(f"Trying to extract text from non-list object: {evidence}")
            return []
        
            

    def classify_evidence(self, evidence: list[NodeWithScore]) -> list[NodeWithScore]:
        """
        Filter and classify evidence based on similarity scores.

        Args:
            evidence (list): List of evidence nodes to classify

        Returns:
            list: Filtered list of evidence nodes that meet the similarity threshold

        """
        processor = SimilarityPostprocessor(similarity_cutoff=self.min_score, **self.options)
        filtered_evidence = processor.postprocess_nodes(evidence)
        return filtered_evidence
