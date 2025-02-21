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
        self.min_score = self.options.pop('min_score', 0.75)

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
        filtered_evidence = self.classify_evidence(evidence)
        evidence_texts = self.extract_text_from_evidence(filtered_evidence)
        return evidence_texts
    
    def extract_text_from_evidence(self, evidence) -> list[str]:
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
            raise ValueError("Evidence must be a list")
        
            

    def classify_evidence(self, evidence):
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
