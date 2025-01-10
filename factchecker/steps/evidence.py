from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from factchecker.core.llm import load_llm

class EvidenceStep:
    """
    A step in the fact-checking process that gathers and classifies evidence for claims.
    
    This class handles the retrieval and filtering of relevant evidence from a document store,
    using similarity-based search and post-processing to ensure quality results.
    """

    def __init__(self, retriever: LlamaBaseRetriever, options=None):
        """
        Initialize an EvidenceStep instance.

        Args:
            retriever (LlamaBaseRetriever): The retriever instance for finding relevant documents
            options (dict, optional): Configuration options including:
                - query_template: Template for formatting the search query
                - top_k: Number of top results to retrieve
                - min_score: Minimum similarity score threshold
        """
        self.options = options if options is not None else {}
        # Extract specific options and remove them from the options hash
        # The query_template can be customized to target specific types of evidence,
        # e.g. "evidence for: {claim}" for supporting evidence or "evidence against: {claim}" 
        # for contradicting evidence
        self.query_template = self.options.pop('query_template', "{claim}")
        self.top_k = self.options.pop('top_k', 5)
        self.min_score = self.options.pop('min_score', 0.75)
        
        # Update retriever options with top_k
        retriever_options = getattr(retriever, 'options', {})
        retriever_options['similarity_top_k'] = self.top_k
        retriever.options = retriever_options
        self.retriever = retriever

    def build_query(self, claim):
        """
        Build a search query from a claim using the configured template.

        Args:
            claim (str): The claim to build a query for

        Returns:
            str: The formatted search query
        """
        return self.query_template.format(claim=claim)

    def gather_evidence(self, claim):
        """
        Gather and filter evidence relevant to a given claim.

        Args:
            claim (str): The claim to gather evidence for

        Returns:
            list: List of filtered evidence nodes that meet the similarity threshold
        """
        query = self.build_query(claim)
        evidence = self.retriever.retrieve(query)
        return self.classify_evidence(evidence)

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

# Example usage:
# evidence_strategy = EvidenceStrategy(indexer, retriever, top_k=5, min_score=0.75)
# evidence = evidence_strategy.gather_evidence("The Earth is round.")
# classified_evidence = evidence_strategy.classify_evidence(evidence)