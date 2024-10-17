from factchecker.retrieval.base import BaseRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from factchecker.core.llm import load_llm

class EvidenceStep:
    def __init__(self, retriever: BaseRetriever, options=None):
        self.retriever = retriever
        self.options = options if options is not None else {}
        # Extract specific options and remove them from the options hash
        self.query_template = self.options.pop('query_template', "evidence for: {claim}")
        self.top_k = self.options.pop('top_k', 5)
        self.min_score = self.options.pop('min_score', 0.75)

    def build_query(self, claim):
        return self.query_template.format(claim=claim)

    def gather_evidence(self, claim):
        query = self.build_query(claim)
        evidence = self.retriever.retrieve(query, top_k=self.top_k, min_score=self.min_score, **self.options)
        return evidence

    def classify_evidence(self, evidence):
        processor = SimilarityPostprocessor(similarity_cutoff=self.min_score, **self.options)
        filtered_evidence = processor.postprocess_nodes(evidence)
        return filtered_evidence

# Example usage:
# evidence_strategy = EvidenceStrategy(indexer, retriever, top_k=5, min_score=0.75)
# evidence = evidence_strategy.gather_evidence("The Earth is round.")
# classified_evidence = evidence_strategy.classify_evidence(evidence)