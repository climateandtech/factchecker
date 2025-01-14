from factchecker.steps.evidence import EvidenceStep
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.steps.evaluate import EvaluateStep

class EvidenceEvaluationStrategy:
    """
    A strategy that evaluates claims by gathering and analyzing both supporting and contradicting evidence.
    
    This strategy uses a single evidence gatherer to find both supporting and contradicting evidence,
    then evaluates the claim based on the balance of evidence found.
    """

    def __init__(self, indexer_options, retriever_options, evidence_options, evaluate_options):
        """
        Initialize an EvidenceEvaluationStrategy instance.

        Args:
            indexer_options (dict): Configuration for the document indexer
            retriever_options (dict): Configuration for the document retriever
            evidence_options (dict): Configuration for evidence gathering including:
                - pro_query_template: Template for queries seeking supporting evidence
                - contra_query_template: Template for queries seeking contradicting evidence
            evaluate_options (dict): Configuration for the evaluation step
        """
        self.indexer = LlamaVectorStoreIndexer(indexer_options)
        self.retriever = LlamaBaseRetriever(self.indexer, retriever_options)
        self.evidence_step = EvidenceStep(self.retriever, evidence_options)
        self.evaluate_step = EvaluateStep(options=evaluate_options)
        # Store pro and contra query templates from evidence_options
        self.pro_query_template = evidence_options.get('pro_query_template', "evidence for: {claim}")
        self.contra_query_template = evidence_options.get('contra_query_template', "evidence against: {claim}")

    def evaluate_claim(self, claim):
        """
        Evaluate a claim by gathering and analyzing supporting and contradicting evidence.

        Args:
            claim (str): The claim to evaluate

        Returns:
            tuple: A tuple containing:
                - evaluation_result (str): The final verdict
                - pro_count (int): Number of supporting evidence pieces
                - contra_count (int): Number of contradicting evidence pieces
                - pro_evidence (list): List of supporting evidence nodes
                - contra_evidence (list): List of contradicting evidence nodes
        """
        # Use the stored pro and contra query templates
        pro_evidence = self.evidence_step.gather_evidence(self.pro_query_template.format(claim=claim))
        contra_evidence = self.evidence_step.gather_evidence(self.contra_query_template.format(claim=claim))

        # Evaluate the evidence using the LLM
        evaluation_result = self.evaluate_step.evaluate_evidence(claim, pro_evidence, contra_evidence)

        # Count the pieces of evidence
        pro_count = len(pro_evidence)
        contra_count = len(contra_evidence)

        return evaluation_result, pro_count, contra_count, pro_evidence, contra_evidence
