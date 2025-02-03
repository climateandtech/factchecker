from factchecker.steps.advocate import AdvocateStep
from factchecker.steps.mediator import MediatorStep
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever

class AdvocateMediatorStrategy:
    """
    A strategy that combines multiple advocates and a mediator for fact-checking claims.
    
    This strategy uses multiple advocates, each with their own evidence sources, to evaluate
    a claim independently. A mediator then synthesizes their verdicts into a final consensus.
    """

    def __init__(self, indexer_options_list, retriever_options_list, advocate_options, evidence_options, mediator_options):
        """
        Initialize an AdvocateMediatorStrategy instance.

        Args:
            indexer_options_list (list): List of options for initializing document indexers
            retriever_options_list (list): List of options for configuring retrievers
            advocate_options (dict): Configuration options for advocates
            evidence_options (dict): Configuration options for evidence step
            mediator_options (dict): Configuration options for the mediator
        """
        # Initialize indexers with their options
        self.indexers = [LlamaVectorStoreIndexer(options) for options in indexer_options_list]

        # Initialize retrievers with their options
        self.retrievers = [LlamaBaseRetriever(
            indexer=indexer,
            options=retriever_options
            ) 
        for retriever_options, indexer in zip(retriever_options_list, self.indexers, strict=True)]
        
        # Create advocate steps with proper options
        self.advocate_steps = []

        # Create advocate step for each retriever
        for retriever in self.retrievers:
            
            # Create advocate step
            advocate_step = AdvocateStep(
                retriever=retriever,
                options=advocate_options,
                evidence_options=evidence_options
            )

            self.advocate_steps.append(advocate_step)
            
        self.mediator_step = MediatorStep(options=mediator_options)

    def evaluate_claim(self, claim):
        """
        Evaluate a claim using multiple advocates and a mediator.

        Args:
            claim (str): The claim to evaluate

        Returns:
            tuple: A tuple containing:
                - final_verdict (str): The consensus verdict
                - verdicts (list): List of individual advocate verdicts
                - reasonings (list): List of advocate reasonings
        """
        # Each advocate evaluates the claim based on their own evidence
        verdicts_and_reasonings = [advocate.evaluate_evidence(claim) for advocate in self.advocate_steps]

        # Separate verdicts and reasonings
        verdicts = [verdict for verdict, reasoning in verdicts_and_reasonings]
        reasonings = [reasoning for verdict, reasoning in verdicts_and_reasonings]

        # The mediator synthesizes the verdicts
        final_verdict = self.mediator_step.synthesize_verdicts(verdicts_and_reasonings, claim)

        return final_verdict, verdicts, reasonings