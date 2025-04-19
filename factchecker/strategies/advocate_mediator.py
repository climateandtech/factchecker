from factchecker.steps.advocate import AdvocateStep
from factchecker.steps.mediator import MediatorStep
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever
from factchecker.utils.logging_config import setup_logging
import logging
import debugpy

logger = logging.getLogger('factchecker.strategies')

class AdvocateMediatorStrategy:
    """
    A strategy that combines multiple advocates and a mediator for fact-checking claims.
    
    This strategy uses multiple advocates, each with their own evidence sources, to evaluate
    a claim independently. A mediator then synthesizes their verdicts into a final consensus.
    """

    def __init__(
            self, 
            indexer_options_list: list[dict], 
            retriever_options_list: list[dict], 
            advocate_options: dict, 
            evidence_options: dict, 
            mediator_options: dict,
        ) -> None:
        """
        Initialize an AdvocateMediatorStrategy instance.

        Args:
            indexer_options_list (list): List of options for initializing document indexers
            retriever_options_list (list): List of options for configuring retrievers
            advocate_options (dict): Configuration options for advocates
            evidence_options (dict): Configuration options for evidence step
            mediator_options (dict): Configuration options for the mediator
            
        """
        if not indexer_options_list or not retriever_options_list:
            raise ValueError("At least one source (advocate) must be provided")

        # Initialize indexers with their options
        for i, options in enumerate(indexer_options_list):
            logger.debug(f"Creating indexer {i} with options: {options}")
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
        Evaluates a claim using multiple advocates and a mediator.
        """
        # Get verdicts from each advocate
        verdicts_and_reasonings = []
        evidences_list = []
        
        logger.info(f"Starting evaluation with {len(self.advocate_steps)} advocates")
        
        for advocate in self.advocate_steps:
            logger.info(f"Getting verdict from advocate")
            verdict, reasoning, evidences = advocate.evaluate_evidence(claim)
            logger.info(f"Advocate returned verdict: {verdict}")
            logger.info(f"Advocate returned reasoning: {reasoning}")
            verdicts_and_reasonings.append((verdict, reasoning))
            evidences_list.append(evidences)
        
        logger.info(f"All advocate verdicts: {verdicts_and_reasonings}")
        
        # Get final verdict from mediator
        # TODO: consider an option to use the evidence list in the mediator
        final_verdict, mediator_reasoning = self.mediator_step.synthesize_verdicts(verdicts_and_reasonings, claim)
        logger.info(f"Mediator returned verdict: {final_verdict}")
        logger.info(f"Mediator reasoning: {mediator_reasoning}")
        
        # Return all results
        verdicts = [v for v, _ in verdicts_and_reasonings]
        reasonings = [r for _, r in verdicts_and_reasonings]
        
        return final_verdict, mediator_reasoning, verdicts, reasonings, evidences_list
