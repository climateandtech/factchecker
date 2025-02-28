from factchecker.steps.advocate import AdvocateStep
from factchecker.steps.mediator import MediatorStep
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever
from factchecker.utils.logging_config import setup_logging
import logging
import debugpy

logger = logging.getLogger('factchecker.strategies')

class AdvocateMediatorStrategy:
    def __init__(self, indexer_options_list, retriever_options_list, advocate_options, mediator_options, advocate_prompt, mediator_prompt):
        logger.info("Initializing AdvocateMediatorStrategy")
        
        if not indexer_options_list or not retriever_options_list:  # Add this check
            raise ValueError("At least one source (advocate) must be provided")
            
        logger.debug("Creating indexers with options:")
        for i, options in enumerate(indexer_options_list):
            logger.debug(f"Indexer {i}: {options}")
        # Initialize indexers with their options
        self.indexers = [LlamaVectorStoreIndexer(options) for options in indexer_options_list]
        
        # Create advocate steps with proper options
        self.advocate_steps = []
        for retriever_options, indexer in zip(retriever_options_list, self.indexers):
            # Extract indexer options from retriever options
            indexer_opts = retriever_options.pop('indexer_options', {})
            
            # Create evidence options with proper structure
            evidence_options = {
                'indexer': indexer,
                'top_k': advocate_options.get('top_k', 5),
                'min_score': advocate_options.get('min_score', 0.75),
                'query_template': "evidence for: {claim}"
            }
            
            # Create advocate step
            advocate_step = AdvocateStep(
                options={**advocate_options, 'system_prompt_template': advocate_prompt},
                evidence_options=evidence_options
            )
            self.advocate_steps.append(advocate_step)
            
        self.mediator_step = MediatorStep(options={**mediator_options, 'arbitrator_primer': mediator_prompt})

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