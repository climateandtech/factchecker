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
        # Each advocate evaluates the claim based on their own evidence
        results = [advocate.evaluate_evidence(claim) for advocate in self.advocate_steps]
        
        # Separate verdicts, reasonings, and evidences
        verdicts = [verdict for verdict, reasoning, evidence in results]
        reasonings = [reasoning for verdict, reasoning, evidence in results]
        evidences = [evidence for verdict, reasoning, evidence in results]
        
        # The mediator synthesizes the verdicts
        final_verdict, mediator_reasoning = self.mediator_step.synthesize_verdicts(list(zip(verdicts, reasonings)), claim)
        
        # Add mediator's reasoning to the list
        reasonings.append(mediator_reasoning)
        
        return final_verdict, verdicts, reasonings, evidences