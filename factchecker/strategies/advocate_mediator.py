import logging

from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.retrieval.abstract_retriever import AbstractRetriever
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever
from factchecker.steps.advocate import AdvocateStep
from factchecker.steps.mediator import MediatorStep
from factchecker.utils.logging_config import setup_logging

logger = logging.getLogger('factchecker.strategies')

class AdvocateMediatorStrategy:
    """
    A strategy that combines multiple advocates and a mediator for fact-checking claims.
    
    This strategy uses multiple advocates, each with their own evidence sources, to evaluate
    a claim independently. A mediator then synthesizes their verdicts into a final consensus.
    """

    def __init__(
        self,
        indexer_retriever_configs: list[dict],
        evidence_options: dict,
        advocate_options: dict,
        mediator_options: dict,
    ) -> None:
        """
        Initialize an AdvocateMediatorStrategy instance.

        Args:
            indexer_retriever_configs (list): List of configs. Each config must either contain:
                - 'indexer' (an AbstractIndexer) and 'retriever' (a retriever)
                - OR 'indexer_options' and 'retriever_options' to build them.
            evidence_options (dict): Options for evidence handling.
            advocate_options (dict): Options for advocate behavior.
            mediator_options (dict): Options for mediator behavior.
        """
        if not indexer_retriever_configs:
            raise ValueError("At least one indexer-retriever configuration must be provided")

        # Set up the retrievers to be used for the advocates
        self.retrievers = self._build_retrievers_from_configs(indexer_retriever_configs)
        
        # Build AdvocateSteps from retrievers
        self.advocate_steps = [
            AdvocateStep(
                retriever=retriever,
                options=advocate_options,
                evidence_options=evidence_options
            )
            for retriever in self.retrievers
        ]
            
        # Build the MediatorStep
        self.mediator_step = MediatorStep(options=mediator_options)


    def _build_retrievers_from_configs(self, configs: list[dict]) -> list[AbstractRetriever]:
        """
        Build retrievers (and their indexers if necessary) from the given configurations.

        Args:
            configs (list): List of configurations. Each config must contain either:
                - 'retriever' (an AbstractRetriever), optionally alone
                - OR 'indexer' and 'retriever_options'
                - OR 'indexer_options' and 'retriever_options'
        
        Returns:
            list: List of retrievers built from the configurations.
        """
        retrievers = []

        for i, config in enumerate(configs):
            if 'retriever' in config:
                retriever = config['retriever']
                if 'indexer' in config or 'indexer_options' in config:
                    logger.warning(
                        f"Ignoring provided indexer or indexer_options at position {i} because a retriever object "
                        f"of type {type(retriever).__name__} was provided (which already has its own indexer)."
                    )
                logger.info(f"Using provided retriever of type {type(retriever).__name__} at position {i}")

            elif 'retriever_options' in config:
                if 'indexer' in config:
                    indexer = config['indexer']
                    logger.info(f"Using provided indexer of type {type(indexer).__name__} at position {i}")
                elif 'indexer_options' in config:
                    # TODO: Make indexer type dynamic based on retriever or explicit config.
                    indexer = LlamaVectorStoreIndexer(config['indexer_options'])
                    logger.info(f"Built new indexer of type {type(indexer).__name__} at position {i} from options")
                else:
                    raise ValueError(
                        f"Missing indexer or indexer_options for retriever_options at position {i}"
                    )
                
                # TODO: Validate retriever/indexer compatibility at construction
                # TODO: Make retriever type dynamic based on indexer or explicit config.

                # Create the retriever using the indexer and retriever options
                # For now, we hard-code the retriever type to LlamaBaseRetriever
                retriever = LlamaBaseRetriever(
                    indexer=indexer,
                    options=config['retriever_options']
                )
                logger.info(f"Created new retriever of type {type(retriever).__name__} at position {i}")

            else:
                raise ValueError(
                    f"Invalid config at position {i}: "
                    "Must provide at least 'retriever' or ('retriever_options' and 'indexer'/'indexer_options')."
                )

            retrievers.append(retriever)

        return retrievers
    

    def evaluate_claim(self, claim: str) -> tuple[str, str, list[str], list[str], list[list]]:
        """
        Evaluates a claim using multiple advocates and a mediator.

        This method processes a claim through the following steps:
        1. Each advocate independently evaluates the claim using its own evidence sources
        2. A mediator synthesizes all advocate verdicts into a final consensus
        
        Args:
            claim (str): The claim text to be fact-checked
            
        Returns:
            Tuple containing:
                - final_verdict (str): The final verdict from the mediator
                - mediator_reasoning (str): The mediator's reasoning for the verdict
                - verdicts (List[str]): List of individual advocate verdicts
                - reasonings (List[str]): List of individual advocate reasonings
                - evidences_list (List[Any]): List of evidence sets used by each advocate
        """
        # Get verdicts from each advocate
        verdicts_and_reasonings = []
        evidences_list = []
        
        logger.info(f"Starting evaluation with {len(self.advocate_steps)} advocates")
        
        for i, advocate in enumerate(self.advocate_steps):
            logger.info(f"Getting verdict from advocate #{i+1}")

            # Get advocate's evaluation
            verdict, reasoning, evidences = advocate.evaluate_claim(claim=claim)

            logger.info(f"Advocate returned verdict: {verdict}")
            logger.info(f"Advocate returned reasoning: {reasoning}")

            # Store the results
            verdicts_and_reasonings.append((verdict, reasoning))
            evidences_list.append(evidences)
        
        logger.info(f"All advocate verdicts: {verdicts_and_reasonings}")

        # Extract separate lists of verdicts and reasonings
        verdicts = [v for v, _ in verdicts_and_reasonings]
        reasonings = [r for _, r in verdicts_and_reasonings]

        # Synthesize a final verdict through the mediator
        final_verdict, mediator_reasoning = self.mediator_step.synthesize_verdicts(verdicts_and_reasonings, claim)

        logger.info(f"Mediator returned verdict: {final_verdict}")
        logger.info(f"Mediator reasoning: {mediator_reasoning}")
        
        return final_verdict, mediator_reasoning, verdicts, reasonings, evidences_list
