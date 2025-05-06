from factchecker.steps.advocate import AdvocateStep
from factchecker.steps.mediator import MediatorStep
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever
from typing import List, Dict, Union

class AdvocateMediatorStrategy:
    """
    A strategy that combines multiple advocates and a mediator for fact-checking claims.
    
    This strategy uses multiple advocates to evaluate a claim independently. 
    Advocates can share the same indexer or use different ones.
    A mediator then synthesizes their verdicts into a final consensus.
    """

    def __init__(
            self, 
            indexer_options_list: Union[list[dict], LlamaVectorStoreIndexer], 
            retriever_options_list: list[dict], 
            advocate_options_list: list[dict], 
            evidence_options: dict, 
            mediator_options: dict,
        ) -> None:
        """
        Initialize an AdvocateMediatorStrategy instance.

        Args:
            indexer_options_list: Either:
                - List of options for initializing document indexers (one per advocate)
                - Single LlamaVectorStoreIndexer instance to be shared by all advocates
            retriever_options_list (list): List of options for configuring retrievers
            advocate_options_list (list): List of configuration options for each advocate
            evidence_options (dict): Configuration options for evidence step
            mediator_options (dict): Configuration options for the mediator
        """
        # Handle single shared indexer case
        if isinstance(indexer_options_list, LlamaVectorStoreIndexer):
            self.indexers = [indexer_options_list] * len(retriever_options_list)
        else:
            # Initialize indexers with their options
            self.indexers = [LlamaVectorStoreIndexer(options) for options in indexer_options_list]

        # Initialize retrievers with their options
        self.retrievers = [
            LlamaBaseRetriever(
                indexer=indexer,
                options=retriever_options
            ) 
            for retriever_options, indexer in zip(retriever_options_list, self.indexers)
        ]
        
        # Create advocate steps with proper options
        self.advocate_steps = []

        # Create advocate step for each retriever with its specific options
        for retriever, advocate_options in zip(self.retrievers, advocate_options_list):
            # Extract HyDE config if present
            hyde_config = advocate_options.pop('hyde_config', None)
            
            # Create advocate step
            advocate_step = AdvocateStep(
                retriever=retriever,
                options=advocate_options,
                evidence_options=evidence_options,
                hyde_config=hyde_config
            )
            self.advocate_steps.append(advocate_step)
            
        self.mediator_step = MediatorStep(options=mediator_options)

    def evaluate_claim(self, claim: str) -> tuple[str, List[str], List[str]]:
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
        # Each advocate evaluates the claim
        verdicts_and_reasonings = [advocate.evaluate_claim(claim) for advocate in self.advocate_steps]

        # Separate verdicts and reasonings
        verdicts = [verdict for verdict, reasoning in verdicts_and_reasonings]
        reasonings = [reasoning for verdict, reasoning in verdicts_and_reasonings]

        # The mediator synthesizes the verdicts
        final_verdict = self.mediator_step.synthesize_verdicts(verdicts_and_reasonings, claim)

        return final_verdict, verdicts, reasonings