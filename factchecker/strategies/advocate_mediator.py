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

    def __init__(self, indexer_options_list, retriever_options_list, advocate_options, mediator_options, advocate_prompt, mediator_prompt):
        """
        Initialize an AdvocateMediatorStrategy instance.

        Args:
            indexer_options_list (list): List of options for initializing document indexers
            retriever_options_list (list): List of options for configuring retrievers
            advocate_options (dict): Configuration options for advocates
            mediator_options (dict): Configuration options for the mediator
            advocate_prompt (str): Template for advocate system prompts
            mediator_prompt (str): Template for mediator system prompt
        """
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