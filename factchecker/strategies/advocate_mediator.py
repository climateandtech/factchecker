from factchecker.steps.advocate import AdvocateStep
from factchecker.steps.mediator import MediatorStep
from factchecker.indexing.base import BaseIndexer
from factchecker.retrieval.base import BaseRetriever

class AdvocateMediatorStrategy:
    def __init__(self, indexer_options, retriever_options, advocate_options, mediator_options, advocate_prompt, mediator_prompt):
        self.indexer = BaseIndexer(indexer_options)
        self.retriever = BaseRetriever(self.indexer, retriever_options)
        self.advocate_steps = [AdvocateStep(options={**advocate_options, 'system_prompt_template': advocate_prompt}) for _ in range(1)]  # Example with 3 advocates
        self.mediator_step = MediatorStep(options={**mediator_options, 'arbitrator_primer': mediator_prompt})

    def evaluate_claim(self, claim):
        # Retrieve evidence for the claim
        evidence = self.retriever.retrieve(claim)

        # Each advocate evaluates the claim based on the evidence
        verdicts_and_reasonings = [advocate.evaluate_evidence(claim, evidence) for advocate in self.advocate_steps]
        import pdb; pdb.set_trace()
        # Separate verdicts and reasonings
        verdicts = [verdict for verdict, reasoning in verdicts_and_reasonings]
        reasonings = [reasoning for verdict, reasoning in verdicts_and_reasonings]

        # The mediator synthesizes the verdicts
        final_verdict = self.mediator_step.synthesize_verdicts(verdicts, claim)

        return final_verdict, verdicts, reasonings