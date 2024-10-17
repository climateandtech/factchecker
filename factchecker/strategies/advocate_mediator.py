from factchecker.steps.advocate import AdvocateStep
from factchecker.steps.mediator import MediatorStep
from factchecker.indexing.base import BaseIndexer
from factchecker.retrieval.base import BaseRetriever

class AdvocateMediatorStrategy:
    def __init__(self, indexer_options_list, retriever_options_list, advocate_options, mediator_options, advocate_prompt, mediator_prompt):
        self.indexers = [BaseIndexer(options) for options in indexer_options_list]
        self.advocate_steps = [
            AdvocateStep(
                options={**advocate_options, 'system_prompt_template': advocate_prompt},
                evidence_options={**retriever_options, 'indexer': indexer, 'top_k': advocate_options.get('top_k', 5), 'min_score': advocate_options.get('min_score', 0.75)}
            ) for retriever_options, indexer in zip(retriever_options_list, self.indexers)
        ]
        self.mediator_step = MediatorStep(options={**mediator_options, 'arbitrator_primer': mediator_prompt})

    def evaluate_claim(self, claim):
        # Each advocate evaluates the claim based on their own evidence
        verdicts_and_reasonings = [advocate.evaluate_evidence(claim) for advocate in self.advocate_steps]

        # Separate verdicts and reasonings
        verdicts = [verdict for verdict, reasoning in verdicts_and_reasonings]
        reasonings = [reasoning for verdict, reasoning in verdicts_and_reasonings]

        # The mediator synthesizes the verdicts
        final_verdict = self.mediator_step.synthesize_verdicts(verdicts_and_reasonings, claim)

        return final_verdict, verdicts, reasonings