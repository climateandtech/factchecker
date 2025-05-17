import logging

from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever
from factchecker.steps.advocate import AdvocateStep
from factchecker.steps.extract import ExtractStep
from factchecker.steps.mediator import MediatorStep


class ExtractorAdvocateMediatorStrategy:
    """
    A strategy that combines multiple advocates and a mediator for fact-checking claims.
    An extractor is used to break the text into subclaims.

    This strategy uses multiple advocates, each with their own evidence sources, to evaluate
    a claim independently. A mediator then synthesizes their verdicts into a final consensus.
    """

    def __init__(
        self,
        indexer_options_list: list[dict],
        retriever_options_list: list[dict],
        extractor_options: dict,
        advocate_options: dict,
        evidence_options: dict,
        mediator_options: dict,
    ) -> None:
        """
        Initialize an ExtractorAdvocateMediatorStrategy instance.

        Args:
            indexer_options_list (list): List of options for initializing document indexers
            retriever_options_list (list): List of options for configuring retrievers
            extractor_options (dict): Configuration options for extractor
            advocate_options (dict): Configuration options for advocates
            evidence_options (dict): Configuration options for evidence step
            mediator_options (dict): Configuration options for the mediator

        """
        self.extractor_step = ExtractStep(options=extractor_options)

        # Initialize indexers with their options
        self.indexers = [
            LlamaVectorStoreIndexer(options) for options in indexer_options_list
        ]

        # Initialize retrievers with their options
        self.retrievers = [
            LlamaBaseRetriever(indexer=indexer, options=retriever_options)
            for retriever_options, indexer in zip(
                retriever_options_list, self.indexers, strict=True
            )
        ]

        # Create advocate steps with proper options
        self.advocate_steps = []

        # Create advocate step for each retriever
        for retriever in self.retrievers:
            # Create advocate step
            advocate_step = AdvocateStep(
                retriever=retriever,
                options=advocate_options,
                evidence_options=evidence_options,
            )

            self.advocate_steps.append(advocate_step)

        self.mediator_step = MediatorStep(options=mediator_options)

    def evaluate_claim(self, claim: str, context: str = ""):
        """
        Evaluate a claim using multiple advocates and a mediator.

        Args:
            claim (str): The claim to evaluate
            context (str): The context of the discourse from where the claim was extracted.

        Returns:
            tuple: A tuple containing:
                - final_verdict (str): The consensus verdict
                - verdicts (list): List of individual advocate verdicts
                - reasonings (list): List of advocate reasonings
        """

        # Each advocate evaluates the claim based on their own evidence
        logging.info(f"Analyzing the following claim: {claim}")
        logging.info(f"In the following context: {context}")
        verdicts_and_reasonings = [
            advocate.evaluate_claim(claim, context) for advocate in self.advocate_steps
        ]

        # Separate verdicts and reasonings
        verdicts = [verdict for verdict, reasoning in verdicts_and_reasonings]
        reasonings = [reasoning for verdict, reasoning in verdicts_and_reasonings]

        # The mediator synthesizes the verdicts
        final_verdict = self.mediator_step.synthesize_verdicts(
            verdicts_and_reasonings, claim
        )

        return final_verdict, verdicts, reasonings

    def evaluate_text(self, text: str):
        """
        Evaluate a text by splitting it into claims and using multiple advocates and a mediator.

        Args:
            text (str): The text to evaluate

        Returns:
            List[Tuple[Union[List, str]]]: A list of verdicts for all the claims extracted
                Each verdict is represented by a tuple containing:
                    - final_verdict (str): The consensus verdict
                    - verdicts (list): List of individual advocate verdicts
                    - reasonings (list): List of advocate reasonings

            str: The final verdict of the text. final_verdict = any(verdicts)
        """
        claims_list = self.extractor_step.extract_claims(text=text)
        claims = claims_list.claims
        context = claims_list.context
        logging.info(f"Found the following claims: {claims}")
        classifications = []
        # Need enough lists to store outputs from all advocates.
        advocate_verdicts_list = [[] * len(self.advocate_steps)]
        advocate_reasonings_list = [[] * len(self.advocate_steps)]
        if claims == []:
            advocate_verdicts = ""
            advocate_reasonings = ""
            final_verdict = "correct"
        else:
            for claim in claims:
                verdict, verdicts, reasonings = self.evaluate_claim(
                    claim=claim, context=context
                )
                classifications.append(verdict)
                verdicts.append((verdicts, reasonings))
                for idx, (advocate_verdict, advocate_reasoning) in enumerate(
                    zip(verdicts, reasonings)
                ):
                    # Store the verdict and reasoning in the list corresponding to the correct advocate
                    advocate_verdicts_list[idx].append(advocate_verdict)
                    advocate_reasonings_list[idx].append(advocate_reasoning)

            advocate_verdicts = [
                "|".join(_verdicts) for _verdicts in advocate_verdicts_list
            ]
            advocate_reasonings = [
                "|".join(_reasonings) for _reasonings in advocate_reasonings_list
            ]
            if any([c.lower() == "incorrect" for c in classifications]):
                final_verdict = "incorrect"
            else:
                final_verdict = "correct"
        return final_verdict, advocate_verdicts, advocate_reasonings
