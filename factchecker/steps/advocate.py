import logging

from llama_index.core.llms import ChatMessage

from factchecker.config.config import DEFAULT_LABEL_OPTIONS
from factchecker.core.llm import load_llm
from factchecker.datastructures import LabelOption
from factchecker.prompts.advocate_prompts import get_default_system_prompt, get_default_user_prompt
from factchecker.retrieval.abstract_retriever import AbstractRetriever
from factchecker.steps.evidence import EvidenceStep


class AdvocateStep:
    """
    A step in the fact-checking process that acts as an advocate by evaluating claims based on evidence.
    
    This class retrieves relevant evidence for a claim and uses an LLM to evaluate whether the claim
    is supported by the evidence, producing a verdict and reasoning.

    Args:
        retriever (AbstractRetriever): Retriever instance to use for evidence retrieval.
        llm (TODO): Language model instance to use for evaluation. If None, loads default model.
        options (dict, optional): Configuration options for the advocate step including
        evidence_options (dict, optional): Configuration for evidence gathering including
        
    Attributes:
        retriever (AbstractRetriever): Retriever instance to use for evidence retrieval.
        llm (TODO): Language model instance to use for evaluation.
        options (dict): Configuration options for the advocate step.
        system_prompt (str): The system prompt to display to the user.
        label_options (dict): The available label options for the verdict.
        max_retries (int): The maximum number of retries to attempt when parsing the LLM response.
        chat_completion_options (dict): Additional options to pass to the LLM chat method.
    """

    def __init__(
            self, 
            retriever: AbstractRetriever,
            llm = None, # TODO: Add type hint
            options: dict = None,
            evidence_options: dict = None
        ) -> None:
        """Initialize an AdvocateStep instance."""
        self.retriever = retriever
        self.llm = llm if llm is not None else load_llm()
        self.options = options if options is not None else {}
        self.evidence_options = evidence_options if evidence_options is not None else {}
        self.system_prompt = self.options.pop('system_prompt', get_default_system_prompt())
        self.label_options = self.options.pop('label_options', DEFAULT_LABEL_OPTIONS)
        self.max_retries = self.options.pop('max_retries', 3)
        self.chat_completion_options = self.options.pop('chat_completion_options', {})
        
        # Initialize EvidenceStep
        self.evidence_step = EvidenceStep(
            retriever=retriever,
            options={
                **self.evidence_options,
            }
        )

    def retrieve_evidence(self, claim: str) -> list[str]:
        """
        Retrieve relevant evidence for a given claim.

        Args:
            claim (str): The claim for which to retrieve evidence.

        Returns:
            list[str]: A list of evidence pieces relevant to the claim.

        """
        return self.evidence_step.gather_evidence(claim)

    def evaluate_claim(self, claim: str) -> tuple[str, str]:
        """
        Evaluate a claim based on gathered evidence using the language model.

        Args:
            claim (str): The claim to evaluate.

        Returns:
            A tuple including the label and reasoning.

        """
        # Retrieve evidence for the claim
        evidence_list = self.retrieve_evidence(claim)

        # Define the message containing the payload for the LLM
        user_prompt = get_default_user_prompt(claim=claim, evidence=evidence_list, label_options=self.label_options)

        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ]


        for attempt in range(self.max_retries):
            response = self.llm.chat(messages, **self.chat_completion_options)
            response_content = response.message.content.strip()
            # Extract the verdict from the response
            start = response_content.find("((")
            end = response_content.find("))")
            if start != -1 and end != -1:
                label = response_content[start+2:end].strip().upper().replace(" ", "_")
                # Remove the label inside (( )) from the response content
                reasoning = response_content[:start].strip() + response_content[end+2:].strip()
                return label, reasoning
            else:
                logging.warning(f"Unexpected response content on attempt {attempt + 1}: {response_content}")
        
        return "ERROR_PARSING_RESPONSE", "No reasoning available"