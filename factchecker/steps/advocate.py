import logging

from llama_index.core.llms import ChatMessage

from factchecker.core.llm import load_llm
from factchecker.config.config import DEFAULT_LABEL_OPTIONS
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.prompts.advocate_prompts import get_default_system_prompt, get_default_user_prompt
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever
from factchecker.steps.evidence import EvidenceStep


class AdvocateStep:
    """
    A step in the fact-checking process that acts as an advocate by evaluating claims based on evidence.
    
    This class retrieves relevant evidence for a claim and uses an LLM to evaluate whether the claim
    is supported by the evidence, producing a verdict and reasoning.
    """

    def __init__(self, llm=None, options=None, evidence_options=None):
        """
        Initialize an AdvocateStep instance.

        Args:
            llm: Language model instance to use for evaluation. If None, loads default model.
            options (dict, optional): Configuration options for the advocate step including:
                - evidence_prompt_template: Template for formatting evidence
                - system_prompt_template: Template for system information
                - format_prompt: Instructions for response format
                - max_evidences: Maximum number of evidence pieces to consider
            evidence_options (dict, optional): Configuration for evidence gathering including:
                - top_k: Number of top results to retrieve
                - min_score: Minimum similarity score for evidence
                - indexer: Instance of document indexer
        """
        self.llm = llm if llm is not None else load_llm()
        self.options = options if options is not None else {}
        self.system_prompt = self.options.pop('system_prompt', get_default_system_prompt())
        self.label_options = self.options.pop('label_options', DEFAULT_LABEL_OPTIONS)
        self.thinking_llm = self.options.pop("thinking_llm", False)
        self.thinking_token = self.options.pop("thinking_token", "think")
        
        # Filter out retriever-specific options
        self.additional_options = {
            key: self.options.pop(key) 
            for key in list(self.options.keys()) 
            if key not in ['top_k', 'similarity_top_k', 'min_score']
        }
        self.max_retries = 3

        # Extract top_k and min_score from evidence_options
        evidence_options = evidence_options if evidence_options is not None else {}
        top_k = evidence_options.pop('top_k', 5)
        min_score = evidence_options.pop('min_score', 0.75)

        # Get the indexer instance from evidence_options
        indexer = evidence_options.pop('indexer', None)
        if indexer is None:
            raise ValueError("No indexer provided in evidence_options")

        # Create retriever with the provided indexer
        retriever = LlamaBaseRetriever(indexer, {'similarity_top_k': top_k})
        
        # Initialize EvidenceStep
        self.evidence_step = EvidenceStep(
            retriever=retriever,
            options={
                **evidence_options,
                'top_k': top_k,
                'min_score': min_score
            }
        )

    def evaluate_evidence(self, claim):
        """
        Evaluate a claim based on gathered evidence using the language model.

        Args:
            claim (str): The claim to evaluate.

        Returns:
            tuple: A tuple containing:
                - label (str): The verdict (TRUE, FALSE, or NOT_ENOUGH_INFORMATION)
                - reasoning (str): The detailed reasoning behind the verdict
        """
        # Retrieve evidence for the claim
        evidences = self.evidence_step.gather_evidence(claim)
        system_prompt_with_claim = self.system_prompt_template.format(claim=claim)
        evidence_text = "\n".join([self.evidence_prompt_template.format(evidence=evidence.text) for evidence in evidences])
        format_prompt = self.format_prompt
        combined_prompt = f"Factcheck the following claim:\n\nClaim: {claim}\nGiven the following evidence:\n{evidence_text}\n{format_prompt}"
        messages = [
            ChatMessage(role="system", content=system_prompt_with_claim),
            ChatMessage(role="user", content=combined_prompt)
        ]

        for attempt in range(self.max_retries):
            response = self.llm.chat(messages, **self.additional_options)
            response_content = response.message.content.strip()
            if self.thinking_llm:
                start_thought = response_content.find(f"<{self.thinking_token}>")
                end_thought = response_content.find(f"</{self.thinking_token}>")
                if end_thought > start_thought:
                    response_content = response_content[end_thought + len(f"</{self.thinking_token}>"):]
                logging.warning(f"Thinking segment found with start at {start_thought} and end at {end_thought}")

            start = response_content.find("((")
            end = response_content.find("))")
            if start != -1 and end != -1:
                label = response_content[start+2:end].strip().upper().replace(" ", "_")
                reasoning = response_content.strip()  # Return the whole response_content as reasoning
                return label, reasoning
            else:
                logging.warning(f"Unexpected response content on attempt {attempt + 1}: {response_content}")
        
        return "ERROR_PARSING_RESPONSE", "No reasoning available"