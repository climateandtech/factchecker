from llama_index.core.llms import ChatMessage
import logging
from factchecker.core.llm import load_llm
from dataclasses import dataclass
from typing import List, Dict
import json
import os

@dataclass
class ClaimList:
    claims: Dict[str, List[str]]

class ExtractStep:
    """
    A preprocessing step in the fact-checking process that examines the text being classified and splits it into multiple claims if necessary. 
    
    This class then returns a list of claims to be analysed
    """

    def __init__(self, llm=None, options=None):
        """
        Initialize an ExtractStep instance.

        Args:
            llm: Language model instance to use for evaluation. If None, loads default model.
            options (dict, optional): Configuration options for the advocate step including:
                - system_prompt_template: Template for system information
                - format_prompt: Instructions for response format
        """
        self.llm = llm if llm is not None else load_llm()
        self.options = options if options is not None else {}
        self.claim_prompt_template = self.options.pop('claim_prompt_template', "Split the following text into claims: {text}")
        self.system_prompt_template = self.options.pop('system_prompt_template', "You are an assistant splitting text into indivudual claims.")
        self.format_prompt = self.options.pop("format_prompt", "Return as a list of claims [<claim_1>, <claim_2>, <claim_3>, ...]")
        self.additional_options = {key: self.options.pop(key) for key in list(self.options.keys())}

    def extract_claims(self, text):
        """
        Evaluate a claim based on gathered evidence using the language model.

        Args:
            text (str): The text being split into multiple claims.

        Returns:
            claims (ClaimList): Data structure containing the list of claims extracted.
        """
        claim_prompt_with_text = self.claim_prompt_template.format(text=text)
        format_prompt = self.format_prompt
        combined_prompt = f"{claim_prompt_with_text}\n{format_prompt}"
        messages = [
            ChatMessage(role="system", content=self.system_prompt_template),
            ChatMessage(role="user", content=combined_prompt)
        ]
        logging.info("Extracting claims")
        response = self.llm.chat(messages, **self.additional_options)
        response_content = response.message.content.strip()
        try:
            claims = ClaimList(json.loads(response_content))
            return claims.claims
        except Exception as e:
            logging.warning(f"Extractor unexpected response content: {response_content}")
            logging.warning(f"Caught exception {e}, will return full text.")
            return text