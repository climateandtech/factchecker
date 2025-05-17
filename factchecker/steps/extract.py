from llama_index.core.llms import ChatMessage
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
import logging
from factchecker.core.llm import load_llm
from dataclasses import dataclass
from typing import List, Dict
import json
import os


@dataclass
class ClaimList:
    claims: List[str]
    context: str


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
        self.context_prompt_template = self.options.pop(
            "context_prompt_template",
            "Summarise the following text into a couple of sentences: {text}",
        )
        self.splitter = SentenceSplitter(
            chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap
        )
        self.additional_options = {
            key: self.options.pop(key) for key in list(self.options.keys())
        }

    def extract_claims(self, text: str):
        """
        Evaluate a claim based on gathered evidence using the language model.

        Args:
            text (str): The text being split into multiple claims.

        Returns:
            claims (ClaimList): Data structure containing the list of claims extracted.
        """
        logging.info("Extracting claims")
        claims = self.splitter.split_text(text)
        logging.info("Creating Context")
        context_prompt_with_text = self.context_prompt_template.format(text=text)

        messages = [ChatMessage(role="user", content=context_prompt_with_text)]
        response = self.llm.chat(messages, **self.additional_options)
        response_content = response.message.content.strip()

        try:
            claims = ClaimList(claims=claims, context=response_content)
            return claims
        except Exception as e:
            logging.warning(
                f"Extractor unexpected response content: {response_content}"
            )
            logging.warning(f"Caught exception {e}, will return full text as context.")
            return ClaimList(claims=claims, context=text)
