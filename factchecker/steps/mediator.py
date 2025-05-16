from llama_index.core.llms import ChatMessage
import logging
import os
from factchecker.core.llm import load_llm
from typing import List, Dict, Any, Optional
import re
from factchecker.parsing.response_parsers import ResponseParser

logger = logging.getLogger(__name__)

class XMLParser(ResponseParser):
    """Parser that looks for verdict in XML format."""
    
    def parse_verdict(self, response_content: str) -> Optional[str]:
        verdict_match = re.search(r'<verdict>(.*?)</verdict>', response_content)
        if verdict_match:
            return verdict_match.group(1).strip().upper().replace(" ", "_")
        return None

class MediatorStep:
    """
    A step in the fact-checking process that mediates between multiple advocate verdicts.
    
    This class synthesizes multiple verdicts and their associated reasoning to produce
    a final consensus verdict on a claim's veracity.
    """

    def __init__(self, llm=None, options=None, parser: Optional[ResponseParser] = None):
        """
        Initialize a MediatorStep instance.

        Args:
            llm: Language model instance to use for mediation. If None, loads default model.
            options (dict, optional): Configuration options including:
                - system_prompt: Template for the system prompt
            parser (ResponseParser, optional): Parser instance to use for parsing LLM responses.
                If None, uses the XML format parser.
        """
        self.llm = llm if llm is not None else load_llm()
        self.options = options if options is not None else {}
        self.system_prompt = self.options.pop('system_prompt', '')
        self.parser = parser if parser is not None else XMLParser()
        self.additional_options = {key: self.options.pop(key) for key in list(self.options.keys())}
        self.max_retries = 3

    def synthesize_verdicts(self, verdicts_and_reasonings, claim):
        """
        Synthesize multiple verdicts and their reasoning into a final consensus verdict.

        Args:
            verdicts_and_reasonings (list): List of (verdict, reasoning) tuples from advocates
            claim (str): The claim being evaluated

        Returns:
            str: The final consensus verdict (CORRECT, INCORRECT, NOT_ENOUGH_INFORMATION, or ERROR_PARSING_RESPONSE)
        """
        logger.info(f"Synthesizing verdicts for claim: {claim}")
        logger.info(f"Number of advocate verdicts: {len(verdicts_and_reasonings)}")
        
        # Format the verdicts and reasonings with XML tags
        formatted_verdicts = []
        for i, (verdict, reasoning) in enumerate(verdicts_and_reasonings):
            logger.info(f"Advocate {i+1} verdict: {verdict}")
            logger.debug(f"Advocate {i+1} reasoning preview: {reasoning[:200]}...")
            formatted_verdicts.append(f"<verdict>{verdict}</verdict><reasoning>{reasoning}</reasoning>")
        
        formatted_input = "\n".join(formatted_verdicts)
        
        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=f"Here are the verdicts and reasonings of the different advocates:\n{formatted_input}\nPlease provide the final verdict for the claim: {claim}")
        ]
        
        valid_options = {key: value for key, value in self.additional_options.items() if key in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]}

        for attempt in range(self.max_retries):
            response = self.llm.chat(messages, **valid_options)
            
            # Add debugger for response inspection
            import pdb; pdb.set_trace()  # Debugger for response inspection
            
            # Handle different response formats (Ollama vs OpenAI)
            if isinstance(response, dict):
                # Ollama format
                response_content = response['message']['content'].strip()
            else:
                # OpenAI format
                response_content = response.message.content.strip() if hasattr(response.message, 'content') else response.content.strip()
            
            logger.debug(f"LLM response content: {response_content}")
            
            final_verdict = self.parser.parse_verdict(response_content)
            
            if final_verdict:
                logger.info(f"Successfully parsed final verdict: {final_verdict}")
                return final_verdict
            else:
                logger.warning(f"Unexpected response format on attempt {attempt + 1}. Expected XML verdict tags. Got: {response_content}")
        
        logger.error("Failed to parse response after all retries")
        return "ERROR_PARSING_RESPONSE"