from llama_index.core.llms import ChatMessage
import logging
import os
from factchecker.core.llm import load_llm

class MediatorStep:
    """
    A step in the fact-checking process that mediates between multiple advocate verdicts.
    
    This class synthesizes multiple verdicts and their associated reasoning to produce
    a final consensus verdict on a claim's veracity.
    """

    def __init__(self, llm=None, options=None):
        """
        Initialize a MediatorStep instance.

        Args:
            llm: Language model instance to use for mediation. If None, loads default model.
            options (dict, optional): Configuration options including:
                - arbitrator_primer: Template for the system prompt
        """
        self.llm = llm if llm is not None else load_llm()
        self.options = options if options is not None else {}
        self.system_prompt = self.options.pop('system_prompt', '')
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
        # Format the verdicts and reasonings with <> tags
        formatted_verdicts_and_reasonings = "\n".join(
            [f"<verdict>{verdict}</verdict><reasoning>{reasoning}</reasoning>" for verdict, reasoning in verdicts_and_reasonings]
        )
        
        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=f"Here are the verdicts and reasonings of the different advocates:\n{formatted_verdicts_and_reasonings}\nPlease provide the final verdict as ((correct)), ((incorrect)), or ((not_enough_information)) for the claim: {claim}")
        ]
        
        valid_options = {key: value for key, value in self.additional_options.items() if key in ["response_format", "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]}

        for attempt in range(self.max_retries):
            response = self.llm.chat(messages, **valid_options)
            response_content = response.message.content.strip()
            # Extract the final verdict from the response
            start = response_content.find("((")
            end = response_content.find("))")
            if start != -1 and end != -1:
                final_verdict = response_content[start+2:end].strip().upper().replace(" ", "_")
                return final_verdict
            else:
                logging.warning(f"Unexpected response content on attempt {attempt + 1}: {response_content}")
        
        return "ERROR_PARSING_RESPONSE"