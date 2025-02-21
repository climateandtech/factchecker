from llama_index.core.llms import ChatMessage
import logging
import os
from factchecker.core.llm import load_llm

logger = logging.getLogger('factchecker.steps')

class MediatorStep:
    def __init__(self, llm=None, options=None):
        self.llm = llm if llm is not None else load_llm()

        self.options = options if options is not None else {}
        self.prompt = self.options.get('arbitrator_primer', '')
        self.additional_options = {key: self.options.pop(key) for key in list(self.options.keys())}
        self.max_retries = 3

    def synthesize_verdicts(self, verdicts_and_reasonings, claim):
        logger.info(f"Mediator received claim: {claim}")
        system_prompt_with_claim = self.prompt.format(claim=claim)
        logger.info(f"System prompt after formatting: {system_prompt_with_claim[:200]}...")  # First 200 chars
        
        # Format the verdicts and reasonings with <> tags
        formatted_verdicts_and_reasonings = "\n".join(
            [f"<verdict>{verdict}</verdict><reasoning>{reasoning}</reasoning>" for verdict, reasoning in verdicts_and_reasonings]
        )
        
        messages = [
            ChatMessage(role="system", content=system_prompt_with_claim),
            ChatMessage(role="user", content=f"Claim to evaluate: {claim}\n\nHere are the verdicts and reasonings of the different advocates:\n{formatted_verdicts_and_reasonings}\nPlease provide the final verdict as ((correct)), ((incorrect)), or ((not_enough_information))")
        ]
        logger.info(f"User message content: {messages[1].content[:200]}...")  # First 200 chars
        
        valid_options = {key: value for key, value in self.additional_options.items() if key in ["response_format", "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]}

        for attempt in range(self.max_retries):
            response = self.llm.chat(messages, **valid_options)
            response_content = response.message.content.strip()
            # Extract the final verdict from the response
            start = response_content.find("((")
            end = response_content.find("))")
            if start != -1 and end != -1:
                final_verdict = response_content[start+2:end].strip().upper().replace(" ", "_")
                return final_verdict, response_content
            else:
                logging.warning(f"Unexpected response content on attempt {attempt + 1}: {response_content}")
        
        return "ERROR_PARSING_RESPONSE", "No reasoning available"