from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
import logging

class AdvocateStep:
    def __init__(self, llm=None, options=None):
        self.llm = llm if llm is not None else OpenAI()
        self.options = options if options is not None else {}
        self.evidence_prompt_template = self.options.pop('evidence_prompt_template', "Evidence: {evidence}")
        self.system_prompt_template = self.options.pop('system_prompt_template', "System information:")
        self.format_prompt = self.options.pop("format_prompt", "Answer with TRUE or FALSE in the format ((correct)), ((incorrect)), or ((not_enough_information))")
        self.max_evidences = self.options.pop("max_evidences", 1)
        self.additional_options = {key: self.options.pop(key) for key in list(self.options.keys())}
        self.max_retries = 3

    def evaluate_evidence(self, claim, evidences):
        system_prompt_with_claim = self.system_prompt_template.format(claim=claim)
        evidence_text = "\n".join([self.evidence_prompt_template.format(evidence=evidence) for evidence in evidences])
        format_prompt = self.format_prompt
        combined_prompt = f"Factcheck the following claim:\n\nClaim: {claim}\nGiven the following evidence:\n{evidence_text}\n{format_prompt}"
        messages = [
            ChatMessage(role="system", content=system_prompt_with_claim),
            ChatMessage(role="user", content=combined_prompt)
        ]

        for attempt in range(self.max_retries):
            response = self.llm.chat(messages, **self.additional_options)
            response_content = response.message.content.strip()
            
            # Extract the verdict from the response
            start = response_content.find("((")
            end = response_content.find("))")
            if start != -1 and end != -1:
                label = response_content[start+2:end].strip().upper().replace(" ", "_")
                reasoning = response_content[end+2:].strip()  # Extract reasoning after the verdict
                return label, reasoning
            else:
                logging.warning(f"Unexpected response content on attempt {attempt + 1}: {response_content}")
        
        return "ERROR_PARSING_RESPONSE", "No reasoning available"