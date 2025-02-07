from llama_index.core.llms import ChatMessage
import json
import os
from factchecker.core.llm import load_llm

class EvaluateStep:
    """
    A step in the fact-checking process that evaluates claims based on pro and con evidence.
    
    This class uses a language model to analyze evidence both supporting and contradicting a claim,
    producing a final verdict on the claim's veracity.
    """

    def __init__(self, llm=None, options=None):
        """
        Initialize an EvaluateStep instance.

        Args:
            llm: Language model instance to use for evaluation. If None, loads default model.
            options (dict, optional): Configuration options including:
                - pro_prompt_template: Template for formatting supporting evidence
                - con_prompt_template: Template for formatting contradicting evidence
                - system_prompt_template: Template for system information
                - format_prompt: Instructions for response format
        """
        self.llm = llm if llm is not None else load_llm()

        self.options = options if options is not None else {}
        # Extract prompt templates from options
        self.pro_prompt_template = self.options.pop('pro_prompt_template', "Pro evidence: {evidence}")
        self.con_prompt_template = self.options.pop('con_prompt_template', "Con evidence: {evidence}")
        self.system_prompt_template = self.options.pop('system_prompt_template', "System information:")
        self.format_prompt = self.options.pop("format_prompt", "Answer with TRUE or FALSE")
        # Pop additional options if necessary
        self.additional_options = {key: self.options.pop(key) for key in list(self.options.keys())}
        if "response_format" not in self.additional_options:
            self.additional_options["response_format"] = {"type": "json_object"}

    def evaluate_claim(self, claim, pro_evidence, con_evidence):
        """
        Evaluate a claim by analyzing both supporting and contradicting evidence.

        Args:
            claim (str): The claim to evaluate
            pro_evidence (str): Evidence supporting the claim
            con_evidence (str): Evidence contradicting the claim

        Returns:
            str: The final verdict (TRUE, FALSE, or ERROR_PARSING_RESPONSE)
        """
        # Format the system prompt with the claim
        system_prompt_with_claim = self.system_prompt_template.format(claim=claim)
        
        # Format the pro and con prompts with their respective evidence
        pro_prompt = self.pro_prompt_template.format(evidence=pro_evidence)
        con_prompt = self.con_prompt_template.format(evidence=con_evidence)
        format_prompt = self.format_prompt
        # Combine the system prompt with the claim, pro, and con prompts
        combined_prompt = f"{system_prompt_with_claim}\n{pro_prompt}\n{con_prompt}"
        

        # TODOs
        # * pass the messages as messages with role (system, user) instead of just joining them
        # * 


        # Call the completion endpoint with the system prompt as a system message and the claim, pro, and con prompts as user messages
        messages = [
            ChatMessage(role="system", content=system_prompt_with_claim),
            ChatMessage(role="user", content=pro_prompt),
            ChatMessage(role="user", content=con_prompt),
            ChatMessage(
                role="user", content=format_prompt)
        ]


        response = self.llm.chat(messages, **self.additional_options)
        # Parse the JSON response from the LLM to extract the label
        try:
            # Assuming the response is a ChatResponse object as shown in the message
            response_content = response.message.content
            # Load the content as a JSON object
            response_data = json.loads(response_content)
            # Extract the label and convert it to uppercase with underscores
            label = response_data['label'].upper().replace(" ", "_")
        except (json.JSONDecodeError, KeyError) as e:
            # Handle potential errors in JSON parsing or missing keys
            label = "ERROR_PARSING_RESPONSE"


        return label