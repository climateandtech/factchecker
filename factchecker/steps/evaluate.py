from llama_index.llms.openai import OpenAI

class EvaluateStep:
    def __init__(self, llm=None, options=None):
        self.llm = llm if llm is not None else OpenAI()
        self.options = options if options is not None else {}
        
        # Extract prompt templates from options
        self.pro_prompt_template = self.options.pop('pro_prompt_template', "Pro evidence: {evidence}")
        self.con_prompt_template = self.options.pop('con_prompt_template', "Con evidence: {evidence}")
        self.system_prompt_template = self.options.pop('system_prompt_template', "System information:")
        
        # Pop additional options if necessary
        self.additional_options = {key: self.options.pop(key) for key in list(self.options.keys())}

    def evaluate_evidence(self, claim, pro_evidence, con_evidence):
        # Format the system prompt with the claim
        system_prompt_with_claim = self.system_prompt_template.format(claim=claim)
        
        # Format the pro and con prompts with their respective evidence
        pro_prompt = self.pro_prompt_template.format(evidence=pro_evidence)
        con_prompt = self.con_prompt_template.format(evidence=con_evidence)
        
        # Combine the system prompt with the claim, pro, and con prompts
        combined_prompt = f"{system_prompt_with_claim}\n{pro_prompt}\n{con_prompt}"
        
        # Call the completion endpoint with the combined prompt
        response = self.llm.complete(prompt=combined_prompt, **self.additional_options)
        return response