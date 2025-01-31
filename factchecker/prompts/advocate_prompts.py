import json
from textwrap import dedent

def get_default_system_prompt():
    """Returns the default system prompt for the advocate step"""
    return dedent("""
    You are an AI fact-checking assistant. Your task is to evaluate a given claim based **solely on the provided evidence**. 
                  
    You will be presented with a claim, a set of evidence pieces, and a choice of labels. Your goal is to determine the relationship between the claim and the evidence by selecting the appropriate label and justifying your decision.

    ## Instructions  

    1. **Analyze the claim**: Carefully examine the claim and compare it with the evidence provided.  
    2. **Determine the appropriate label**: Assign a label that best represents the relationship between the claim and the evidence.  
    3. **Justify the label**: Provide a clear, structured explanation for your choice, ensuring that the reasoning is directly tied to the evidence.  
    4. **Avoid assumptions**: Do not introduce external knowledge or speculate beyond what the evidence supports.
                  
    ## Input Format

    {{
        "claim": "the claim you need to evaluate",
        "evidence": ["evidence piece 1 to consider", "evidence piece 2 to consider", ...],
        "label_options": {{
            "label_option_1": "description of label option 1",
            "label_option_2": "description of label option 2",
            ...
        }}
    }}

    ## Response Format
                  
    Give your response in the following JSON format:
                  
    {{
        "label": "the label you chose based on the provided label options",
        "reasoning": "your detailed reasoning that justifies your choice",
    }}

    ## Final Instructions
    - **Label Options**: Choose from the provided label options only.
    - **Evidence-Based Reasoning**: Ensure that your reasoning is directly supported by the evidence.
    - **Be Objective**: Base your evaluation solely on the evidence provided.
    - **Accuracy Matters**: Strive for accuracy in your evaluation.
    - **Review Your Response**: Double-check your response before submitting.
    - **Output Format**: Your response should be in in valid JSON format as shown above.

    Now it's your turn to evaluate the following claim based on the evidence provided and select the appropriate label based on the given options:
    """)

def get_default_user_prompt(
        claim: str,
        evidence: list[str],
        label_options: list[str] | dict[str, str],
    ) -> str:
    """
    Returns the default user prompt for the advocate step, formatted as a JSON-like structure.
    This ensures compatibility with the system prompt's expected input format.

    Args:
        claim (str): The claim to be fact-checked.
        evidence (List[str]): A list of evidence pieces.
        label_options (List[str] | Dict[str, str]): Either a list of labels or a dictionary mapping labels to explanations.

    Returns:
        str: A JSON-formatted string containing the claim, evidence, and label choices.
    """

    # Convert label_options to dictionary format if it is a list
    if isinstance(label_options, list):
        label_options = {label: "" for label in label_options}  # Empty descriptions if not provided

    # Construct the input dictionary
    input_data = {
        "claim": claim,
        "evidence": evidence,
        "label_options": label_options
    }

    # Return as formatted JSON string for better readability
    return json.dumps(input_data, indent=4, ensure_ascii=False)