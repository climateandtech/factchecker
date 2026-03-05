import json
from textwrap import dedent

def get_default_system_prompt():
    """Returns the default system prompt for the advocate step."""
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
                  
    Give your response in the following format:
                  
    "A concise and clear reasoning for your choice. ((your chosen label))"

    ## Final Instructions
    - **Label Options**: Choose from the provided label options only.
    - **Evidence-Based Reasoning**: Ensure that your reasoning is directly supported by the evidence.
    - **Be Objective**: Base your evaluation solely on the evidence provided.
    - **Accuracy Matters**: Strive for accuracy in your evaluation.
    - **Review Your Response**: Double-check your response before submitting.
    - **Output Format**: Your response should be in the specified format to be correctly evaluated.

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
        # join the list of label_options by ", " to create a string
        label_options = ','.join(label_options)

    # Construct the input dictionary
    input_data = {
        "claim": claim,
        "evidence": evidence,
        "label_options": label_options
    }

    # Return as formatted JSON string for better readability
    return json.dumps(input_data, indent=4, ensure_ascii=False)


def chunk_stats_from_classified(classified_evidence: list[dict]) -> dict:
    """
    Compute counts from classified evidence so NIN and chunks without relevant_phrase are tracked.
    Returns dict with n_supports, n_refutes, n_nin, n_with_relevant_phrase, n_without_relevant_phrase, n_total.
    """
    n_supports = n_refutes = n_nin = n_with_phrase = 0
    for c in classified_evidence or []:
        if not isinstance(c, dict):
            continue
        stance = (c.get("stance") or "").strip().lower()
        if stance == "supports":
            n_supports += 1
        elif stance == "refutes":
            n_refutes += 1
        else:
            n_nin += 1
        if (c.get("relevant_phrase") or "").strip():
            n_with_phrase += 1
    n_total = n_supports + n_refutes + n_nin
    n_without_phrase = n_total - n_with_phrase
    return {
        "n_supports": n_supports,
        "n_refutes": n_refutes,
        "n_nin": n_nin,
        "n_with_relevant_phrase": n_with_phrase,
        "n_without_relevant_phrase": n_without_phrase,
        "n_total": n_total,
    }


def get_classified_evidence_user_prompt(
        claim: str,
        classified_evidence: list[dict],
        label_options: dict[str, str],
        evidence_display_mode: str = "full",
        chunk_stats: dict | None = None,
    ) -> str:
    """
    User prompt variant that includes pre-classified evidence (with stance labels).

    Args:
        claim: The claim to be fact-checked.
        classified_evidence: List of dicts with text, stance, optional relevant_phrase, chunk_id, reason.
        label_options: Dict mapping verdict labels to their Science Feedback descriptions.
        evidence_display_mode: "full" = pass full chunk text; "relevant_only" = pass only relevant_phrase
            or "[no specific relevant part]" so chunks without a phrase and NIN chunks still appear with a placeholder.
        chunk_stats: Optional dict from chunk_stats_from_classified(); added as chunk_summary so model sees counts.

    Returns:
        JSON-formatted string for the advocate LLM.
    """
    display = classified_evidence
    if evidence_display_mode == "relevant_only" and classified_evidence:
        display = []
        for c in classified_evidence:
            copy = dict(c)
            phrase = (c.get("relevant_phrase") or "").strip()
            copy["text"] = phrase if phrase else "[no specific relevant part]"
            display.append(copy)
    input_data = {
        "claim": claim,
        "classified_evidence": display,
        "label_options": label_options,
    }
    if chunk_stats:
        input_data["chunk_summary"] = (
            f"Chunks: {chunk_stats.get('n_supports', 0)} supporting, {chunk_stats.get('n_refutes', 0)} refuting, "
            f"{chunk_stats.get('n_nin', 0)} NIN; {chunk_stats.get('n_with_relevant_phrase', 0)} with highlighted relevant phrase, "
            f"{chunk_stats.get('n_without_relevant_phrase', 0)} without."
        )
    return json.dumps(input_data, indent=4, ensure_ascii=False)