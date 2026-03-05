"""
Prompts for the EvidenceClassifierStep.

Classifies each retrieved evidence chunk as supports / refutes / not_enough_information
relative to a given claim. Single LLM call for all chunks.
"""
import json
from textwrap import dedent
from typing import Optional

EVIDENCE_STANCES = ("supports", "refutes", "not_enough_information")

EVIDENCE_CLASSIFIER_SYSTEM_PROMPT = dedent("""\
You are a scientific evidence classifier. Your sole task is to determine
the relationship between a claim and individual pieces of evidence.

For each evidence piece you must assign exactly one stance:
- "supports"                — the evidence directly supports or is consistent with the claim
- "refutes"                 — the evidence contradicts or is inconsistent with the claim
- "not_enough_information"  — the evidence is irrelevant or too vague to support or refute the claim

You may use not_enough_information whenever the evidence does not clearly support or refute the claim.

Rules:
1. Judge each evidence piece independently.
2. Do NOT evaluate whether the claim itself is true — only whether the evidence relates to it.
3. Base your classification solely on the text provided; do not add external knowledge.
4. Respond with ONLY a valid JSON array (no commentary before or after).
""")


def get_evidence_classifier_user_prompt(
    claim: str,
    evidence_texts: list[str],
    include_relevant_phrase: bool = False,
    chunk_ids: Optional[list[str]] = None,
) -> str:
    """Build the user prompt containing the claim and numbered evidence pieces."""
    if chunk_ids is not None and len(chunk_ids) != len(evidence_texts):
        chunk_ids = None
    numbered = []
    for i, t in enumerate(evidence_texts):
        piece = {"id": i, "text": t}
        if chunk_ids is not None:
            piece["chunk_id"] = chunk_ids[i]
        numbered.append(piece)
    instructions = (
        "For each evidence piece, respond with a JSON array of objects: "
        '[{"id": 0, "stance": "supports"}, {"id": 1, "stance": "refutes"}, ...]. '
        "Use exactly one of: supports, refutes, not_enough_information."
    )
    if include_relevant_phrase:
        instructions += (
            ' For each piece also include "relevant_phrase": a short quote or phrase from '
            "that evidence text that is most relevant to the claim. "
            "When stance is \"supports\" or \"refutes\", relevant_phrase is required (non-empty). "
            "When stance is \"not_enough_information\", use empty string for relevant_phrase. "
            "If the link to the claim is unclear, you may use not_enough_information (with empty relevant_phrase)."
        )
    payload = {
        "claim": claim,
        "evidence_pieces": numbered,
        "instructions": instructions,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
