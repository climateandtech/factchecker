"""
Prompts and category definitions for the Spectrum experiment, thoroughly aligned
with Science Feedback's claim review methodology.

Reference: https://science.feedback.org/process/#Claim-review

Key distinction Science Feedback makes:
- **Statements of fact** are rated on the accurate/inaccurate axis
  (does the statement match available data/observations?).
- **Explanations, hypotheses, and theories** are rated on the correct/incorrect axis
  (has the explanation been well-tested and confirmed by observations?).

Credibility tiers:
  Very High  — accurate, correct
  High       — mostly_accurate, mostly_correct
  Neutral    — correct_but, partially_correct, lacks_context, imprecise
  Low        — unsupported, misleading
  Very Low   — inaccurate, incorrect
  Issues     — flawed_reasoning
  Abstain    — not_enough_information
"""

import json
from textwrap import dedent
from typing import Optional

# ---------------------------------------------------------------------------
# Verdict categories (full granular scale, snake_case)
# ---------------------------------------------------------------------------
SPECTRUM_CATEGORIES = [
    "accurate",
    "correct",
    "mostly_accurate",
    "mostly_correct",
    "correct_but",
    "partially_correct",
    "lacks_context",
    "imprecise",
    "unsupported",
    "misleading",
    "inaccurate",
    "incorrect",
    "flawed_reasoning",
    "not_enough_information",
]

# ---------------------------------------------------------------------------
# Label options dict — maps each verdict to its Science Feedback description.
# Passed as label_options to the advocate so the LLM understands every choice.
# ---------------------------------------------------------------------------
SPECTRUM_LABEL_OPTIONS: dict[str, str] = {
    "accurate": (
        "Statement of fact that is consistent with available data/observations "
        "and does not omit relevant context."
    ),
    "correct": (
        "Explanation, hypothesis, or theory that has been well-tested in "
        "scientific studies and generates predictions confirmed by observations."
    ),
    "mostly_accurate": (
        "Statement of fact that is largely consistent with data but needs "
        "some clarification or additional information to be fully accurate."
    ),
    "mostly_correct": (
        "Explanation that is well-tested but whose formulation slightly "
        "overstates scientific confidence or distorts predictions."
    ),
    "correct_but": (
        "Explanation that is correct in substance but contains a caveat — "
        "e.g. it is true under specific conditions not mentioned in the claim."
    ),
    "partially_correct": (
        "Claim that significantly overstates scientific confidence in a theory, "
        "or is only correct in a narrow sense."
    ),
    "lacks_context": (
        "Claim that omits important observations or explanations that would "
        "change the reader's takeaway."
    ),
    "imprecise": (
        "Claim that uses ill-defined terms or lacks specifics so that one "
        "cannot unambiguously know what is meant."
    ),
    "unsupported": (
        "Claim made without adequate reference, or the available evidence "
        "does not support the statement."
    ),
    "misleading": (
        "Claim that contains an element of truth but leaves the reader "
        "with a false understanding of reality (e.g. by omitting context)."
    ),
    "inaccurate": (
        "Statement of fact in direct contradiction with available "
        "data/observations."
    ),
    "incorrect": (
        "Explanation or theory whose predictions have been invalidated "
        "by observations."
    ),
    "flawed_reasoning": (
        "The conclusion does not follow from the premises; logical error "
        "in the argument."
    ),
    "not_enough_information": (
        "The provided evidence is insufficient to assess the claim."
    ),
}

# ---------------------------------------------------------------------------
# Evidence classifier — optional experiment-level prompts for
# EvidenceClassifierStep (supports / refutes / not_enough_information per chunk).
# Winning strategy: persona + direct order in user message; optional reason sub-category.
# ---------------------------------------------------------------------------
EVIDENCE_CLASSIFIER_REASONS = (
    "same_number",
    "higher_number",
    "smaller_number",
    "contradiction",
    "partial_or_qualification",
    "different_scope_or_timeframe",
)

spectrum_evidence_classifier_system_prompt = dedent("""\
You are an expert evidence classifier for climate and environmental claims, with knowledge of
climate change, climate science, environmental science, physics, and energy science.

Your task is to classify each evidence piece as supports, refutes, or not_enough_information
with respect to the claim. Do NOT evaluate whether the claim is true — only whether the
evidence relates to it.

For each evidence piece assign exactly one stance:
- "supports"                — the evidence supports or is consistent with the claim
- "refutes"                 — the evidence contradicts or is inconsistent with the claim
- "not_enough_information"  — the evidence is irrelevant or too vague to support or refute the claim

You may use not_enough_information whenever the evidence does not clearly support or refute the claim.

Rules:
1. Judge each evidence piece independently. Base your classification only on the text provided.
2. For climate/science claims, consider whether the evidence addresses the same quantities,
   mechanisms, or time frames as the claim; if it does not, use not_enough_information.
3. When stance is supports or refutes, also provide a reason (see allowed values in the user message).
4. Respond with ONLY a valid JSON array (no commentary before or after).
""")


def spectrum_evidence_classifier_user_prompt(
    claim: str,
    evidence_texts: list[str],
    include_relevant_phrase: bool = False,
    chunk_ids: Optional[list[str]] = None,
    include_reason: bool = True,
) -> str:
    """Build the evidence-classifier user prompt for Spectrum: direct order + payload (claim + chunks)."""
    if chunk_ids is not None and len(chunk_ids) != len(evidence_texts):
        chunk_ids = None
    numbered = []
    for i, t in enumerate(evidence_texts):
        piece = {"id": i, "text": t}
        if chunk_ids is not None:
            piece["chunk_id"] = chunk_ids[i]
        numbered.append(piece)

    # Direct order (winning strategy): explicit instruction at the top
    parts = [
        "Classify each evidence piece below as supports, refutes, or not_enough_information "
        "relative to the claim. "
    ]
    if include_relevant_phrase:
        parts.append(
            "For supports or refutes, also provide relevant_phrase (a short quote from the evidence) "
            "and reason (one of the allowed values below). "
        )
    elif include_reason:
        parts.append(
            "For supports or refutes, also provide reason (one of the allowed values below). "
        )
    parts.append(
        "Return only a valid JSON array of objects with keys: id, stance"
        + (", relevant_phrase" if include_relevant_phrase else "")
        + (", reason" if include_reason else "")
        + ".\n\n"
    )

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
    if include_reason:
        reasons_str = ", ".join(f'"{r}"' for r in EVIDENCE_CLASSIFIER_REASONS)
        instructions += (
            ' When stance is "supports" or "refutes", include "reason" with exactly one of: '
            f"{reasons_str}. "
            "(same_number: evidence gives same number/finding; higher_number/smaller_number: "
            "claim implies a quantity, evidence gives higher/lower value; contradiction: evidence "
            "directly contradicts; partial_or_qualification: partial support/contradiction or "
            "qualification; different_scope_or_timeframe: different region, period, or definition.) "
            "When stance is \"not_enough_information\", use empty string for reason."
        )

    payload = {
        "claim": claim,
        "evidence_pieces": numbered,
        "instructions": instructions,
    }
    direct_order = "".join(parts)
    payload_str = json.dumps(payload, indent=2, ensure_ascii=False)
    return direct_order + payload_str


# Retry format instruction for advocates using the full SF label set
SPECTRUM_ADVOCATE_VERDICT_FORMAT = (
    "Respond with exactly one verdict in double parentheses. "
    "Choose from: "
    + ", ".join(f"(({c}))" for c in SPECTRUM_CATEGORIES)
    + ". "
    "Put your reasoning first, then the verdict at the very end."
)

# When mediator will see advocate proof: ask advocate to pick chunks that support their verdict
SPECTRUM_ADVOCATE_SELECT_CHUNKS_INSTRUCTION = (
    "After your verdict, list the chunk IDs (up to 3) that best support your verdict. "
    "Use the chunk_id values from the evidence. Format exactly: Selected chunks: <id1>, <id2>, <id3>. "
    "Pick chunks that provide the strongest evidence for the verdict you chose."
)

# Advocate output: one label + weighted distribution (model provides both)
SPECTRUM_ADVOCATE_VERDICT_FORMAT_LABEL_AND_WEIGHTS = (
    "Respond with: (1) your reasoning, (2) exactly one verdict in double parentheses, "
    "and (3) a JSON object of weights over the same categories that sum to 1.0. "
    "Verdict choices: " + ", ".join(f"(({c}))" for c in SPECTRUM_CATEGORIES) + ". "
    "After the verdict, output a JSON object with keys: "
    + ", ".join(SPECTRUM_CATEGORIES)
    + " and non-negative values summing to 1.0. Put reasoning first, then verdict, then JSON."
)

# Advocate output: weighted distribution only (we derive primary by argmax)
SPECTRUM_ADVOCATE_VERDICT_FORMAT_WEIGHTS_ONLY = (
    "Respond with: (1) your reasoning, then (2) a JSON object whose keys are exactly: "
    + ", ".join(SPECTRUM_CATEGORIES)
    + ". "
    "The values must be non-negative numbers that sum to 1.0 (your confidence per category). "
    "Do not output a single verdict in parentheses; output only reasoning and the JSON weights."
)

# Instruction for the mediator to output a JSON object with weights (sum to 1)
SPECTRUM_VERDICT_FORMAT = (
    "Provide a JSON object whose keys are exactly these category names (use underscores): "
    + ", ".join(SPECTRUM_CATEGORIES)
    + ". "
    "The values must be non-negative numbers that sum to 1.0. "
    "You may include a short explanation before or after the JSON. Output only valid JSON for the weights."
)


# ---------------------------------------------------------------------------
# Advocate primer — replaces the generic advocate_primer for Spectrum.
# Explains classified evidence AND the Science Feedback verdict scale.
# ---------------------------------------------------------------------------
spectrum_advocate_primer = dedent("""\
You are a scientific fact-checking Advocate with expertise in climate change,
climate science, environmental science, physics, and energy science.

## Your task
Evaluate a claim based SOLELY on the provided evidence and select the single
most appropriate verdict from the Science Feedback claim review scale.

## Understanding the evidence you receive
Each evidence piece has been pre-classified with a stance:
- "supports"                — evidence directly consistent with the claim
- "refutes"                 — evidence contradicts the claim
- "not_enough_information"  — evidence is irrelevant or too vague

Use these stance labels to focus your analysis: weigh supporting and refuting
evidence, and disregard pieces marked as not_enough_information unless they
add indirect context.

## Science Feedback verdict scale
Science Feedback distinguishes between two types of claims:

A) **Statements of fact** — rated on the ACCURATE / INACCURATE axis:
   Does the statement match available data and observations?
   - accurate            (Very High) — fully consistent with data, no missing context
   - mostly_accurate     (High)      — largely correct, needs minor clarification
   - inaccurate          (Very Low)  — directly contradicts available data

B) **Explanations, hypotheses, or theories** — rated on the CORRECT / INCORRECT axis:
   Has the explanation been well-tested and confirmed by observations?
   - correct             (Very High) — well-tested, predictions confirmed
   - mostly_correct      (High)      — well-tested, slightly overstated confidence
   - correct_but         (Neutral)   — correct with an unstated caveat
   - partially_correct   (Neutral)   — significantly overstates confidence
   - incorrect           (Very Low)  — predictions invalidated by observations

C) **Cross-cutting issues** (apply to both types):
   - lacks_context       (Neutral)   — omits observations/explanations that change the takeaway
   - imprecise           (Neutral)   — ill-defined terms, ambiguous meaning
   - unsupported         (Low)       — no adequate reference or evidence
   - misleading          (Low)       — element of truth but false overall impression
   - flawed_reasoning    (Low/Issue) — conclusion does not follow from premises

D) **Insufficient evidence**:
   - not_enough_information — the provided evidence is insufficient to assess the claim

First determine whether the claim is a **statement of fact** or an
**explanation/theory**, then choose the verdict that best matches the evidence.

## Response format
Provide a concise reasoning tied to the evidence, then your verdict in double
parentheses at the very end.  Example:

"The claim overstates the rate of warming reported in IPCC AR6. ((mostly_accurate))"

Choose EXACTLY ONE verdict from the label_options provided.
""")


# ---------------------------------------------------------------------------
# Mediator — lead fact-checker; picks exactly one of the 14 labels (no distribution).
# When mediator_mode is "formula", the strategy uses combine_advocate_weights_formula
# and does not call the mediator LLM.
# ---------------------------------------------------------------------------
SPECTRUM_MEDIATOR_LABELS_LIST = ", ".join(SPECTRUM_CATEGORIES)

SPECTRUM_MEDIATOR_VERDICT_FORMAT_INSTRUCTION = (
    f"Pick exactly one of these labels: {SPECTRUM_MEDIATOR_LABELS_LIST}. "
    "Output your final verdict as that single label in double parentheses, e.g. ((inaccurate)) or ((not_enough_information))."
)

SPECTRUM_MEDIATOR_USER_MESSAGE_SUFFIX = (
    "\n\nPick exactly one of the 14 labels above and provide your final verdict as ((label)), e.g. ((inaccurate))."
)

spectrum_mediator_primer = dedent("""\
You are a lead fact checker, receiving an assessment of the following claim
from one or more senior fact checkers (advocates). You will see their verdict(s)
and reasoning.

If multiple advocates disagree, your role is to come up with the best verdict.
If there is consensus or just one advocate, your only task is to pick the single
most likely verdict you see. If you see an obvious mistake, you may correct it.
Provide brief reasoning.

You must pick exactly one of the 14 Science Feedback verdict labels and output
it in double parentheses, e.g. ((inaccurate)) or ((mostly_accurate)).
Do not output a distribution or JSON — only one label in (( )) format.
""")
