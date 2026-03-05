import logging
from typing import Callable, List, Optional, Tuple, Union

from llama_index.core.llms import ChatMessage

from factchecker.config.config import DEFAULT_LABEL_OPTIONS
from factchecker.core.llm import load_llm
from factchecker.datastructures import LabelOption
from factchecker.prompts.advocate_prompts import (
    chunk_stats_from_classified,
    get_classified_evidence_user_prompt,
    get_default_system_prompt,
    get_default_user_prompt,
)
from factchecker.retrieval.abstract_retriever import AbstractRetriever
from factchecker.steps.evidence import EvidenceStep
from factchecker.steps.verdict_retry import chat_with_verdict_retry

logger = logging.getLogger(__name__)

# Format we require from the LLM; repeated in retry instructions
ADVOCATE_VERDICT_FORMAT = (
    "Respond with exactly one verdict in double parentheses: ((correct)), ((incorrect)), or ((not_enough_information)). "
    "Put your reasoning first, then the verdict at the end, e.g. \"Your reasoning here. ((correct))\"."
)


ADVOCATE_OUTPUT_MODES = ("label_only", "label_and_weights", "weights_only")


def _default_advocate_parser(response_content: str) -> Optional[Tuple[str, str]]:
    """Extract verdict and reasoning from response using (( )) format. Returns (label, reasoning) or None."""
    start = response_content.find("((")
    end = response_content.find("))")
    if start == -1 or end == -1:
        return None
    label = response_content[start + 2 : end].strip().upper().replace(" ", "_")
    reasoning = (response_content[:start].strip() + " " + response_content[end + 2 :].strip()).strip()
    return label, reasoning


def _normalize_parse_result(
    result: Union[
        Tuple[str, str],
        Tuple[Optional[str], str, Optional[dict]],
        Tuple[Optional[str], str, Optional[dict], Optional[List[str]]],
    ],
) -> Tuple[Optional[str], str, Optional[dict], Optional[List[str]]]:
    """Normalize parser output to (label_or_none, reasoning, weights_or_none, selected_chunk_ids_or_none)."""
    if result is None or len(result) < 2:
        return None, "", None, None
    if len(result) == 2:
        return result[0], result[1], None, None
    if len(result) == 3:
        return result[0], result[1], result[2], None
    return result[0], result[1], result[2] if len(result) > 2 else None, result[3] if len(result) > 3 else None


def _primary_from_label_and_weights(
    label_or_none: Optional[str],
    weights_or_none: Optional[dict],
) -> Optional[str]:
    """Compute primary verdict: use label if present, else argmax(weights)."""
    if label_or_none:
        return label_or_none
    if weights_or_none and isinstance(weights_or_none, dict) and len(weights_or_none) > 0:
        return max(weights_or_none, key=weights_or_none.get)
    return None


class AdvocateStep:
    """
    A step in the fact-checking process that acts as an advocate by evaluating claims based on evidence.
    
    This class retrieves relevant evidence for a claim and uses an LLM to evaluate whether the claim
    is supported by the evidence, producing a verdict and reasoning.

    Args:
        retriever (AbstractRetriever): Retriever instance to use for evidence retrieval.
        llm (TODO): Language model instance to use for evaluation. If None, loads default model.
        options (dict, optional): Configuration options for the advocate step including
        evidence_options (dict, optional): Configuration for evidence gathering including
        
    Attributes:
        retriever (AbstractRetriever): Retriever instance to use for evidence retrieval.
        llm (TODO): Language model instance to use for evaluation.
        options (dict): Configuration options for the advocate step.
        system_prompt (str): The system prompt to display to the user.
        label_options (dict): The available label options for the verdict.
        max_retries (int): The maximum number of retries to attempt when parsing the LLM response.
        chat_completion_options (dict): Additional options to pass to the LLM chat method.
    """

    def __init__(
            self, 
            retriever: AbstractRetriever,
            llm = None, # TODO: Add type hint
            options: dict = None,
            evidence_options: dict = None
        ) -> None:
        """Initialize an AdvocateStep instance."""
        self.retriever = retriever
        self.llm = llm if llm is not None else load_llm()
        self.options = options if options is not None else {}
        self.evidence_options = evidence_options if evidence_options is not None else {}
        self.system_prompt = self.options.pop('system_prompt', get_default_system_prompt())
        self.label_options = self.options.pop('label_options', DEFAULT_LABEL_OPTIONS)
        self.max_retries = self.options.pop('max_retries', 3)
        self.chat_completion_options = self.options.pop('chat_completion_options', {})
        # Optional custom parser: may return (label, reasoning) or (label?, reasoning, weights?).
        self.verdict_parser: Optional[Callable[[str], Optional[Union[Tuple[str, str], Tuple[Optional[str], str, Optional[dict]]]]]] = self.options.pop('verdict_parser', None)
        # Optional evidence classifier: when set, each retrieved chunk is
        # classified as supports/refutes/nei before being sent to the LLM.
        self.evidence_classifier = self.options.pop('evidence_classifier', None)
        self.verdict_format = self.options.pop('verdict_format', ADVOCATE_VERDICT_FORMAT)
        # Advocate output mode: label_only (default), label_and_weights, or weights_only.
        self.advocate_output_mode: str = self.options.pop('advocate_output_mode', 'label_only')
        if self.advocate_output_mode not in ADVOCATE_OUTPUT_MODES:
            self.advocate_output_mode = 'label_only'
        # "full" = advocate sees full chunk text; "relevant_only" = only relevant_phrase or placeholder
        self.advocate_evidence_display: str = self.options.pop('advocate_evidence_display', 'full')
        if self.advocate_evidence_display not in ('full', 'relevant_only'):
            self.advocate_evidence_display = 'full'

        # Initialize EvidenceStep
        self.evidence_step = EvidenceStep(
            retriever=retriever,
            options={
                **self.evidence_options,
            }
        )

    def retrieve_evidence(self, claim: str) -> list[str]:
        """
        Retrieve relevant evidence for a given claim.

        Args:
            claim (str): The claim for which to retrieve evidence.

        Returns:
            list[str]: A list of evidence pieces relevant to the claim.

        """
        return self.evidence_step.gather_evidence(claim)

    def evaluate_claim(self, claim: str) -> tuple[str, str, Optional[list], Optional[dict], Optional[list]]:
        """
        Evaluate a claim based on gathered evidence using the language model.

        Returns:
            (primary_verdict, reasoning, evidence_metadata, optional_weights, selected_chunk_ids).
            primary_verdict is always a single label (from parser or argmax(weights)).
            evidence_metadata is a list of dicts when evidence_classifier is set; else None.
            optional_weights is the advocate weight distribution when mode is label_and_weights or weights_only; else None.
            selected_chunk_ids is a list of chunk IDs the advocate chose as proof (when parser returns 4-tuple); else None.
        """
        evidence_metadata = None

        if self.evidence_classifier is not None:
            evidence_items = self.evidence_step.gather_evidence_with_metadata(claim)
            if evidence_items:
                classified = self.evidence_classifier.classify(claim, evidence_items)
                evidence_metadata = classified
                chunk_stats = chunk_stats_from_classified(classified)
                user_prompt = get_classified_evidence_user_prompt(
                    claim=claim,
                    classified_evidence=classified,
                    label_options=self.label_options,
                    evidence_display_mode=self.advocate_evidence_display,
                    chunk_stats=chunk_stats,
                )
            else:
                user_prompt = get_classified_evidence_user_prompt(
                    claim=claim,
                    classified_evidence=[],
                    label_options=self.label_options,
                    evidence_display_mode=self.advocate_evidence_display,
                    chunk_stats=chunk_stats_from_classified([]),
                )
        else:
            evidence_list = self.retrieve_evidence(claim)
            user_prompt = get_default_user_prompt(
                claim=claim, evidence=evidence_list, label_options=self.label_options
            )

        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        if evidence_metadata:
            logger.info(
                "Advocate evidence for claim (first 120 chars): %s",
                (claim[:120] + "..." if len(claim) > 120 else claim),
            )
            for j, chunk in enumerate(evidence_metadata):
                text_preview = (chunk.get("text") or "")[:200]
                if len(chunk.get("text") or "") > 200:
                    text_preview += "..."
                logger.info(
                    "  Chunk %s [%s] (relevant_phrase: %s): %s",
                    chunk.get("chunk_id", j),
                    chunk.get("stance", "?"),
                    (chunk.get("relevant_phrase") or "")[:80] or "(none)",
                    text_preview,
                )
        parse_fn = self.verdict_parser if self.verdict_parser is not None else _default_advocate_parser
        result = chat_with_verdict_retry(
            messages,
            self.llm,
            parse_fn,
            self.verdict_format,
            self.max_retries,
            self.chat_completion_options,
            logger,
        )
        if result is not None:
            label_or_none, reasoning, weights_or_none, selected_chunk_ids = _normalize_parse_result(result)
            primary = _primary_from_label_and_weights(label_or_none, weights_or_none)
            if primary is not None:
                return primary, reasoning, evidence_metadata, weights_or_none, selected_chunk_ids
        return "ERROR_PARSING_RESPONSE", "No reasoning available", evidence_metadata, None, None