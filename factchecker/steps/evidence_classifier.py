"""
EvidenceClassifierStep — classifies each retrieved chunk as
supports / refutes / not_enough_information relative to a claim.

One LLM call per invocation (all chunks classified in a single request).
Can optionally request and parse a "relevant_phrase" per chunk.

Prompts can be overridden per experiment via options:
  - system_prompt: str
  - user_prompt_fn: callable (claim, evidence_items, include_relevant_phrase) -> str
"""
import json
import logging
from typing import Callable, List, Optional, Union

from llama_index.core.llms import ChatMessage

from factchecker.core.llm import load_llm
from factchecker.prompts.evidence_classifier_prompts import (
    EVIDENCE_CLASSIFIER_SYSTEM_PROMPT,
    EVIDENCE_STANCES,
    get_evidence_classifier_user_prompt,
)

logger = logging.getLogger(__name__)


def _has_required_phrases(parsed: dict, n_expected: int) -> bool:
    """True if every chunk with stance supports/refutes has a non-empty relevant_phrase."""
    for i in range(n_expected):
        entry = parsed.get(i, {})
        stance = entry.get("stance", "")
        if stance in ("supports", "refutes"):
            phrase = (entry.get("relevant_phrase") or "").strip()
            if not phrase:
                return False
    return True


def _has_required_reasons(parsed: dict, n_expected: int) -> bool:
    """True if every chunk with stance supports/refutes has a non-empty reason (when reason is used)."""
    for i in range(n_expected):
        entry = parsed.get(i, {})
        stance = entry.get("stance", "")
        if stance in ("supports", "refutes"):
            reason = (entry.get("reason") or "").strip()
            if not reason:
                return False
    return True


def _normalize_evidence_items(
    evidence_items: Union[List[str], List[dict]],
) -> List[dict]:
    """Normalize to list of dicts with chunk_id and text."""
    out = []
    for i, item in enumerate(evidence_items):
        if isinstance(item, str):
            out.append({"chunk_id": str(i), "text": item})
        else:
            out.append({
                "chunk_id": item.get("chunk_id", str(i)),
                "text": item.get("text", ""),
            })
    return out


class EvidenceClassifierStep:
    """Classify a list of evidence texts against a claim (supports / refutes / nei)."""

    def __init__(self, llm=None, options: Optional[dict] = None):
        opts = dict(options) if options else {}
        self.llm = llm if llm is not None else load_llm()
        self.max_retries: int = opts.pop("max_retries", 3)
        self.system_prompt: str = opts.pop("system_prompt", EVIDENCE_CLASSIFIER_SYSTEM_PROMPT)
        self.include_relevant_phrase: bool = opts.pop("include_relevant_phrase", True)
        self.include_reason: bool = opts.pop("include_reason", False)
        self.allowed_reasons: tuple = opts.pop("allowed_reasons", ())
        self.user_prompt_fn: Optional[Callable] = opts.pop("user_prompt_fn", None)
        if self.user_prompt_fn is None:
            self.user_prompt_fn = get_evidence_classifier_user_prompt

    # ------------------------------------------------------------------
    def classify(
        self,
        claim: str,
        evidence_items: Union[List[str], List[dict]],
    ) -> List[dict]:
        """
        Classify each evidence piece relative to *claim*.

        evidence_items: List of str (plain text) or List of dict with "text" and optional "chunk_id".

        Returns:
            List of dicts with "chunk_id", "text", "stance", and optionally "relevant_phrase".

        Raises:
            ValueError: If the LLM response cannot be parsed after all retries (no default to NEI).
        """
        items = _normalize_evidence_items(evidence_items)
        if not items:
            return []

        evidence_texts = [e["text"] for e in items]
        chunk_ids = [e["chunk_id"] for e in items]
        n_expected = len(items)
        user_prompt = self._build_user_prompt(claim, evidence_texts, chunk_ids)
        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        last_raw = ""
        for attempt in range(1, self.max_retries + 1):
            raw = self._call_llm(messages)
            last_raw = raw
            parsed = self._parse_response(
                raw,
                n_expected,
                self.include_relevant_phrase,
                self.include_reason,
                self.allowed_reasons,
            )
            if parsed and all(i in parsed for i in range(n_expected)):
                missing_phrase = (
                    self.include_relevant_phrase
                    and not _has_required_phrases(parsed, n_expected)
                )
                missing_reason = (
                    self.include_reason
                    and not _has_required_reasons(parsed, n_expected)
                )
                if (missing_phrase or missing_reason) and attempt < 3:
                    if missing_phrase:
                        logger.warning(
                            "EvidenceClassifier: supports/refutes missing required relevant_phrase; retrying (attempt %s).",
                            attempt,
                        )
                    if missing_reason:
                        logger.warning(
                            "EvidenceClassifier: supports/refutes missing required reason; retrying (attempt %s).",
                            attempt,
                        )
                else:
                    if missing_phrase and attempt >= 3:
                        logger.info(
                            "EvidenceClassifier: accepting response after 2 retries; some supports/refutes have empty relevant_phrase."
                        )
                    if missing_reason and attempt >= 3:
                        logger.info(
                            "EvidenceClassifier: accepting response after 2 retries; some supports/refutes have empty reason."
                        )
                    return [
                        {
                            "chunk_id": items[i]["chunk_id"],
                            "text": items[i]["text"],
                            "stance": parsed.get(i, {}).get("stance", "not_enough_information"),
                            **(
                                {"relevant_phrase": parsed.get(i, {}).get("relevant_phrase", "")}
                                if self.include_relevant_phrase
                                else {}
                            ),
                            **(
                                {"reason": parsed.get(i, {}).get("reason", "")}
                                if self.include_reason
                                else {}
                            ),
                        }
                        for i in range(n_expected)
                    ]
            else:
                logger.warning(
                    "EvidenceClassifier: parse attempt %s/%s failed (got %s entries, expected %s). Raw response (first 600 chars): %s",
                    attempt,
                    self.max_retries,
                    len(parsed) if parsed else 0,
                    n_expected,
                    (raw[:600] + "..." if len(raw) > 600 else raw) or "(empty)",
                )

        logger.error(
            "EvidenceClassifier: all %s retries exhausted. Last raw response (first 800 chars): %s",
            self.max_retries,
            (last_raw[:800] + "..." if len(last_raw) > 800 else last_raw) or "(empty)",
        )
        raise ValueError(
            "Evidence classifier failed to produce a valid parse after %s retries; cannot default to NEI."
            % self.max_retries
        )

    def _build_user_prompt(
        self, claim: str, evidence_texts: List[str], chunk_ids: List[str]
    ) -> str:
        if self.user_prompt_fn is None:
            return get_evidence_classifier_user_prompt(
                claim, evidence_texts, self.include_relevant_phrase, chunk_ids
            )
        sig = getattr(self.user_prompt_fn, "__code__", None)
        varnames = (sig.co_varnames if sig else [])[: (sig.co_argcount if sig else 0)]
        if "include_reason" in varnames:
            return self.user_prompt_fn(
                claim,
                evidence_texts,
                self.include_relevant_phrase,
                chunk_ids,
                self.include_reason,
            )
        if "include_relevant_phrase" in varnames:
            return self.user_prompt_fn(
                claim, evidence_texts, self.include_relevant_phrase, chunk_ids
            )
        return self.user_prompt_fn(claim, evidence_texts)

    # ------------------------------------------------------------------
    def _call_llm(self, messages: list) -> str:
        response = self.llm.chat(messages)
        return response.message.content

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(
        raw: str,
        n_expected: int,
        include_relevant_phrase: bool = False,
        include_reason: bool = False,
        allowed_reasons: tuple = (),
    ) -> dict:
        """
        Parse LLM JSON array into {id: {stance, relevant_phrase?, reason?}}.
        reason is normalized to one of allowed_reasons or empty string.
        """
        text = raw.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            logger.warning("EvidenceClassifier: no JSON array found in response")
            return {}
        try:
            arr = json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            logger.warning("EvidenceClassifier: JSON parse failed: %s", e)
            return {}

        allowed_set = frozenset(str(r).strip().lower().replace(" ", "_") for r in allowed_reasons)
        mapping: dict[int, dict] = {}
        for item in arr:
            if not isinstance(item, dict):
                continue
            idx = item.get("id")
            stance = str(item.get("stance", "")).strip().lower().replace(" ", "_")
            if stance not in EVIDENCE_STANCES:
                stance = "not_enough_information"
            if isinstance(idx, int) and 0 <= idx < n_expected:
                entry = {"stance": stance}
                if include_relevant_phrase:
                    rp = item.get("relevant_phrase", "")
                    entry["relevant_phrase"] = str(rp).strip() if rp else ""
                if include_reason:
                    raw_reason = str(item.get("reason", "")).strip().lower().replace(" ", "_")
                    entry["reason"] = raw_reason if raw_reason in allowed_set else ""
                mapping[idx] = entry
        return mapping
