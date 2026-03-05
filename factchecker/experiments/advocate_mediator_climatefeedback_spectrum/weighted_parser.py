"""
Parser for Spectrum mediator and advocate output: extract JSON weights, markdown-style
distribution, or ((verdict)) single label. Advocate parsers return (label?, reasoning, weights?).
With chunk selection: (label?, reasoning, weights?, selected_chunk_ids).
"""
import json
import re
from typing import List, Optional, Tuple

from factchecker.experiments.advocate_mediator_climatefeedback_spectrum.advocate_mediator_climatefeedback_spectrum_prompts import (
    SPECTRUM_CATEGORIES,
)


def _normalize_key(key: str) -> str:
    """Normalize category key to canonical Science Feedback verdict (snake_case)."""
    k = key.strip().lower().replace(" ", "_").replace("-", "_")
    if k == "not_enough_info":
        return "not_enough_information"
    return k


def _parse_markdown_style_weights(text: str) -> Optional[dict]:
    """
    Parse lines like "* accurate: 0.00" or "* lacks_context: 0.50" into a weight dict.
    Many models output a bullet list instead of raw JSON; accept that so we don't lose the distribution.
    """
    weights = {}
    # Match * key: value or * key = value
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("*"):
            continue
        line = line.lstrip("*").strip()
        m = re.match(r"^([a-z_]+)\s*[:=]\s*([\d.]+)", line, re.I)
        if m:
            key = _normalize_key(m.group(1))
            try:
                val = float(m.group(2))
            except ValueError:
                continue
            if key in SPECTRUM_CATEGORIES:
                weights[key] = val
    if not weights:
        return None
    return _normalize_weights_dict(weights)


def _normalize_weights_dict(weights: dict) -> dict:
    """Ensure all SPECTRUM_CATEGORIES present and values sum to 1."""
    for c in SPECTRUM_CATEGORIES:
        if c not in weights:
            weights[c] = 0.0
    total = sum(weights.values())
    if total > 0:
        for c in weights:
            weights[c] = weights[c] / total
    return weights


def _extract_weights_from_text(text: str) -> Optional[dict]:
    """Extract a normalized weight dict from text (JSON or markdown-style). Returns None if none found."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            raw = json.loads(text[start : end + 1])
            if not isinstance(raw, dict):
                return None
            weights = {}
            for k, v in raw.items():
                norm = _normalize_key(k)
                if norm in SPECTRUM_CATEGORIES:
                    try:
                        weights[norm] = float(v)
                    except (TypeError, ValueError):
                        weights[norm] = 0.0
            if weights:
                return _normalize_weights_dict(weights)
        except (json.JSONDecodeError, ValueError):
            pass
    md_weights = _parse_markdown_style_weights(text)
    return md_weights


def _extract_selected_chunk_ids(text: str) -> List[str]:
    """
    Extract chunk IDs from a line like "Selected chunks: id1, id2, id3" or
    "Selected chunk IDs: id1; id2". Returns list of stripped non-empty IDs.
    """
    ids: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if "selected chunk" not in line.lower():
            continue
        idx = line.lower().find("selected chunk")
        rest = line[idx:]
        colon = rest.find(":")
        if colon == -1:
            continue
        rest = rest[colon + 1 :].strip()
        for part in re.split(r"[,;\n]", rest):
            pid = part.strip()
            if pid:
                ids.append(pid)
        break
    return ids[:10]  # cap at 10


def _extract_advocate_label_and_reasoning(text: str) -> Tuple[Optional[str], str]:
    """Extract ((category)) label (if any) and reasoning. Label normalized to snake_case and must be in SPECTRUM_CATEGORIES."""
    match = re.search(r"\(\(\s*([^)]+)\s*\)\)", text)
    label = None
    if match:
        raw = match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        if raw == "not_enough_info":
            raw = "not_enough_information"
        if raw in SPECTRUM_CATEGORIES:
            label = raw
    reasoning = text
    if match:
        reasoning = (text[: match.start()].strip() + " " + text[match.end() :].strip()).strip()
    return label, reasoning


def parse_mediator_single_label(response_content: str) -> Optional[str]:
    """
    Parse mediator response when output is a single label: extract ((label)) and return
    the matching SPECTRUM_CATEGORIES entry, or None if not found or not in categories.
    """
    text = (response_content or "").strip()
    match = re.search(r"\(\(\s*([^)]+)\s*\)\)", text)
    if not match:
        return None
    raw = match.group(1).strip()
    norm = _normalize_key(raw)
    if norm in SPECTRUM_CATEGORIES:
        return norm
    return None


def parse_advocate_response_label_and_weights(response_content: str) -> Optional[Tuple[str, str, dict]]:
    """
    Parse advocate response when mode is label_and_weights: extract single label, reasoning, and weight dict.
    Returns (label, reasoning, weights) or None if required parts missing (label and weights both required).
    """
    text = response_content.strip()
    label, reasoning = _extract_advocate_label_and_reasoning(text)
    weights = _extract_weights_from_text(text)
    if label is not None and weights is not None:
        return (label, reasoning, weights)
    return None


def parse_advocate_response_label_only_with_chunk_selection(
    response_content: str,
) -> Optional[Tuple[str, str, None, List[str]]]:
    """
    Parse advocate response when we ask for verdict + selected chunk IDs (label_only mode).
    Returns (label, reasoning, None, selected_chunk_ids) or None if label missing.
    selected_chunk_ids may be empty if the advocate did not list any.
    """
    text = (response_content or "").strip()
    label, reasoning = _extract_advocate_label_and_reasoning(text)
    if label is None:
        return None
    selected = _extract_selected_chunk_ids(text)
    return (label, reasoning, None, selected)


def parse_advocate_response_weights_only(response_content: str) -> Optional[Tuple[None, str, dict]]:
    """
    Parse advocate response when mode is weights_only: extract reasoning and weight dict only.
    Returns (None, reasoning, weights) or None if weights missing.
    """
    text = response_content.strip()
    weights = _extract_weights_from_text(text)
    if weights is None:
        return None
    # Reasoning: text before the JSON block, or full text if no JSON
    start = text.find("{")
    if start != -1:
        reasoning = text[:start].strip()
    else:
        reasoning = text
    return (None, reasoning, weights)


def parse_weighted_response(response_content: str) -> Optional[str]:
    """
    Parse mediator response: if it contains a valid JSON object of category weights,
    return json.dumps(weights) string. Also accepts markdown-style "* key: value" lists.
    Otherwise try ((correct)) / ((incorrect)) / ((not_enough_information))
    and return that verdict string (normalized). Returns None if unparseable.

    Returns:
        Either a JSON string of weights (for spectrum) or a single verdict string (fallback).
    """
    text = response_content.strip()

    # 1. Try to extract JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            json_str = text[start : end + 1]
            raw = json.loads(json_str)
            if not isinstance(raw, dict):
                raise ValueError("Not a dict")
            weights = {}
            for k, v in raw.items():
                norm = _normalize_key(k)
                if norm in SPECTRUM_CATEGORIES:
                    try:
                        weights[norm] = float(v)
                    except (TypeError, ValueError):
                        weights[norm] = 0.0
            for c in SPECTRUM_CATEGORIES:
                if c not in weights:
                    weights[c] = 0.0
            total = sum(weights.values())
            if total > 0:
                for c in weights:
                    weights[c] = weights[c] / total
            return json.dumps(weights)
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Try markdown-style "* key: value" lines (so we don't lose the distribution on retry)
    md_weights = _parse_markdown_style_weights(text)
    if md_weights is not None:
        return json.dumps(md_weights)

    # 3. Fallback: ((correct)), ((incorrect)), ((not_enough_information))
    match = re.search(r"\(\(\s*([^)]+)\s*\)\)", text)
    if match:
        verdict = match.group(1).strip().upper().replace(" ", "_")
        if verdict in ("CORRECT", "INCORRECT", "NOT_ENOUGH_INFORMATION"):
            return verdict
        if verdict in ("NOT_ENOUGH_INFO", "NEI"):
            return "NOT_ENOUGH_INFORMATION"
    return None


def combine_advocate_weights_formula(
    advocate_weights_per_advocate: List[Optional[dict]],
) -> Tuple[str, dict]:
    """
    Combine advocate weight distributions by averaging. Final verdict = argmax of averaged weights.

    Args:
        advocate_weights_per_advocate: List of weight dicts (one per advocate); entries can be None.

    Returns:
        (primary_verdict, combined_weights_dict). combined_weights_dict is normalized (sum=1).
        If no valid weights, returns ("not_enough_information", {not_enough_information: 1.0, ...}).
    """
    valid = [w for w in advocate_weights_per_advocate if w and isinstance(w, dict) and len(w) > 0]
    if not valid:
        out = {c: 0.0 for c in SPECTRUM_CATEGORIES}
        out["not_enough_information"] = 1.0
        return "not_enough_information", out
    combined = {}
    for c in SPECTRUM_CATEGORIES:
        combined[c] = sum(w.get(c, 0.0) for w in valid) / len(valid)
    combined = _normalize_weights_dict(combined)
    primary = max(combined, key=combined.get)
    return primary, combined
