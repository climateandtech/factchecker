import json
from typing import List, Optional, Sequence

from factchecker.steps.advocate import AdvocateStep
from factchecker.steps.mediator import MediatorStep
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever


def _build_evidence_summary_for_mediator(
    evidence_metadata_per_advocate: List[Optional[List[dict]]],
    use_relevant_only: bool = False,
) -> str:
    """
    Build a short summary of supporting and refuting evidence from per-advocate chunk metadata
    for the mediator prompt. Each advocate's chunks are lists of dicts with stance, text, relevant_phrase.
    If use_relevant_only is True, only the relevant_phrase is shown (or "[no relevant part]" when missing),
    and a statistics line is appended so NIN and chunks without a phrase are accounted for.
    """
    supporting: List[str] = []
    refuting: List[str] = []
    n_with_phrase = 0
    n_without_phrase = 0
    for meta_list in evidence_metadata_per_advocate or []:
        if not meta_list:
            continue
        for chunk in meta_list:
            if not isinstance(chunk, dict):
                continue
            stance = (chunk.get("stance") or "").strip().lower()
            phrase = (chunk.get("relevant_phrase") or "").strip()
            text = (chunk.get("text") or "").strip()
            if use_relevant_only:
                snippet = phrase if phrase else "[no relevant part]"
                if phrase:
                    n_with_phrase += 1
                else:
                    n_without_phrase += 1
            else:
                snippet = phrase if phrase else (text[:200] + "..." if len(text) > 200 else text)
            if not snippet and not use_relevant_only:
                continue
            if stance == "supports":
                supporting.append(snippet)
            elif stance == "refutes":
                refuting.append(snippet)
    lines = []
    if supporting:
        lines.append("Supporting:")
        for s in supporting[:10]:  # cap to avoid huge prompts
            lines.append(f"- {s}")
    if refuting:
        lines.append("Refuting:")
        for r in refuting[:10]:
            lines.append(f"- {r}")
    if use_relevant_only and (n_with_phrase > 0 or n_without_phrase > 0):
        lines.append(f"Statistics: {n_with_phrase} chunk(s) with relevant phrase, {n_without_phrase} without.")
    return "\n".join(lines) if lines else ""


def _verdict_wants_supports(verdict: str) -> Optional[bool]:
    """
    True = advocate says claim is right → show chunks that support the claim ("supports").
    False = advocate says claim is wrong → show chunks that refute the claim ("refutes").
    None = neutral (e.g. not_enough_information) → no proof chunks.
    """
    v = (verdict or "").strip().lower().replace(" ", "_")
    if v == "not_enough_information" or v == "not_enough_info":
        return None
    pro = (
        "accurate", "correct", "mostly_accurate", "mostly_correct",
        "correct_but", "partially_correct", "lacks_context", "imprecise", "unsupported",
    )
    con = ("inaccurate", "incorrect", "flawed_reasoning", "misleading")
    if v in pro:
        return True
    if v in con:
        return False
    return None


def _build_advocate_proof_for_mediator(
    evidence_metadata_per_advocate: List[Optional[List[dict]]],
    verdicts: List[str],
    max_chunks: int = 3,
    selected_chunk_ids_per_advocate: Optional[Sequence[Optional[List[str]]]] = None,
    use_relevant_only: bool = False,
) -> str:
    """
    Build a short summary of up to max_chunks chunks that the advocate uses to prove its verdict.
    - When selected_chunk_ids_per_advocate is provided and non-empty for an advocate, use only
      those chunk IDs (the advocate explicitly picked them).
    - Otherwise fall back to verdict-based selection: pro-claim → "supports" chunks,
      anti-claim → "refutes" chunks, neutral → none.
    If use_relevant_only is True, only the relevant_phrase is shown (or "[no relevant part]");
    a statistics line is appended for chunks with/without relevant phrase.
    """
    chosen: List[str] = []
    n_with_phrase = 0
    n_without_phrase = 0

    def _snippet(chunk: dict) -> tuple:
        phrase = (chunk.get("relevant_phrase") or "").strip()
        text = (chunk.get("text") or "").strip()
        if use_relevant_only:
            s = phrase if phrase else "[no relevant part]"
            return s, bool(phrase)
        s = phrase if phrase else (text[:200] + "..." if len(text) > 200 else text)
        return s, True

    for adv_idx, meta_list in enumerate(evidence_metadata_per_advocate or []):
        if len(chosen) >= max_chunks or not meta_list:
            continue
        selected_ids: Optional[List[str]] = None
        if selected_chunk_ids_per_advocate and adv_idx < len(selected_chunk_ids_per_advocate):
            sel = selected_chunk_ids_per_advocate[adv_idx]
            if sel:
                selected_ids = [str(s).strip() for s in sel if s]
        if selected_ids:
            id_set = set(selected_ids)
            for chunk in meta_list:
                if len(chosen) >= max_chunks:
                    break
                if not isinstance(chunk, dict):
                    continue
                cid = chunk.get("chunk_id")
                if cid is None or str(cid).strip() not in id_set:
                    continue
                snippet, has_phrase = _snippet(chunk)
                if snippet:
                    chosen.append(snippet)
                    if use_relevant_only:
                        if has_phrase:
                            n_with_phrase += 1
                        else:
                            n_without_phrase += 1
            continue
        verdict = verdicts[adv_idx] if adv_idx < len(verdicts) else ""
        use_supports = _verdict_wants_supports(verdict)
        if use_supports is None:
            continue
        want_stance = "supports" if use_supports else "refutes"
        for chunk in meta_list:
            if len(chosen) >= max_chunks:
                break
            if not isinstance(chunk, dict):
                continue
            stance = (chunk.get("stance") or "").strip().lower()
            if stance != want_stance:
                continue
            snippet, has_phrase = _snippet(chunk)
            if snippet:
                chosen.append(snippet)
                if use_relevant_only:
                    if has_phrase:
                        n_with_phrase += 1
                    else:
                        n_without_phrase += 1
    if not chosen:
        return ""
    lines = ["Chunks the advocate selected to support its verdict:"]
    for s in chosen:
        lines.append(f"- {s}")
    if use_relevant_only and (n_with_phrase > 0 or n_without_phrase > 0):
        lines.append(f"Statistics: {n_with_phrase} with relevant phrase, {n_without_phrase} without.")
    return "\n".join(lines)


class AdvocateMediatorStrategy:
    """
    A strategy that combines multiple advocates and a mediator for fact-checking claims.
    
    This strategy uses multiple advocates, each with their own evidence sources, to evaluate
    a claim independently. A mediator then synthesizes their verdicts into a final consensus.
    """

    def __init__(
            self, 
            indexer_options_list: list[dict], 
            retriever_options_list: list[dict], 
            advocate_options: dict, 
            evidence_options: dict, 
            mediator_options: dict,
        ) -> None:
        """
        Initialize an AdvocateMediatorStrategy instance.

        Args:
            indexer_options_list (list): List of options for initializing document indexers
            retriever_options_list (list): List of options for configuring retrievers
            advocate_options (dict): Configuration options for advocates
            evidence_options (dict): Configuration options for evidence step
            mediator_options (dict): Configuration options for the mediator
            
        """
        # Initialize indexers with their options
        self.indexers = [LlamaVectorStoreIndexer(options) for options in indexer_options_list]

        # Initialize retrievers with their options
        self.retrievers = [LlamaBaseRetriever(
            indexer=indexer,
            options=retriever_options
            ) 
        for retriever_options, indexer in zip(retriever_options_list, self.indexers, strict=True)]
        
        # Create advocate steps with proper options
        self.advocate_steps = []

        # Create advocate step for each retriever
        for retriever in self.retrievers:
            
            # Create advocate step
            advocate_step = AdvocateStep(
                retriever=retriever,
                options=advocate_options,
                evidence_options=evidence_options
            )

            self.advocate_steps.append(advocate_step)
            
        self.mediator_step = MediatorStep(options=mediator_options)
        self.mediator_mode = mediator_options.get("mediator_mode", "llm")
        self._formula_fn = mediator_options.get("formula_fn")
        self._mediator_proof_max_chunks = mediator_options.get("mediator_proof_max_chunks", 3)
        self._mediator_evidence_relevant_only = mediator_options.get("mediator_evidence_relevant_only", False)

    def evaluate_claim(self, claim):
        """
        Evaluate a claim using multiple advocates and a mediator.

        Args:
            claim (str): The claim to evaluate

        Returns:
            tuple: (final_verdict, verdicts, reasonings, evidence_metadata_per_advocate, advocate_weights_per_advocate).
            evidence_metadata_per_advocate is a list of list-of-dicts (one per advocate).
            advocate_weights_per_advocate is a list of optional dicts (one per advocate); None when advocate did not output weights.
        """
        results = [advocate.evaluate_claim(claim) for advocate in self.advocate_steps]
        verdicts = [r[0] for r in results]
        reasonings = [r[1] for r in results]
        evidence_metadata_per_advocate = [r[2] if len(r) > 2 else None for r in results]
        advocate_weights_per_advocate = [r[3] if len(r) > 3 else None for r in results]
        selected_chunk_ids_per_advocate = [r[4] if len(r) > 4 else None for r in results]

        # Each item is (verdict, reasoning) or (verdict, reasoning, weights_dict) for mediator
        verdicts_and_reasonings = []
        for v, r, w in zip(verdicts, reasonings, advocate_weights_per_advocate):
            if w is not None:
                verdicts_and_reasonings.append((v, r, w))
            else:
                verdicts_and_reasonings.append((v, r))

        if self.mediator_mode == "formula" and self._formula_fn is not None:
            _primary, combined_weights = self._formula_fn(advocate_weights_per_advocate)
            final_verdict = json.dumps(combined_weights)
        elif self.mediator_mode == "llm_with_evidence":
            evidence_summary = _build_evidence_summary_for_mediator(
                evidence_metadata_per_advocate,
                use_relevant_only=getattr(self, "_mediator_evidence_relevant_only", False),
            )
            final_verdict = self.mediator_step.synthesize_verdicts(
                verdicts_and_reasonings, claim, evidence_summary=evidence_summary
            )
        elif self.mediator_mode == "llm_with_advocate_proof":
            proof_summary = _build_advocate_proof_for_mediator(
                evidence_metadata_per_advocate,
                verdicts,
                max_chunks=getattr(self, "_mediator_proof_max_chunks", 3),
                selected_chunk_ids_per_advocate=selected_chunk_ids_per_advocate,
                use_relevant_only=getattr(self, "_mediator_evidence_relevant_only", False),
            )
            final_verdict = self.mediator_step.synthesize_verdicts(
                verdicts_and_reasonings, claim, evidence_summary=proof_summary
            )
        else:
            final_verdict = self.mediator_step.synthesize_verdicts(verdicts_and_reasonings, claim)

        return final_verdict, verdicts, reasonings, evidence_metadata_per_advocate, advocate_weights_per_advocate