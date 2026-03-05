import json
import logging
from typing import Callable, Optional

from llama_index.core.llms import ChatMessage

from factchecker.core.llm import load_llm

from factchecker.steps.verdict_retry import chat_with_verdict_retry

logger = logging.getLogger(__name__)

# Format we require from the LLM; repeated in retry instructions
MEDIATOR_VERDICT_FORMAT = (
    "Provide the final verdict as exactly one of: ((correct)), ((incorrect)), or ((not_enough_information)). "
    "Use double parentheses with the single word inside, nothing else."
)


def _default_mediator_parser(response_content: str) -> Optional[str]:
    """Extract final verdict from response using (( )) format. Returns verdict string or None."""
    start = response_content.find("((")
    end = response_content.find("))")
    if start == -1 or end == -1:
        return None
    verdict = response_content[start + 2 : end].strip().upper().replace(" ", "_")
    return verdict


class MediatorStep:
    """
    A step in the fact-checking process that mediates between multiple advocate verdicts.
    
    This class synthesizes multiple verdicts and their associated reasoning to produce
    a final consensus verdict on a claim's veracity.
    """

    def __init__(self, llm=None, options=None):
        """
        Initialize a MediatorStep instance.

        Args:
            llm: Language model instance to use for mediation. If None, loads default model.
            options (dict, optional): Configuration options including:
                - arbitrator_primer: Template for the system prompt
        """
        self.llm = llm if llm is not None else load_llm()
        self.options = options if options is not None else {}
        self.system_prompt = self.options.pop('system_prompt', '')
        self.max_retries = self.options.pop('max_retries', 3)
        self.verdict_parser: Optional[Callable[[str], Optional[str]]] = self.options.pop('verdict_parser', None)
        self.verdict_format_instruction: Optional[str] = self.options.pop('verdict_format_instruction', None)
        self.user_message_suffix: Optional[str] = self.options.pop('user_message_suffix', None)
        self.additional_options = {key: self.options.pop(key) for key in list(self.options.keys())}

    def synthesize_verdicts(
        self,
        verdicts_and_reasonings,
        claim: str,
        evidence_summary: Optional[str] = None,
    ):
        """
        Synthesize multiple verdicts and their reasoning into a final consensus verdict.

        Args:
            verdicts_and_reasonings (list): List of (verdict, reasoning) or (verdict, reasoning, weights_dict) per advocate
            claim (str): The claim being evaluated
            evidence_summary (str, optional): Most relevant supporting/refuting evidence to show the mediator

        Returns:
            str: The final consensus verdict (or weighted JSON string when using custom verdict_parser)
        """
        lines = []
        for item in verdicts_and_reasonings:
            verdict = item[0]
            reasoning = item[1] if len(item) > 1 else ""
            line = f"<verdict>{verdict}</verdict><reasoning>{reasoning}</reasoning>"
            if len(item) > 2 and item[2] is not None:
                weights = item[2]
                if isinstance(weights, dict):
                    line += f"<weight_distribution>{json.dumps(weights)}</weight_distribution>"
            lines.append(line)
        formatted_verdicts_and_reasonings = "\n".join(lines)
        
        user_content = (
            f"Here are the verdicts and reasonings of the different advocates:\n{formatted_verdicts_and_reasonings}\n"
            f"Please provide the final verdict as ((correct)), ((incorrect)), or ((not_enough_information)) for the claim: {claim}"
        )
        if evidence_summary:
            user_content += f"\n\nMost relevant evidence from the advocates:\n{evidence_summary}"
        if self.user_message_suffix:
            user_content += self.user_message_suffix
        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_content)
        ]

        valid_options = {key: value for key, value in self.additional_options.items() if key in ["response_format", "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]}
        parse_fn = self.verdict_parser if self.verdict_parser is not None else _default_mediator_parser
        format_instruction = self.verdict_format_instruction if self.verdict_format_instruction is not None else MEDIATOR_VERDICT_FORMAT
        result = chat_with_verdict_retry(
            messages,
            self.llm,
            parse_fn,
            format_instruction,
            self.max_retries,
            valid_options,
            logger,
        )
        return result if result is not None else "ERROR_PARSING_RESPONSE"