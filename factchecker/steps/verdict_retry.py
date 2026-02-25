"""
Shared retry logic for LLM verdict parsing.

When the model returns a verdict in the wrong format, we append a follow-up message
showing the last response and the required format, then retry. Used by AdvocateStep
and MediatorStep.
"""
import logging
from typing import Any, Callable, List, Optional, TypeVar

from llama_index.core.llms import ChatMessage

T = TypeVar("T")


def chat_with_verdict_retry(
    messages: List[ChatMessage],
    llm: Any,
    parse_fn: Callable[[str], Optional[T]],
    format_instruction: str,
    max_retries: int,
    chat_kwargs: dict,
    logger: logging.Logger,
) -> Optional[T]:
    """
    Call the LLM in a loop; parse each response with parse_fn. On invalid format,
    append a retry user message (last answer + format instruction) and try again.

    Args:
        messages: Initial chat messages (system + user).
        llm: LLM instance with .chat(messages, **kwargs).
        parse_fn: Function that returns parsed result or None if format invalid.
        format_instruction: Text to show the model when retrying (required format).
        max_retries: Maximum number of chat attempts.
        chat_kwargs: Extra kwargs for llm.chat().
        logger: Logger for attempt/retry messages.

    Returns:
        Parsed result from parse_fn, or None if all attempts failed.
    """
    for attempt in range(max_retries):
        response = llm.chat(messages, **chat_kwargs)
        response_content = response.message.content.strip()
        result = parse_fn(response_content)
        if result is not None:
            log_label = result[0] if isinstance(result, tuple) else result
            logger.info(
                "Attempt %s response (parsed): %s -> %s",
                attempt + 1,
                response_content[:200] + ("..." if len(response_content) > 200 else ""),
                log_label,
            )
            return result
        logger.warning("Attempt %s response (invalid format): %s", attempt + 1, response_content)
        logger.info("Model response on attempt %s (could not parse): %s", attempt + 1, response_content)
        if attempt < max_retries - 1:
            logger.info("Sending format feedback and retrying (attempt %s of %s).", attempt + 2, max_retries)
            preview = response_content[:500] + ("..." if len(response_content) > 500 else "")
            retry_user = (
                "Your previous response was in the wrong format.\n\n"
                "This was your last answer (format is wrong):\n{preview}\n\n"
                "Stick to the format: {format_instruction}"
            ).format(preview=preview, format_instruction=format_instruction)
            messages.append(ChatMessage(role="assistant", content=response_content))
            messages.append(ChatMessage(role="user", content=retry_user))
    return None
