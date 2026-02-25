"""Tests for the shared verdict retry helper (no LLM/ollama imports)."""
import logging
from unittest.mock import MagicMock

import pytest
from llama_index.core.llms import ChatMessage

from factchecker.steps.verdict_retry import chat_with_verdict_retry


def test_parsed_on_first_try_returns_result():
    """First response parses; no retry."""
    messages = [ChatMessage(role="user", content="Hi")]
    llm = MagicMock()
    llm.chat.return_value = MagicMock(message=MagicMock(content="((correct))"))
    parse = lambda s: "CORRECT" if "((" in s and "))" in s else None
    logger = logging.getLogger("test")
    result = chat_with_verdict_retry(
        messages, llm, parse, "Use ((...))", 3, {}, logger
    )
    assert result == "CORRECT"
    assert llm.chat.call_count == 1


def test_retry_appends_format_feedback_then_succeeds():
    """First response unparseable; retry message includes format instruction; second parses."""
    messages = [ChatMessage(role="user", content="Hi")]
    llm = MagicMock()
    llm.chat.side_effect = [
        MagicMock(message=MagicMock(content="invalid")),
        MagicMock(message=MagicMock(content="((incorrect))")),
    ]
    def parse(s):
        if "((" not in s or "))" not in s:
            return None
        start, end = s.find("((") + 2, s.find("))")
        return s[start:end].strip().upper().replace(" ", "_")
    format_instruction = "Use ((correct)) or ((incorrect))"
    logger = logging.getLogger("test")
    result = chat_with_verdict_retry(
        messages, llm, parse, format_instruction, 2, {}, logger
    )
    assert result == "INCORRECT"
    assert llm.chat.call_count == 2
    second_messages = llm.chat.call_args[0][0]
    retry_user = next(m for m in second_messages if m.role == "user" and "wrong format" in m.content)
    assert "Your previous response was in the wrong format" in retry_user.content
    assert "invalid" in retry_user.content
    assert format_instruction in retry_user.content


def test_all_retries_fail_returns_none():
    """All attempts unparseable; returns None."""
    messages = [ChatMessage(role="user", content="Hi")]
    llm = MagicMock()
    llm.chat.return_value = MagicMock(message=MagicMock(content="nope"))
    parse = lambda s: None
    logger = logging.getLogger("test")
    result = chat_with_verdict_retry(
        messages, llm, parse, "Format", 2, {}, logger
    )
    assert result is None
    assert llm.chat.call_count == 2
