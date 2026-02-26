"""Tests for AbstractIndexer.initialize_index load/save persistence behavior."""
import os
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from llama_index.core import Document

from factchecker.indexing.abstract_indexer import AbstractIndexer


class MinimalConcreteIndexer(AbstractIndexer):
    """Minimal concrete indexer to test AbstractIndexer.initialize_index behavior."""

    def __init__(self, options: Optional[Dict[str, Any]] = None, *, calls: Optional[Dict[str, int]] = None):
        super().__init__(options)
        self.calls = calls if calls is not None else {}

    def build_index(self, documents: List[Document]) -> None:
        self.calls["build_index"] = self.calls.get("build_index", 0) + 1
        self.index = "built"

    def save_index(self, index_path: Optional[str] = None) -> None:
        self.calls["save_index"] = self.calls.get("save_index", 0) + 1

    def load_index(self) -> None:
        self.calls["load_index"] = self.calls.get("load_index", 0) + 1
        self.index = "loaded"

    def add_to_index(self, documents: List[Any]) -> None:
        raise NotImplementedError

    def delete_from_index(self, document_ids: List[Any]) -> None:
        raise NotImplementedError


def test_initialize_index_loads_when_index_path_exists(tmp_path):
    """When index_path is set and path exists, initialize_index calls load_index (no build/save)."""
    (tmp_path / "some_file").write_text("x")
    calls = {}
    indexer = MinimalConcreteIndexer(
        {"index_path": str(tmp_path), "documents": [Document(text="d")]},
        calls=calls,
    )
    with patch.object(MinimalConcreteIndexer, "check_persisted_index_exists", return_value=True):
        indexer.initialize_index()
    assert indexer.index == "loaded"
    assert calls.get("load_index") == 1
    assert calls.get("build_index", 0) == 0
    assert calls.get("save_index", 0) == 0


def test_initialize_index_builds_and_saves_when_index_path_set_but_not_exists(tmp_path):
    """When index_path is set but path does not exist, initialize_index builds then saves."""
    index_path = tmp_path / "missing"
    assert not index_path.exists()
    calls = {}
    indexer = MinimalConcreteIndexer(
        {"index_path": str(index_path), "documents": [Document(text="d")]},
        calls=calls,
    )
    indexer.initialize_index()
    assert indexer.index == "built"
    assert calls.get("build_index") == 1
    assert calls.get("save_index") == 1
    assert calls.get("load_index", 0) == 0


def test_initialize_index_build_only_when_no_index_path(tmp_path):
    """When index_path is not set, initialize_index only builds (no load, no save)."""
    calls = {}
    indexer = MinimalConcreteIndexer(
        {"documents": [Document(text="d")]},
        calls=calls,
    )
    indexer.initialize_index()
    assert indexer.index == "built"
    assert calls.get("build_index") == 1
    assert calls.get("save_index", 0) == 0
    assert calls.get("load_index", 0) == 0


def test_initialize_index_skips_when_index_already_set():
    """When index is already set, initialize_index does nothing."""
    calls = {}
    indexer = MinimalConcreteIndexer({"documents": [Document(text="d")]}, calls=calls)
    indexer.index = "already_there"
    indexer.initialize_index()
    assert indexer.index == "already_there"
    assert calls.get("build_index", 0) == 0
    assert calls.get("save_index", 0) == 0
    assert calls.get("load_index", 0) == 0


def test_check_persisted_index_exists(tmp_path):
    """check_persisted_index_exists returns True only when index_path exists."""
    indexer = MinimalConcreteIndexer({})
    assert indexer.check_persisted_index_exists() is False
    indexer.index_path = "/nonexistent/path"
    assert indexer.check_persisted_index_exists() is False
    indexer.index_path = str(tmp_path)
    assert indexer.check_persisted_index_exists() is True
