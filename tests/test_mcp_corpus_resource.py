"""Tests for the `lxd://corpus/{path*}` MCP resource (B-STACK-4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lxd.mcp.server import _read_corpus_file


def _write(corpus: Path, rel: str, body: str) -> None:
    target = corpus / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def test_returns_markdown_body_for_known_relative_path(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus, "Guides/alpha.md", "# alpha\n\nbody.")

    text = _read_corpus_file(corpus_root=corpus, relative_path="Guides/alpha.md")

    assert text == "# alpha\n\nbody."


def test_path_traversal_dotdot_is_refused(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    secret = tmp_path / "secret.md"
    secret.write_text("you shall not see me", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="traversal"):
        _read_corpus_file(corpus_root=corpus, relative_path="../secret.md")


def test_absolute_path_is_refused(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    with pytest.raises(FileNotFoundError, match="Invalid corpus path"):
        _read_corpus_file(corpus_root=corpus, relative_path="/etc/passwd")


def test_missing_file_returns_filenotfound(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    with pytest.raises(FileNotFoundError, match="not found"):
        _read_corpus_file(corpus_root=corpus, relative_path="Guides/nope.md")


def test_non_text_extension_is_refused(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus, "Guides/diagram.png", "fake png bytes")

    with pytest.raises(ValueError, match="text suffixes"):
        _read_corpus_file(corpus_root=corpus, relative_path="Guides/diagram.png")


def test_symlink_escaping_corpus_is_refused(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("escape", encoding="utf-8")
    (corpus / "escape.md").symlink_to(outside)

    with pytest.raises(FileNotFoundError, match="traversal"):
        _read_corpus_file(corpus_root=corpus, relative_path="escape.md")


def test_empty_path_is_refused(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    with pytest.raises(FileNotFoundError, match="Invalid corpus path"):
        _read_corpus_file(corpus_root=corpus, relative_path="")
