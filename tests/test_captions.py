"""Tests for PNG asset captioning (Phase 4: caption + embed captions, not full image embeds).

No live OpenAI calls: every test monkeypatches the client/captioner/embedder
boundaries in ``lxd.ingest.captions``.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lxd.ingest.captions import (
    build_caption_chunk_record,
    caption_asset_source,
    caption_png,
)


def _config(
    *,
    caption_model: str = "gpt-4o-mini",
    caption_timeout_secs: int = 60,
    caption_max_tokens: int = 200,
) -> SimpleNamespace:
    return SimpleNamespace(
        multimodal=SimpleNamespace(
            captions_enabled=True,
            caption_model=caption_model,
            caption_timeout_secs=caption_timeout_secs,
            caption_max_tokens=caption_max_tokens,
        ),
        openai=SimpleNamespace(api_key_env="OPENAI_API_KEY"),
        models=SimpleNamespace(embed="text-embedding-3-small", embed_dims=1536),
        chunking=SimpleNamespace(tokenizer_backend="tiktoken", tokenizer_name="cl100k_base"),
    )


def _mock_openai_response(content: str | None) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


# ---------------------------------------------------------------------------
# caption_png
# ---------------------------------------------------------------------------


def test_caption_png_returns_empty_on_missing_file(tmp_path: Path) -> None:
    """A path that does not exist yields "" rather than raising."""
    missing = tmp_path / "does-not-exist.png"
    result = caption_png(missing, _config())
    assert result == ""


def test_caption_png_returns_empty_on_empty_file(tmp_path: Path) -> None:
    """A zero-byte PNG yields "" without attempting an API call."""
    empty_png = tmp_path / "empty.png"
    empty_png.write_bytes(b"")
    result = caption_png(empty_png, _config())
    assert result == ""


def test_caption_png_returns_empty_when_client_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing API key (RuntimeError from the client factory) yields ""."""
    png_path = tmp_path / "diagram.png"
    png_path.write_bytes(b"\x89PNG fake bytes")

    def _raise(_api_key_env: str) -> None:
        raise RuntimeError("Environment variable 'OPENAI_API_KEY' is not set.")

    monkeypatch.setattr("lxd.ingest.captions.get_sync_openai_client", _raise)

    result = caption_png(png_path, _config())
    assert result == ""


def test_caption_png_returns_empty_on_openai_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OpenAI SDK failure during the vision call yields "" (best-effort)."""
    png_path = tmp_path / "diagram.png"
    png_path.write_bytes(b"\x89PNG fake bytes")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("upstream 500")
    monkeypatch.setattr(
        "lxd.ingest.captions.get_sync_openai_client", lambda _env: mock_client
    )

    result = caption_png(png_path, _config())
    assert result == ""


def test_caption_png_returns_stripped_caption_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful vision call returns the stripped message content."""
    png_path = tmp_path / "diagram.png"
    png_path.write_bytes(b"\x89PNG fake bytes")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(
        "  A funnel diagram showing the ADDIE model stages.  "
    )
    monkeypatch.setattr(
        "lxd.ingest.captions.get_sync_openai_client", lambda _env: mock_client
    )

    result = caption_png(png_path, _config())
    assert result == "A funnel diagram showing the ADDIE model stages."

    # The vision request must carry a base64 data URI, not a bare path.
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    user_content = call_kwargs["messages"][1]["content"]
    image_part = next(part for part in user_content if part["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_caption_png_returns_empty_when_content_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A response with no message content yields ""."""
    png_path = tmp_path / "diagram.png"
    png_path.write_bytes(b"\x89PNG fake bytes")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(None)
    monkeypatch.setattr(
        "lxd.ingest.captions.get_sync_openai_client", lambda _env: mock_client
    )

    result = caption_png(png_path, _config())
    assert result == ""


# ---------------------------------------------------------------------------
# build_caption_chunk_record — chunk assembly helper
# ---------------------------------------------------------------------------


def test_build_caption_chunk_record_assembles_expected_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunk assembly produces a searchable, clearly-labelled ChunkRecord."""
    fake_vector = [0.1, 0.2, 0.3]
    monkeypatch.setattr(
        "lxd.ingest.captions.embed_chunk_text", lambda _config, _text: fake_vector
    )

    record = build_caption_chunk_record(
        source_rel_path="research/diagrams/addie.png",
        source_filename="addie.png",
        source_domain="research",
        source_type="image_png",
        content_hash="abc123",
        document_id="doc-1",
        caption_text="A funnel diagram showing the ADDIE model stages.",
        config=_config(),
    )

    assert record.source_rel_path == "research/diagrams/addie.png"
    assert record.source_type == "image_png"
    assert record.document_id == "doc-1"
    assert record.text == "A funnel diagram showing the ADDIE model stages."
    assert record.vector == fake_vector
    assert record.embedding_model == "text-embedding-3-small"
    assert record.embedding_dims == 1536
    assert record.chunk_index == 0
    assert record.chunk_occurrence == 0
    # citation_label must clearly mark this as an asset caption, not prose.
    assert "image caption" in record.citation_label
    assert "addie.png" in record.citation_label
    assert '"kind": "image_caption"' in record.metadata_json


def test_build_caption_chunk_record_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-running against an unchanged caption yields the same chunk_id."""
    monkeypatch.setattr(
        "lxd.ingest.captions.embed_chunk_text", lambda _config, _text: [0.0]
    )

    kwargs = dict(
        source_rel_path="a/b.png",
        source_filename="b.png",
        source_domain="a",
        source_type="image_png",
        content_hash="hash-1",
        document_id="doc-1",
        caption_text="Same caption text.",
        config=_config(),
    )
    first = build_caption_chunk_record(**kwargs)
    second = build_caption_chunk_record(**kwargs)
    assert first.chunk_id == second.chunk_id

    different_caption = build_caption_chunk_record(
        **{**kwargs, "caption_text": "A different caption."}
    )
    assert different_caption.chunk_id != first.chunk_id


# ---------------------------------------------------------------------------
# caption_asset_source — orchestration entrypoint
# ---------------------------------------------------------------------------


def test_caption_asset_source_returns_none_when_caption_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the captioner produces no text, no chunk is assembled."""
    monkeypatch.setattr("lxd.ingest.captions.caption_png", lambda _path, _config: "")

    result = caption_asset_source(
        absolute_path=tmp_path / "asset.png",
        source_rel_path="asset.png",
        source_type="image_png",
        source_domain="root",
        content_hash="hash",
        document_id="doc-1",
        config=_config(),
    )
    assert result is None


def test_caption_asset_source_builds_chunk_from_monkeypatched_captioner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful (monkeypatched) caption flows through to a ChunkRecord."""
    monkeypatch.setattr(
        "lxd.ingest.captions.caption_png",
        lambda _path, _config: "A screenshot of the Adobe Learning Manager dashboard.",
    )
    monkeypatch.setattr(
        "lxd.ingest.captions.embed_chunk_text", lambda _config, _text: [1.0, 2.0]
    )

    asset_path = tmp_path / "dashboard.png"
    result = caption_asset_source(
        absolute_path=asset_path,
        source_rel_path="screenshots/dashboard.png",
        source_type="image_png",
        source_domain="screenshots",
        content_hash="hash-xyz",
        document_id="doc-42",
        config=_config(),
    )

    assert result is not None
    assert result.text == "A screenshot of the Adobe Learning Manager dashboard."
    assert result.document_id == "doc-42"
    assert result.vector == [1.0, 2.0]
    assert "dashboard.png" in result.citation_label
