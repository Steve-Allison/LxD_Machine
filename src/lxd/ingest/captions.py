"""Generate short captions for PNG assets so they become queryable via text search.

Responsibility:
    Turn a PNG asset into a searchable chunk without embedding the image
    itself. :func:`caption_png` asks an OpenAI vision model for a short,
    query-dense description; :func:`build_caption_chunk_record` embeds that
    caption text like any other chunk and assembles a persistable
    :class:`ChunkRecord`. :func:`caption_asset_source` composes the two and
    is the single entrypoint the ingest orchestrator and the
    ``caption-assets`` backfill CLI both call.

Design boundary:
    Captioning is best-effort. :func:`caption_png` never raises — a missing
    file, missing API key, or any OpenAI/HTTP failure returns ``""`` so
    callers can fall back to the existing ``asset_only`` retrieval status
    without special-casing exceptions.
"""

import base64
import json
from pathlib import Path
from typing import Final

import httpx
import openai
import structlog

from lxd.domain.ids import blake3_hex, make_chunk_id
from lxd.ingest.chunking import build_tokenizer
from lxd.ingest.embedder import embed_chunk_text
from lxd.ingest.llm_client import get_sync_openai_client
from lxd.settings.models import RuntimeConfig
from lxd.stores.models import ChunkRecord

_log = structlog.get_logger(__name__)

_CAPTION_SYSTEM_PROMPT: Final = (
    "You caption images embedded in instructional-design and "
    "learning-experience-design reference material (diagrams, screenshots, "
    "model illustrations, charts). Describe what the image shows and any "
    "visible text/labels in 2-4 dense sentences, using the concrete nouns a "
    "search query would use. No preamble, no markdown, no hedging."
)

_CAPTION_ERRORS: Final[tuple[type[BaseException], ...]] = (
    openai.OpenAIError,
    httpx.HTTPError,
    OSError,
    ValueError,
    RuntimeError,
)


def caption_png(path: Path, config: RuntimeConfig) -> str:
    """Generate a short caption for one PNG asset via OpenAI vision.

    Args:
        path: Absolute path to the PNG file.
        config: Runtime configuration (``config.multimodal`` and
            ``config.openai`` drive model, timeout, and credentials).

    Returns:
        The stripped caption text, or ``""`` on any failure (missing file,
        empty file, missing API key, HTTP/SDK error, timeout).
    """
    multimodal = config.multimodal
    try:
        image_bytes = path.read_bytes()
    except OSError as exc:
        _log.warning("caption_read_failed", path=str(path), error=str(exc))
        return ""
    if not image_bytes:
        return ""

    api_key_env = config.openai.api_key_env if config.openai else "OPENAI_API_KEY"
    try:
        client = get_sync_openai_client(api_key_env)
    except RuntimeError as exc:
        _log.warning("caption_client_unavailable", path=str(path), error=str(exc))
        return ""

    encoded = base64.b64encode(image_bytes).decode("ascii")
    try:
        response = client.chat.completions.create(
            model=multimodal.caption_model,
            messages=[
                {"role": "system", "content": _CAPTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Caption this image (filename: {path.name}).",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                },
            ],
            max_tokens=multimodal.caption_max_tokens,
            timeout=float(multimodal.caption_timeout_secs),
        )
    except _CAPTION_ERRORS as exc:
        _log.warning("caption_generation_failed", path=str(path), error=str(exc))
        return ""

    content = response.choices[0].message.content
    return content.strip() if content else ""


def build_caption_chunk_record(
    *,
    source_rel_path: str,
    source_filename: str,
    source_domain: str,
    source_type: str,
    content_hash: str,
    document_id: str,
    caption_text: str,
    config: RuntimeConfig,
) -> ChunkRecord:
    """Assemble a persistable, embedded chunk from a generated image caption.

    Mirrors the text-chunk shape (``ChunkRecord``) so caption chunks flow
    through the same persistence, retrieval, and citation paths as any
    other chunk. The chunk id is deterministic
    (``make_chunk_id(document_id, chunk_hash, 0)``), matching the
    convention used for text chunks, so re-running against an unchanged
    caption is a no-op at the persistence layer.

    The citation label is deliberately explicit about the asset origin
    (``"<path> (image caption: <filename>)"``) so synthesis output never
    presents a caption chunk as if it were prose from the source document.

    Args:
        source_rel_path: Corpus-relative path of the PNG asset.
        source_filename: Filename of the PNG asset.
        source_domain: Source domain derived from the corpus path.
        source_type: Always ``"image_png"`` for this caller; kept explicit
            rather than hard-coded so tests can exercise other shapes.
        content_hash: BLAKE3 content hash of the PNG file (from the scan).
        document_id: Stable document identifier for this asset.
        caption_text: Non-empty caption text from :func:`caption_png`.
        config: Runtime configuration (embedding model/dims, tokenizer).

    Returns:
        A fully embedded, ready-to-persist :class:`ChunkRecord`.
    """
    tokenizer = build_tokenizer(config.chunking.tokenizer_backend, config.chunking.tokenizer_name)
    chunk_hash = blake3_hex(caption_text)
    citation_label = f"{source_rel_path} (image caption: {source_filename})"
    metadata_json = json.dumps(
        {"kind": "image_caption", "image_filename": source_filename},
        sort_keys=True,
    )
    vector = embed_chunk_text(config, caption_text)
    return ChunkRecord(
        chunk_id=make_chunk_id(document_id, chunk_hash, 0),
        document_id=document_id,
        source_rel_path=source_rel_path,
        source_filename=source_filename,
        source_type=source_type,
        source_domain=source_domain,
        source_hash=content_hash,
        citation_label=citation_label,
        chunk_index=0,
        chunk_occurrence=0,
        token_count=len(tokenizer.encode(caption_text)),
        text=caption_text,
        chunk_hash=chunk_hash,
        score_hint=caption_text[:160],
        metadata_json=metadata_json,
        vector=vector,
        embedding_model=config.models.embed,
        embedding_dims=config.models.embed_dims,
    )


def caption_asset_source(
    *,
    absolute_path: Path,
    source_rel_path: str,
    source_type: str,
    source_domain: str,
    content_hash: str,
    document_id: str,
    config: RuntimeConfig,
) -> ChunkRecord | None:
    """Caption one PNG asset and assemble a searchable chunk, or ``None``.

    Single entrypoint shared by the ingest orchestrator (inline captioning
    of new/changed PNGs) and the ``caption-assets`` backfill CLI. Returns
    ``None`` when captioning failed or produced empty text — callers should
    keep the existing ``asset_only`` retrieval status in that case.
    """
    caption_text = caption_png(absolute_path, config)
    if not caption_text:
        return None
    return build_caption_chunk_record(
        source_rel_path=source_rel_path,
        source_filename=absolute_path.name,
        source_domain=source_domain,
        source_type=source_type,
        content_hash=content_hash,
        document_id=document_id,
        caption_text=caption_text,
        config=config,
    )
