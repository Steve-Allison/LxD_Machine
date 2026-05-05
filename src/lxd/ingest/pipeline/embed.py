"""Embedding helpers: cache lookup, contextual augmentation, context refinement, reindex."""

from __future__ import annotations

from collections.abc import Callable

from lxd.domain.ids import make_chunk_id
from lxd.ingest.budget import IngestBudgetTracker
from lxd.ingest.chunking import (
    TextChunk,
    build_tokenizer,
    split_chunk_for_context,
    token_count_with_tokenizer,
)
from lxd.ingest.contextual_chunker import (
    augment_chunk_for_embedding,
    generate_chunk_context,
    lookup_summaries,
    store_summaries,
)
from lxd.ingest.embedder import (
    EmbeddingContextError,
    embed_chunk_text,
    embed_texts,
    embed_texts_batched,
)
from lxd.ingest.embedding_cache import lookup as cache_lookup
from lxd.ingest.embedding_cache import store as cache_store
from lxd.ingest.markdown import ExtractedDocument
from lxd.settings.models import RuntimeConfig


def scanned_filename_for_title(extracted_document: ExtractedDocument) -> str:
    """Fallback title when the document has no explicit one."""
    source_path = getattr(extracted_document, "source_rel_path", None)
    if isinstance(source_path, str) and source_path:
        return source_path.rsplit("/", 1)[-1]
    return "(untitled document)"


def embed_with_contextual_augmentation(
    chunks: list[TextChunk],
    *,
    document_id: str,
    config: RuntimeConfig,
    extracted_document: ExtractedDocument,
    summary_cache_table: object,
    budget_tracker: IngestBudgetTracker,
) -> tuple[list[TextChunk], list[list[float]], int, int]:
    """Embed chunks with contextual summary augmentation.

    For each chunk: look up a one-sentence "what this chunk is about"
    summary in the contextual cache; on miss, generate via local Ollama
    (counted against the ingest budget). Build the augmented text
    ``f"{summary}\\n\\n{chunk.text}"`` and embed *that* — the stored
    chunk text remains the original. Returns the cache-hit/miss counts
    for the **summary** cache (not the embedding cache, which is
    bypassed when contextual is enabled).
    """
    if not chunks:
        return [], [], 0, 0

    chunk_hashes = [chunk.chunk_hash for chunk in chunks]
    summary_model = config.chunking.contextual_summary_model
    summary_lookup = lookup_summaries(
        summary_cache_table, chunk_hashes=chunk_hashes, model=summary_model
    )

    document_title = scanned_filename_for_title(extracted_document)
    document_summary = extracted_document.wiki_metadata.summary or ""

    summaries: list[str] = []
    fresh_pairs: list[tuple[str, str]] = []
    for index, chunk in enumerate(chunks):
        cached = summary_lookup.hits.get(index)
        if cached is not None:
            summaries.append(cached)
            continue
        budget_tracker.check()
        summary = generate_chunk_context(
            chunk_text=chunk.text,
            document_title=document_title,
            document_summary=document_summary,
            config=config,
        )
        budget_tracker.record_llm_call()
        summaries.append(summary)
        if summary:
            fresh_pairs.append((chunk.chunk_hash, summary))

    if fresh_pairs:
        store_summaries(
            summary_cache_table,
            chunk_hashes=[h for h, _ in fresh_pairs],
            summaries=[s for _, s in fresh_pairs],
            model=summary_model,
        )

    augmented_texts = [
        augment_chunk_for_embedding(chunk.text, summary)
        for chunk, summary in zip(chunks, summaries, strict=True)
    ]
    vectors = embed_texts(config, augmented_texts)
    reindexed = _reindex_chunks(list(chunks), document_id)
    return reindexed, vectors, summary_lookup.hit_count, summary_lookup.miss_count


def embed_with_cache(
    chunks: list[TextChunk],
    *,
    document_id: str,
    config: RuntimeConfig,
    cache_table: object | None,
) -> tuple[list[TextChunk], list[list[float]], int, int]:
    """Embed ``chunks`` consulting the content-addressed cache first.

    Cache hits avoid all network/API spend. Cache misses go through the
    existing context-refinement embed path (which may split chunks that
    overflow the model's context window). On miss-success, results are
    stored back in the cache for the next run.

    Returns:
        ``(reindexed_chunks, vectors, cache_hits, cache_misses)`` — vectors
        align with ``reindexed_chunks`` 1:1, regardless of cache status.

    Important: the cache key is ``(chunk_hash, embedding_model,
    embedding_dims)``. ``chunk_hash`` is content-addressed, so cache entries
    are intrinsically safe to keep across full rebuilds and need no explicit
    invalidation. Changing the embedding model produces a new key and old
    entries naturally fall out of use.
    """
    if not chunks:
        return [], [], 0, 0

    if cache_table is None:
        # Fallback: no cache configured. Behave as before.
        text_chunks, vectors = _embed_with_context_refinement(chunks, document_id, config)
        return text_chunks, vectors, 0, len(text_chunks)

    chunk_hashes = [chunk.chunk_hash for chunk in chunks]
    lookup_result = cache_lookup(
        cache_table,
        chunk_hashes=chunk_hashes,
        embedding_model=config.models.embed,
        embedding_dims=config.models.embed_dims,
    )

    if not lookup_result.misses_indices:
        # Full hit. No API call.
        reindexed = _reindex_chunks(list(chunks), document_id)
        vectors = [lookup_result.hits[i] for i in range(len(chunks))]
        return reindexed, vectors, lookup_result.hit_count, 0

    # Partial or full miss. Embed only the misses, then merge.
    miss_chunks = [chunks[i] for i in lookup_result.misses_indices]
    if lookup_result.hits:
        # Some chunks hit the cache. Run context-refinement only on the
        # missed chunks, then reassemble in original order. This means
        # context-refinement may split a missed chunk into N — we accept
        # the resulting size mismatch and fall back to re-embedding the
        # whole batch rather than try to splice cached and freshly-split
        # chunks together (correctness > marginal cache savings on edge).
        try:
            miss_text_chunks, miss_vectors = _embed_with_context_refinement(
                miss_chunks, document_id, config
            )
        except EmbeddingContextError:
            # Should not happen here — _embed_with_context_refinement
            # handles this internally — but if it bubbles up, propagate.
            raise

        if len(miss_text_chunks) == len(miss_chunks):
            merged_chunks: list[TextChunk] = []
            merged_vectors: list[list[float]] = []
            miss_iter = iter(zip(miss_text_chunks, miss_vectors, strict=True))
            for idx in range(len(chunks)):
                if idx in lookup_result.hits:
                    merged_chunks.append(chunks[idx])
                    merged_vectors.append(lookup_result.hits[idx])
                else:
                    chunk_, vec_ = next(miss_iter)
                    merged_chunks.append(chunk_)
                    merged_vectors.append(vec_)
            reindexed = _reindex_chunks(merged_chunks, document_id)
            cache_store(
                cache_table,
                chunk_hashes=[chunks[i].chunk_hash for i in lookup_result.misses_indices],
                vectors=miss_vectors,
                embedding_model=config.models.embed,
                embedding_dims=config.models.embed_dims,
            )
            return (
                reindexed,
                merged_vectors,
                lookup_result.hit_count,
                lookup_result.miss_count,
            )

        # Miss chunks were split during context refinement. Fall through
        # to "embed all" path so chunk count stays consistent.

    text_chunks, vectors = _embed_with_context_refinement(chunks, document_id, config)
    cache_store(
        cache_table,
        chunk_hashes=[c.chunk_hash for c in text_chunks],
        vectors=vectors,
        embedding_model=config.models.embed,
        embedding_dims=config.models.embed_dims,
    )
    return text_chunks, vectors, 0, len(text_chunks)


def _embed_with_context_refinement(
    chunks: list[TextChunk],
    document_id: str,
    config: RuntimeConfig,
) -> tuple[list[TextChunk], list[list[float]]]:
    """Embed ``chunks`` in a single batch, splitting any that overflow context.

    Attempts a batch call first so the embedding backend can amortise HTTP
    and model-load overhead. Chunks that trigger
    :class:`EmbeddingContextError` are recursively split via the existing
    token-aware chunker and re-embedded; the returned ``chunks`` list may
    therefore be longer than the input.
    """
    token_counter = token_count_with_tokenizer(
        build_tokenizer(config.chunking.tokenizer_backend, config.chunking.tokenizer_name)
    )
    if not chunks:
        return [], []

    try:
        vectors = embed_texts_batched(config, [chunk.text for chunk in chunks])
        reindexed = _reindex_chunks(list(chunks), document_id)
        return reindexed, vectors
    except EmbeddingContextError:
        pass

    resolved_chunks: list[TextChunk] = []
    vectors = []
    for chunk in chunks:
        refined_chunks, refined_vectors = _embed_chunk_recursively(
            chunk,
            config,
            token_counter=token_counter,
        )
        resolved_chunks.extend(refined_chunks)
        vectors.extend(refined_vectors)

    reindexed_chunks = _reindex_chunks(resolved_chunks, document_id)
    return reindexed_chunks, vectors


def _embed_chunk_recursively(
    chunk: TextChunk,
    config: RuntimeConfig,
    *,
    token_counter: Callable[[str], int],
) -> tuple[list[TextChunk], list[list[float]]]:
    try:
        return [chunk], [embed_chunk_text(config, chunk.text)]
    except EmbeddingContextError:
        split_chunks = split_chunk_for_context(chunk, token_counter=token_counter)
        if len(split_chunks) == 1 and split_chunks[0].text == chunk.text:
            raise
        resolved_chunks: list[TextChunk] = []
        vectors: list[list[float]] = []
        for split_chunk in split_chunks:
            nested_chunks, nested_vectors = _embed_chunk_recursively(
                split_chunk,
                config,
                token_counter=token_counter,
            )
            resolved_chunks.extend(nested_chunks)
            vectors.extend(nested_vectors)
        return resolved_chunks, vectors


def _reindex_chunks(chunks: list[TextChunk], document_id: str) -> list[TextChunk]:
    if not chunks:
        return []
    occurrences: dict[str, int] = {}
    reindexed: list[TextChunk] = []
    for index, chunk in enumerate(chunks):
        chunk_occurrence = occurrences.get(chunk.chunk_hash, 0)
        occurrences[chunk.chunk_hash] = chunk_occurrence + 1
        reindexed.append(
            TextChunk(
                chunk_id=make_chunk_id(document_id, chunk.chunk_hash, chunk_occurrence),
                document_id=document_id,
                source_rel_path=chunk.source_rel_path,
                source_type=chunk.source_type,
                citation_label=chunk.citation_label,
                chunk_index=index,
                chunk_occurrence=chunk_occurrence,
                token_count=chunk.token_count,
                text=chunk.text,
                chunk_hash=chunk.chunk_hash,
                score_hint=chunk.score_hint,
                metadata_json=chunk.metadata_json,
            )
        )
    return reindexed
