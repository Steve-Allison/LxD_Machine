"""Per-source pipeline body: chunk → embed → mention/relation detect → assemble records."""

from collections.abc import Callable

from lxd.domain.status import LifecycleStatus, RetrievalStatus
from lxd.ingest.budget import IngestBudgetTracker
from lxd.ingest.chunking import chunk_document
from lxd.ingest.docling import load_docling_document
from lxd.ingest.markdown import ExtractedDocument, load_markdown_document
from lxd.ingest.mentions import detect_mentions
from lxd.ingest.pipeline.embed import (
    embed_with_cache,
    embed_with_contextual_augmentation,
)
from lxd.ingest.relations import extract_relations_for_chunk
from lxd.ingest.scanner import ScannedCorpusFile
from lxd.ingest.wiki_relations import derive_wiki_link_relations
from lxd.settings.models import RuntimeConfig
from lxd.stores.models import (
    ChunkRecord,
    ExtractedRelationRecord,
    ManifestRecord,
    MentionRecord,
)


def build_source_records(
    *,
    scanned: ScannedCorpusFile,
    document_id: str,
    config: RuntimeConfig,
    automaton: object,
    valid_predicates: frozenset[str],
    slug_index: dict[str, str],
    budget_tracker: IngestBudgetTracker,
    cache_table: object | None = None,
    contextual_summary_table: object | None = None,
    ambiguous_map: dict[str, list[str]] | None = None,
    disambiguator: Callable[[str, list[str]], str | None] | None = None,
) -> tuple[
    list[ChunkRecord],
    list[MentionRecord],
    list[ExtractedRelationRecord],
    int,
    int,
    tuple[str, ...],
    tuple[str, ...],
]:
    """Chunk, embed (with cache), and detect mentions/relations for one source.

    Relations are derived from two sources, joined into a single list:

    * **LLM extraction** (``relations.py``) over chunk text, gated by
      ``valid_predicates`` from the ontology relation schema.
    * **Wiki frontmatter** (``wiki_relations.py``) — for chunks whose host
      page has ``[[slug]]`` cross-references that resolve via
      ``slug_index`` to ontology canonical_ids, emit deterministic
      ``wiki_references`` edges. No LLM cost, no hallucination risk.

    Returns:
        ``(chunk_records, mention_records, relation_records, cache_hits,
        cache_misses, wiki_dangling_slugs, wiki_pages_without_subject)`` —
        the last two are diagnostics that the caller aggregates run-wide.
    """
    extracted_document = _load_extracted_document(scanned)
    initial_chunks = chunk_document(
        extracted_document,
        document_id=document_id,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        min_tokens=config.chunking.min_tokens,
        tokenizer_backend=config.chunking.tokenizer_backend,
        tokenizer_name=config.chunking.tokenizer_name,
        strategy=config.chunking.strategy,
    )
    if config.chunking.contextual_summary_enabled and contextual_summary_table is not None:
        text_chunks, embeddings, cache_hits, cache_misses = embed_with_contextual_augmentation(
            initial_chunks,
            document_id=document_id,
            config=config,
            extracted_document=extracted_document,
            summary_cache_table=contextual_summary_table,
            budget_tracker=budget_tracker,
        )
    else:
        text_chunks, embeddings, cache_hits, cache_misses = embed_with_cache(
            initial_chunks,
            document_id=document_id,
            config=config,
            cache_table=cache_table,
        )
    chunk_records: list[ChunkRecord] = []
    mention_records: list[MentionRecord] = []
    relation_records: list[ExtractedRelationRecord] = []
    page_cited_sources = extracted_document.wiki_metadata.cited_sources
    page_wiki_links = extracted_document.wiki_metadata.wiki_links
    for chunk, vector in zip(text_chunks, embeddings, strict=True):
        chunk_record = ChunkRecord(
            chunk_id=chunk.chunk_id,
            document_id=document_id,
            source_rel_path=chunk.source_rel_path,
            source_filename=scanned.absolute_path.name,
            source_type=chunk.source_type,
            source_domain=scanned.source_domain,
            source_hash=scanned.content_hash,
            citation_label=chunk.citation_label,
            chunk_index=chunk.chunk_index,
            chunk_occurrence=chunk.chunk_occurrence,
            token_count=chunk.token_count,
            text=chunk.text,
            chunk_hash=chunk.chunk_hash,
            score_hint=chunk.score_hint,
            metadata_json=chunk.metadata_json,
            vector=vector,
            embedding_model=config.models.embed,
            embedding_dims=config.models.embed_dims,
            cited_sources=page_cited_sources,
            wiki_links=page_wiki_links,
        )
        chunk_records.append(chunk_record)
        chunk_mentions = list(
            MentionRecord(
                chunk_id=chunk_record.chunk_id,
                entity_id=mention.entity_id,
                term_source=mention.term_source,
                surface_form=mention.surface_form,
                start_char=mention.start_char,
                end_char=mention.end_char,
            )
            for mention in detect_mentions(
                chunk.text,
                automaton,
                ambiguous_map=ambiguous_map,
                disambiguator=disambiguator,
            )
        )
        mention_records.extend(chunk_mentions)
        will_call_llm = len(
            {m.entity_id for m in chunk_mentions}
        ) >= config.relation_extraction.min_entity_mentions and bool(valid_predicates)
        if will_call_llm:
            budget_tracker.check()
        relation_records.extend(
            extract_relations_for_chunk(
                chunk_id=chunk_record.chunk_id,
                document_id=document_id,
                source_rel_path=chunk_record.source_rel_path,
                chunk_text=chunk.text,
                mention_records=chunk_mentions,
                valid_predicates=valid_predicates,
                config=config,
            )
        )
        if will_call_llm:
            budget_tracker.record_llm_call()
    from lxd.ingest.pipeline.orchestrator import utc_now  # local import to avoid cycle

    wiki_outcome = derive_wiki_link_relations(
        chunk_records=chunk_records,
        slug_index=slug_index,
        extracted_at=utc_now(),
    )
    relation_records.extend(wiki_outcome.relations)
    return (
        chunk_records,
        mention_records,
        relation_records,
        cache_hits,
        cache_misses,
        wiki_outcome.dangling_slugs,
        wiki_outcome.pages_without_subject,
    )


def _load_extracted_document(scanned: ScannedCorpusFile) -> ExtractedDocument:
    if scanned.source_type in ("markdown", "docling_md"):
        return load_markdown_document(
            scanned.absolute_path,
            scanned.relative_path,
            source_type=scanned.source_type,
        )
    return load_docling_document(scanned.absolute_path, scanned.relative_path)


def build_manifest_record(
    *,
    scanned: ScannedCorpusFile,
    document_id: str | None,
    parent_source_rel_path: str | None,
    chunk_count: int,
    timestamp: str,
    lifecycle_status: LifecycleStatus,
    retrieval_status: RetrievalStatus,
    error_message: str | None,
) -> ManifestRecord:
    return ManifestRecord(
        source_rel_path=scanned.relative_path,
        absolute_path=scanned.absolute_path.as_posix(),
        source_type=scanned.source_type,
        source_domain=scanned.source_domain,
        document_id=document_id,
        file_size_bytes=scanned.file_size_bytes,
        content_hash=scanned.content_hash,
        parent_source_rel_path=parent_source_rel_path,
        chunk_count=chunk_count,
        last_seen_at=timestamp,
        last_processed_at=timestamp,
        last_committed_at=timestamp if lifecycle_status == LifecycleStatus.COMPLETE else None,
        error_message=error_message,
        lifecycle_status=lifecycle_status,
        retrieval_status=retrieval_status,
    )
