"""Build matcher terms and automata from ontology entries."""

import json
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ahocorasick
import structlog

from lxd.domain.ids import blake3_hex
from lxd.ontology.normalization import normalize_match_text

_log = structlog.get_logger(__name__)


@dataclass(frozen=True, order=True, slots=True)
class MatcherTermRecord:
    """Normalized matcher term mapped to an entity ID."""

    normalized_term: str
    entity_id: str
    term_source: str


def canonical_matcher_term_records(
    entity_definitions: Iterable[dict[str, Any]],
) -> list[MatcherTermRecord]:
    """Build canonical matcher terms from ontology entities.

    Args:
        entity_definitions: Ontology entity definitions to index.

    Returns:
        Sorted unique matcher term records.
    """
    records: set[MatcherTermRecord] = set()
    for entity in entity_definitions:
        canonical_id = _coerce_required_str(entity, "canonical_id")
        for term_source, values in (
            ("canonical_id", [canonical_id]),
            ("alias", _coerce_str_list(entity.get("aliases", []))),
            ("indicator", _coerce_str_list(entity.get("indicators", []))),
        ):
            for value in values:
                normalized = normalize_match_text(value)
                if normalized:
                    records.add(
                        MatcherTermRecord(
                            normalized_term=normalized,
                            entity_id=canonical_id,
                            term_source=term_source,
                        )
                    )
    return sorted(records)


def matcher_termset_hash(records: Iterable[MatcherTermRecord]) -> str:
    """Compute a stable hash for matcher term records.

    Args:
        records: Record collection to hash or index.

    Returns:
        Stable hash for the matcher term set.
    """
    lines = [
        json.dumps(
            {
                "entity_id": record.entity_id,
                "term_source": record.term_source,
                "normalized_term": record.normalized_term,
            },
            sort_keys=False,
            separators=(",", ":"),
        )
        for record in records
    ]
    return blake3_hex("\n".join(lines))


def build_automaton(
    records: Iterable[MatcherTermRecord],
) -> ahocorasick.Automaton:  # type: ignore[type-arg]
    """Build an Aho-Corasick automaton for mention matching.

    Args:
        records: Record collection to hash or index.

    Returns:
        Configured automaton ready for mention detection.
    """
    automaton = ahocorasick.Automaton(ahocorasick.STORE_ANY, ahocorasick.KEY_STRING)
    for record in records:
        payload = {
            "entity_id": record.entity_id,
            "term_source": record.term_source,
            "normalized_term": record.normalized_term,
        }
        automaton.add_word(record.normalized_term, payload)
    automaton.make_automaton()
    return automaton


def build_or_load_automaton(
    records: Iterable[MatcherTermRecord],
    *,
    cache_dir: Path,
) -> ahocorasick.Automaton:  # type: ignore[type-arg]
    """Return an Aho-Corasick automaton, hydrated from disk when possible.

    The disk cache is keyed on :func:`matcher_termset_hash` over the input
    records. A hit avoids the ~1-2 s of automaton construction that
    short-running CLI commands (status, MCP tool startup) otherwise pay on
    every invocation; a miss builds via :func:`build_automaton`, persists
    the pickled automaton under ``cache_dir``, and returns it.

    Hash mismatch is intrinsic invalidation: a different ontology produces a
    different hash, which produces a different filename, so stale entries
    fall out of use without explicit purging. Corrupt or unreadable cache
    files fall through to a fresh build and overwrite themselves.

    Args:
        records: Matcher term records to index. The records are iterated
            twice (once for hashing, once for the build path on cache miss),
            so a non-list iterable is materialised into a list internally.
        cache_dir: Directory the cache file lives in. Created if missing.

    Returns:
        Configured automaton ready for mention detection.

    Side Effects:
        Reads / writes ``cache_dir/matcher-<hash>.pkl``.
    """
    record_list = list(records)
    cache_key = matcher_termset_hash(record_list)
    cache_path = cache_dir / f"matcher-{cache_key}.pkl"
    if cache_path.is_file():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            _log.warning(
                "matcher_cache_load_failed",
                cache_path=str(cache_path),
                error=str(exc),
            )
        else:
            if isinstance(cached, ahocorasick.Automaton):
                _log.debug("matcher_cache_hit", cache_path=str(cache_path))
                return cached
            _log.warning(
                "matcher_cache_unexpected_type",
                cache_path=str(cache_path),
                got_type=type(cached).__name__,
            )

    automaton = build_automaton(record_list)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".pkl.tmp")
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(automaton, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
    except OSError as exc:
        _log.warning(
            "matcher_cache_write_failed",
            cache_path=str(cache_path),
            error=str(exc),
        )
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return automaton


def _coerce_required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required string field: {key}")
    return value


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected list of strings")
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item)
    return result
