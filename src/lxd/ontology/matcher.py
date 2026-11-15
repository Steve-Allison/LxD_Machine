"""Build proper-noun Aho-Corasick automata strictly over ``match_lexicon``."""

import json
import pickle
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import ahocorasick
import structlog

from lxd.domain.ids import blake3_hex

_log = structlog.get_logger(__name__)


@dataclass(frozen=True, order=True, slots=True)
class LexiconRule:
    """Proper-noun surface rule projected from ``Central_Configs`` ``match_lexicon``.

    Only ``LinkableProperNoun`` entities appear in ``match_lexicon``. Craft
    concepts, cognitive principles, rhetorical moves, and rubrics never enter
    this lexicon.
    """

    surface_lower: str
    curie: str
    surface: str
    pref_label: str = ""
    collection_name: str = ""
    match_policy: str = "unambiguous"
    case_sensitive: bool = False
    anchor_window_tokens: int = 25
    context_anchors: tuple[str, ...] = ()
    negative_phrases: tuple[str, ...] = ()
    requires_llm_disambiguation: bool = False


def offset_preserving_lower(text: str) -> str:
    """Lowercase ``text`` while guaranteeing 1:1 character offsets with raw ``text``.

    Standard ASCII and BMP Latin characters preserve length under ``str.lower()``.
    Any rare character whose ``str.lower()`` expands to multiple codepoints (e.g.
    ``'İ'`` -> ``'i\\u0307'``) is mapped to its base character so ``len(result) ==
    len(text)`` always holds.
    """
    if text.isascii():
        return text.lower()
    lowered = text.lower()
    if len(lowered) == len(text):
        return lowered
    return "".join(ch.lower() if len(ch.lower()) == 1 else ch.lower()[0] for ch in text)


def matcher_termset_hash(records: Iterable[LexiconRule]) -> str:
    """Compute a deterministic BLAKE3 digest over sorted ``LexiconRule`` entries."""
    ordered = sorted(set(records))
    lines = [
        json.dumps(
            {
                "curie": rule.curie,
                "surface": rule.surface,
                "surface_lower": rule.surface_lower,
                "match_policy": rule.match_policy,
                "case_sensitive": rule.case_sensitive,
                "anchor_window_tokens": rule.anchor_window_tokens,
                "context_anchors": list(rule.context_anchors),
                "negative_phrases": list(rule.negative_phrases),
                "requires_llm_disambiguation": rule.requires_llm_disambiguation,
            },
            sort_keys=False,
            separators=(",", ":"),
        )
        for rule in ordered
    ]
    return blake3_hex("\n".join(lines))


def build_automaton(records: Iterable[LexiconRule]) -> ahocorasick.Automaton:
    """Build a ``pyahocorasick.Automaton`` strictly over ``match_lexicon`` rules.

    Keys are ``offset_preserving_lower(rule.surface)`` and values are tuples of
    all ``LexiconRule`` records sharing that lowercase surface form, allowing
    Stage 1 to inspect original casing, negative phrases, and context anchors
    without mutating source text.
    """
    grouped: dict[str, list[LexiconRule]] = defaultdict(list)
    for rule in sorted(set(records)):
        key = offset_preserving_lower(rule.surface)
        if not key:
            continue
        grouped[key].append(rule)

    automaton = ahocorasick.Automaton(ahocorasick.STORE_ANY, ahocorasick.KEY_STRING)
    for key, rules in sorted(grouped.items()):
        automaton.add_word(key, tuple(rules))
    automaton.make_automaton()
    return automaton


def build_or_load_automaton(
    records: Iterable[LexiconRule],
    *,
    cache_dir: Path,
) -> ahocorasick.Automaton:
    """Return an Aho-Corasick automaton, hydrated from disk cache when valid."""
    record_list = list(records)
    cache_key = matcher_termset_hash(record_list)
    cache_path = cache_dir / f"matcher-{cache_key}.pkl"
    if cache_path.is_file():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
        except (pickle.UnpicklingError, AttributeError, EOFError, OSError) as exc:
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
