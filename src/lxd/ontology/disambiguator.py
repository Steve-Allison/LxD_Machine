"""Constrained local MLX + ``llguidance`` Stage 2 proper-noun disambiguator.

Stage 1 (``pyahocorasick`` over ``match_lexicon``) resolves unambiguous,
case-exact, and context-anchored proper nouns deterministically on unmodified
text. Spans that carry ``requires_llm_disambiguation=True`` or map to multiple
candidate ``coe:Cxxxxx`` CURIEs escalate to Stage 2.

Stage 2 runs a local ``mlx-lm`` model (default
``mlx-community/gemma-4-26B-A4B-it-4bit``) constrained at token level by an
``llguidance`` regex grammar ``^(:?coe:C...|NIL)$`` at ``temp=0.0``. Offline
unit tests inject any ``MentionDisambiguator`` protocol implementation so no
paid external API calls or multi-GB model loads are required in fast CI.
"""

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import llguidance
import llguidance.hf
import llguidance.mlx
import llguidance.numpy
import structlog

from lxd.settings.models import RuntimeConfig

_log = structlog.get_logger(__name__)

DEFAULT_DISAMBIGUATION_MODEL = "mlx-community/gemma-4-26B-A4B-it-4bit"


@dataclass(frozen=True, slots=True)
class DisambiguationCandidate:
    """Candidate proper-noun entity presented to Stage 2 disambiguation."""

    curie: str
    pref_label: str = ""
    collection_name: str = ""
    definition: str = ""


@runtime_checkable
class MentionDisambiguator(Protocol):
    """Injectable protocol for Stage 2 proper-noun disambiguation."""

    def disambiguate(
        self,
        *,
        text: str,
        start: int,
        end: int,
        surface: str,
        candidates: Sequence[DisambiguationCandidate],
    ) -> str | None:
        """Return a candidate CURIE, or ``"NIL"`` / ``None`` to abstain."""
        ...


def build_disambiguation_regex(candidate_curies: Sequence[str]) -> str:
    """Build the exact anchored regex ``^(c1|c2|...|NIL)$`` for ``llguidance``."""
    unique = list(dict.fromkeys(c.strip() for c in candidate_curies if c.strip() and c != "NIL"))
    if not unique:
        raise ValueError("candidate_curies must contain at least one CURIE.")
    return "^(" + "|".join(re.escape(curie) for curie in [*unique, "NIL"]) + ")$"


def build_disambiguation_grammar(candidate_curies: Sequence[str]) -> str:
    """Compile the ``llguidance`` regex grammar JSON specification."""
    pattern = build_disambiguation_regex(candidate_curies)
    return llguidance.LLMatcher.grammar_from_regex(pattern)


def compile_llguidance_matcher(
    ll_tokenizer: llguidance.LLTokenizer,
    candidate_curies: Sequence[str],
) -> llguidance.LLMatcher:
    """Create and validate an ``LLMatcher`` for ``candidate_curies`` + ``NIL``."""
    grammar = build_disambiguation_grammar(candidate_curies)
    matcher = llguidance.LLMatcher(ll_tokenizer, grammar)
    if matcher.is_error():
        raise ValueError(f"Invalid llguidance disambiguation grammar: {matcher.get_error()}")
    return matcher


def context_window(text: str, *, start: int, end: int, radius: int = 240) -> str:
    """Format a bounded character window around ``[start, end)`` with ``<<...>>`` markers."""
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return f"{text[left:start]}<<{text[start:end]}>>{text[end:right]}"


def build_disambiguation_prompt(
    *,
    text: str,
    start: int,
    end: int,
    surface: str,
    candidates: Sequence[DisambiguationCandidate],
    radius: int = 240,
) -> str:
    """Build the deterministic Stage 2 disambiguation prompt."""
    marked = context_window(text, start=start, end=end, radius=radius)
    option_lines: list[str] = []
    for candidate in candidates:
        label = candidate.pref_label or candidate.curie
        kind = f" ({candidate.collection_name})" if candidate.collection_name else ""
        desc = f" — {candidate.definition}" if candidate.definition else ""
        option_lines.append(f"- {candidate.curie}: {label}{kind}{desc}")
    option_lines.append(
        "- NIL: Generic English word, non-entity usage, or none of the candidates above."
    )
    options_block = "\n".join(option_lines)
    return (
        "Identify whether the highlighted mention <<...>> refers to one of the "
        "listed proper-noun entities or NIL.\n"
        "Respond with ONLY the exact identifier (e.g. coe:C01008 or NIL).\n\n"
        f"Mention: {surface}\n"
        f"Context: {marked}\n\n"
        f"Candidates:\n{options_block}\n\n"
        "Identifier:"
    )


def make_mlx_llguidance_processor(
    matcher: llguidance.LLMatcher,
    eos_token_id: int,
) -> Callable[[Any, Any], Any]:
    """Create an ``mlx_lm`` logits processor backed by ``llguidance.mlx``."""
    import mlx.core as mx

    prompt_len: int | None = None
    consumed = 0
    bitmask_cache: Any = None
    cached_vocab_size = -1

    def processor(tokens: Any, logits: Any) -> Any:
        nonlocal prompt_len, consumed, bitmask_cache, cached_vocab_size
        total_tokens = len(tokens)
        if prompt_len is None:
            prompt_len = total_tokens
        while prompt_len + consumed < total_tokens:
            tok_id = int(tokens[prompt_len + consumed].item())
            matcher.consume_token(tok_id)
            consumed += 1

        if matcher.is_stopped():
            vocab_size = int(logits.shape[-1])
            eos_mask = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
            if 0 <= eos_token_id < vocab_size:
                eos_col = mx.zeros((logits.shape[0], 1), dtype=logits.dtype)
                parts: list[Any] = []
                if eos_token_id > 0:
                    parts.append(eos_mask[:, :eos_token_id])
                parts.append(eos_col)
                if eos_token_id + 1 < vocab_size:
                    parts.append(eos_mask[:, eos_token_id + 1 :])
                return mx.concatenate(parts, axis=-1)
            return eos_mask

        vocab_size = int(logits.shape[-1])
        if bitmask_cache is None or cached_vocab_size != vocab_size:
            bitmask_cache = llguidance.numpy.allocate_token_bitmask(1, vocab_size)
            cached_vocab_size = vocab_size
        llguidance.numpy.fill_next_token_bitmask(matcher, bitmask_cache, 0)
        return llguidance.mlx.apply_token_bitmask(logits, bitmask_cache)

    return processor


class MlxGuidedDisambiguator:
    """Lazy local ``mlx-lm`` + ``llguidance`` Stage 2 proper-noun disambiguator."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_DISAMBIGUATION_MODEL,
        entity_catalog: Mapping[str, DisambiguationCandidate] | None = None,
        window_chars: int = 240,
        max_tokens: int = 16,
    ) -> None:
        self.model_id = model_id
        self.entity_catalog: dict[str, DisambiguationCandidate] = dict(entity_catalog or {})
        self.window_chars = window_chars
        self.max_tokens = max_tokens
        self._model: Any = None
        self._tokenizer: Any = None
        self._ll_tokenizer: llguidance.LLTokenizer | None = None

    def _ensure_loaded(self) -> tuple[Any, Any, llguidance.LLTokenizer]:
        if self._model is None or self._tokenizer is None or self._ll_tokenizer is None:
            import mlx_lm
            from transformers import PreTrainedTokenizerFast

            loaded = mlx_lm.load(self.model_id)
            model = loaded[0]
            tokenizer = loaded[1]
            hf_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
            if not isinstance(hf_tokenizer, PreTrainedTokenizerFast):
                raise TypeError(
                    f"Expected PreTrainedTokenizerFast from {self.model_id}, "
                    f"got {type(hf_tokenizer).__name__}"
                )
            vocab_size: int | None = None
            model_args = getattr(model, "args", None)
            raw_vocab_size = getattr(model_args, "vocab_size", None)
            if isinstance(raw_vocab_size, int) and raw_vocab_size > 0:
                vocab_size = raw_vocab_size
            ll_tokenizer = llguidance.hf.from_tokenizer(hf_tokenizer, n_vocab=vocab_size)
            self._model = model
            self._tokenizer = tokenizer
            self._ll_tokenizer = ll_tokenizer
        return self._model, self._tokenizer, self._ll_tokenizer

    def disambiguate(
        self,
        *,
        text: str,
        start: int,
        end: int,
        surface: str,
        candidates: Sequence[DisambiguationCandidate],
    ) -> str | None:
        """Resolve ``surface`` at ``text[start:end]`` to one candidate CURIE or ``None``."""
        if not candidates:
            return None
        enriched = [self.entity_catalog.get(c.curie, c) for c in candidates]
        candidate_curies = [c.curie for c in enriched]
        valid_curies = set(candidate_curies)

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer, ll_tokenizer = self._ensure_loaded()
        matcher = compile_llguidance_matcher(ll_tokenizer, candidate_curies)
        eos_id = int(getattr(tokenizer, "eos_token_id", ll_tokenizer.eos_token))
        user_prompt = build_disambiguation_prompt(
            text=text,
            start=start,
            end=end,
            surface=surface,
            candidates=enriched,
            radius=self.window_chars,
        )
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            formatted_prompt = str(
                apply_template(
                    [{"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        else:
            formatted_prompt = user_prompt

        processor = make_mlx_llguidance_processor(matcher, eos_id)
        sampler = make_sampler(temp=0.0)
        raw_output = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=self.max_tokens,
            sampler=sampler,
            logits_processors=[processor],
            verbose=False,
        )
        decision = raw_output.strip()
        if decision == "NIL" or decision not in valid_curies:
            return None
        return decision


def build_entity_catalog(
    entity_definitions: Iterable[Mapping[str, Any]],
) -> dict[str, DisambiguationCandidate]:
    """Build a CURIE -> ``DisambiguationCandidate`` lookup map from entity definitions."""
    catalog: dict[str, DisambiguationCandidate] = {}
    for entity in entity_definitions:
        curie = entity.get("canonical_id")
        if not isinstance(curie, str) or not curie:
            continue
        label = entity.get("label")
        kind = entity.get("entity_kind") or entity.get("entity_type")
        desc = entity.get("description")
        catalog[curie] = DisambiguationCandidate(
            curie=curie,
            pref_label=label if isinstance(label, str) else "",
            collection_name=kind if isinstance(kind, str) else "",
            definition=desc if isinstance(desc, str) else "",
        )
    return catalog


def make_disambiguator(
    *,
    config: RuntimeConfig,
    entity_definitions: Iterable[Mapping[str, Any]] = (),
) -> MlxGuidedDisambiguator:
    """Create a lazy ``MlxGuidedDisambiguator`` configured from ``config.models``."""
    return MlxGuidedDisambiguator(
        model_id=config.models.disambiguation_model,
        entity_catalog=build_entity_catalog(entity_definitions),
    )
