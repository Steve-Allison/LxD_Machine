"""In-process Qwen3 reranker scoring on MLX.

The relevance of a query/document pair is the softmax probability of the
``yes`` token against ``no`` at the last position. The prompt matches the
mlx-community Qwen3-Reranker scoring recipe. Weights load on the first
score call, not at import.
"""

from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mlx_lm import load

from lxd.settings.models import RuntimeConfig

_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_loaded: dict[str, tuple[Any, Any, int, int, list[int], list[int]]] = {}
_document_scorer: Callable[[str, list[str]], list[float]] | None = None

_CACHE_LIMIT_BYTES = 2 * 1024**3


def format_rerank_body(query: str, document: str, instruction: str) -> str:
    """Return the user-turn body placed between the Qwen3 reranker prefix and suffix."""
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def set_document_scorer(scorer: Callable[[str, list[str]], list[float]] | None) -> None:
    """Replace the MLX scorer. Tests pass a fake; ``None`` restores the model."""
    global _document_scorer
    _document_scorer = scorer


def probe_mlx_reranker(config: RuntimeConfig) -> tuple[bool, str | None]:
    """Report whether the MLX reranker package can score.

    Weights are not downloaded here. The first ``score_documents`` call
    loads ``models.rerank``.
    """
    if not config.models.rerank.strip():
        return False, "models.rerank is empty."
    if config.mlx is None:
        return False, "reranker.backend=mlx requires an mlx config section."
    return True, None


def score_documents(
    query: str,
    documents: list[str],
    *,
    model_id: str,
    instruction: str,
    max_length: int,
) -> list[float]:
    """Return one relevance score per document, in input order."""
    if _document_scorer is not None:
        return _document_scorer(query, documents)
    if not documents:
        return []
    model, tokenizer, yes_id, no_id, prefix, suffix = _load(model_id)
    scores: list[float] = []
    budget = max_length - len(prefix) - len(suffix)
    if budget < 8:
        raise RuntimeError(
            f"rerank_max_length={max_length} is shorter than the Qwen3 reranker prompt."
        )
    for document in documents:
        body = format_rerank_body(query, document, instruction)
        content = _encode(tokenizer, body)[:budget]
        token_ids = prefix + content + suffix
        logits = model(mx.array([token_ids]))
        if isinstance(logits, tuple):
            logits = logits[0]
        last = logits[0, -1]
        pair = mx.stack([last[no_id], last[yes_id]])
        yes_prob = mx.exp((pair - mx.logsumexp(pair))[1])
        mx.eval(yes_prob)
        scores.append(float(yes_prob))
        del logits, last, pair, yes_prob
    mx.clear_cache()
    return scores


def _load(model_id: str) -> tuple[Any, Any, int, int, list[int], list[int]]:
    cached = _loaded.get(model_id)
    if cached is not None:
        return cached
    mx.set_cache_limit(_CACHE_LIMIT_BYTES)
    loaded_bundle: Any = load(model_id)
    model = loaded_bundle[0]
    tokenizer = loaded_bundle[1]
    yes_id = _token_id(tokenizer, "yes")
    no_id = _token_id(tokenizer, "no")
    prefix = _encode(tokenizer, _PREFIX)
    suffix = _encode(tokenizer, _SUFFIX)
    loaded = (model, tokenizer, yes_id, no_id, prefix, suffix)
    _loaded[model_id] = loaded
    return loaded


def _encode(tokenizer: Any, text: str) -> list[int]:
    inner = getattr(tokenizer, "_tokenizer", None)
    encoder = inner if inner is not None and hasattr(inner, "encode") else tokenizer
    try:
        encoded = encoder.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = encoder.encode(text)
    return [int(token) for token in encoded]


def _token_id(tokenizer: Any, token: str) -> int:
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    convert = getattr(inner, "convert_tokens_to_ids", None)
    if convert is not None:
        token_id = convert(token)
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
    encoded = _encode(tokenizer, token)
    if len(encoded) != 1:
        raise RuntimeError(f"Expected one token id for {token!r}, got {encoded}.")
    return encoded[0]
