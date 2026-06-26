"""ColBERT-style late-interaction reranker (in-process, MaxSim).

The ``colbert`` reranker backend encodes the query and each candidate
document into token-level vectors, then scores each (query, doc) pair
via MaxSim — the late-interaction operation ColBERT v2 introduced:

    score(q, d) = sum_{q_t in q} max_{d_t in d} cosine(q_t, d_t)

In words: every query token finds its best-matching document token; the
final score is the sum of those best matches. This preserves fine-grained
term matches that whole-sequence cross-encoders flatten and gives more
nuanced ranking on long documents.

We use ``BAAI/bge-m3`` by default — a transformer model that produces
ColBERT-style token vectors as a first-class output (alongside dense and
sparse). The model is loaded lazily on first use so test envs that never
exercise this backend don't pay the load cost.

Dependencies:
    Requires ``transformers`` and ``torch``. Both are already in
    ``pixi.toml`` (``transformers >= 5``). The model weights download
    on first use into the HF cache.

Performance note:
    Late interaction is O(query_tokens * doc_tokens) per pair. For the
    LxD wiki scale (<= 200 docs * <= 512 tokens) this fits comfortably in
    a single forward pass per candidate on Apple Silicon (MPS). For
    larger corpora, pre-encoded document vectors should be stored in
    LanceDB — that optimisation is intentionally deferred so this Phase
    ships within scope.
"""

from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, Final

import structlog

if TYPE_CHECKING:
    from lxd.retrieval.query_pipeline import RankedChunk
    from lxd.settings.models import RuntimeConfig

_log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ColbertScored:
    """One (chunk, score) pair from MaxSim reranking."""

    chunk: RankedChunk
    score: float


_MODEL_CACHE_LOCK: Final = Lock()
_MODEL_CACHE: dict[str, _ColbertModel] = {}


@dataclass(frozen=True, slots=True)
class _ColbertModel:
    """Lazy-loaded model + tokenizer pair."""

    name: str
    tokenizer: Any
    model: Any
    device: str


def get_colbert_model(name: str) -> _ColbertModel:
    """Return a process-wide cached ColBERT-style model.

    Lazy load: imports torch and transformers only when first called.
    Subsequent calls reuse the cached instance — the model is several
    GB of weights, loading it twice would be wasteful.

    Raises:
        RuntimeError: If torch / transformers are not installed.
    """
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(name)
        if cached is not None:
            return cached
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The colbert reranker backend requires `torch` and "
                "`transformers`; both are in pixi.toml but failed to import."
            ) from exc

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        model.eval()
        model.to(device)
        _log.info("colbert_model_loaded", model=name, device=device)
        loaded = _ColbertModel(name=name, tokenizer=tokenizer, model=model, device=device)
        _MODEL_CACHE[name] = loaded
        return loaded


def encode_tokens(
    *,
    texts: list[str],
    colbert_model: _ColbertModel,
    max_length: int,
) -> Any:
    """Encode a batch of texts into normalised token-level vectors.

    Returns a tensor of shape ``(batch, seq, hidden)`` with each token
    vector L2-normalised so cosine similarity reduces to dot product.
    Attention-masked positions are zeroed so they contribute nothing to
    downstream MaxSim.
    """
    import torch

    tokenizer = colbert_model.tokenizer
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(colbert_model.device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = colbert_model.model(**encoded)

    hidden = outputs.last_hidden_state  # (batch, seq, hidden)
    hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
    # Zero out padding positions so they cannot win a MaxSim comparison.
    attention_mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return hidden * attention_mask


def maxsim_score(query_tokens: Any, doc_tokens: Any) -> float:
    """Compute MaxSim score between query and document token vectors.

    ``query_tokens`` and ``doc_tokens`` are tensors of shape
    ``(q_seq, hidden)`` and ``(d_seq, hidden)`` respectively.
    Returns the scalar sum-of-maxes score, a float in roughly ``[0, q_seq]``.
    """
    import torch

    # (q_seq, hidden) @ (hidden, d_seq) -> (q_seq, d_seq) cosine matrix
    # (vectors are pre-normalised, so dot product == cosine).
    sim_matrix = torch.matmul(query_tokens, doc_tokens.transpose(0, 1))
    # For each query token, take the max similarity across doc tokens.
    per_query_max = sim_matrix.max(dim=1).values
    return float(per_query_max.sum().item())


def colbert_rerank(
    *,
    question: str,
    candidates: list[RankedChunk],
    config: RuntimeConfig,
) -> list[ColbertScored]:
    """Rerank candidates via late-interaction MaxSim.

    Args:
        question: The user query.
        candidates: Ordered ranked chunks (dense-retrieval output).
        config: Runtime config — uses ``reranker.colbert_model`` and
            ``reranker.colbert_max_length``.

    Returns:
        Candidates re-sorted by descending MaxSim score, each wrapped in
        :class:`ColbertScored` so callers know the raw score.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    if not candidates:
        return []

    colbert_model = get_colbert_model(config.reranker.colbert_model)
    max_length = config.reranker.colbert_max_length

    # Encode the query once, candidates once each, in a single forward pass.
    all_texts = [question, *(c.text for c in candidates)]
    embeddings = encode_tokens(
        texts=all_texts,
        colbert_model=colbert_model,
        max_length=max_length,
    )
    query_tokens = embeddings[0]
    doc_token_batch = embeddings[1:]

    scored: list[ColbertScored] = []
    for candidate, doc_tokens in zip(candidates, doc_token_batch, strict=True):
        score = maxsim_score(query_tokens, doc_tokens)
        scored.append(ColbertScored(chunk=candidate, score=score))

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored
