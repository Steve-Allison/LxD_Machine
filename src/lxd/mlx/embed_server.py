"""Local Octen-Embedding-8B server.

Run with ``python -m lxd.mlx.embed_server``. The process loads MLX weights,
then binds ``/v1/embeddings`` and ``/health``. Last-token pooling matches
the Octen model card. Each text is forwarded on its own so padding cannot
move the pooled position.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import mlx.core as mx
from mlx_lm import load

_model: Any = None
_tokenizer: Any = None
_MODEL_ID = os.environ.get("LXD_EMBED_MODEL_ID", "Octen/Octen-Embedding-8B")
_MAX_TOKENS = int(os.environ.get("LXD_EMBED_MAX_TOKENS", "8192"))
_DIM = 4096


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(200, {"status": "ok", "model": _MODEL_ID, "embedding_dim": _DIM})

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/v1/embeddings":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except UnicodeDecodeError, json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})
            return
        texts = _texts_from_payload(payload)
        if texts is None:
            self._send(400, {"error": "input must be a string or a list of strings"})
            return
        if len(texts) > 32:
            self._send(400, {"error": "batch size exceeds 32"})
            return
        try:
            vectors = _embed(texts)
        except (RuntimeError, ValueError, TypeError) as exc:
            self._send(500, {"error": str(exc)})
            return
        self._send(
            200,
            {
                "object": "list",
                "model": _MODEL_ID,
                "data": [
                    {"object": "embedding", "index": index, "embedding": vector}
                    for index, vector in enumerate(vectors)
                ],
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _texts_from_payload(payload: object) -> list[str] | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("input")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return list(raw)
    return None


def _embed(texts: list[str]) -> list[list[float]]:
    if _model is None or _tokenizer is None:
        raise RuntimeError("Octen embedding model is not loaded.")
    vectors: list[list[float]] = []
    for text in texts:
        tokens = _encode(text)
        if len(tokens) > _MAX_TOKENS:
            tokens = tokens[:_MAX_TOKENS]
        if not tokens:
            tokens = _encode(" ")
        pooled = _forward(mx.array([tokens]))
        mx.eval(pooled)
        row_data: Any = pooled.tolist()
        row = row_data[0]
        vectors.append([float(value) for value in row])
        del pooled
        mx.clear_cache()
    return vectors


def _forward(input_ids: mx.array) -> mx.array:
    """Last-token pool of the decoder stack, then L2-normalise."""
    body = _model.model
    hidden = body.embed_tokens(input_ids)
    seq_len = input_ids.shape[1]
    mask = mx.triu(
        mx.full((seq_len, seq_len), float("-inf"), dtype=hidden.dtype),
        k=1,
    )
    for layer in body.layers:
        hidden = layer(hidden, mask=mask, cache=None)
    hidden = body.norm(hidden)
    last = hidden[:, -1, :]
    norm = mx.linalg.norm(last, axis=1, keepdims=True)
    return last / mx.maximum(norm, mx.array(1e-9))


def _encode(text: str) -> list[int]:
    encoded = _tokenizer.encode(text)
    return [int(token) for token in encoded]


def _configure_memory() -> None:
    memory_gb = float(os.environ.get("LXD_MLX_MEMORY_LIMIT_GB", "24"))
    cache_gb = float(os.environ.get("LXD_MLX_CACHE_LIMIT_GB", "2"))
    mx.set_memory_limit(int(memory_gb * 1024**3))
    mx.set_cache_limit(int(cache_gb * 1024**3))


def main() -> None:
    """Load weights, then serve until the process is killed."""
    global _model, _tokenizer
    _configure_memory()
    weights = os.environ.get("LXD_EMBED_WEIGHTS", "chulcher/Octen-Embedding-8B-mlx")
    host = os.environ.get("LXD_EMBED_HOST", "127.0.0.1")
    port = int(os.environ.get("LXD_EMBED_PORT", "8100"))
    loaded: Any = load(weights)
    _model = loaded[0]
    _tokenizer = loaded[1]
    if not hasattr(_model, "model"):
        raise RuntimeError(f"MLX weights at {weights} have no decoder body.")
    server = ThreadingHTTPServer((host, port), _Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
