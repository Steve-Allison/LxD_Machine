"""Rerank retrieved chunks with cross-encoder or API models."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from lxd.settings.models import RuntimeConfig

if TYPE_CHECKING:
    from lxd.retrieval.query_pipeline import RankedChunk


_probe_cache: dict[tuple[str, str, str, str, str], tuple[bool, str | None]] = {}
_RUNTIME_DIRNAME = "runtime"


@dataclass(frozen=True)
class RerankOutcome:
    """Reranked chunk list and reranker execution status."""

    ranked: list[RankedChunk]
    warnings: list[str]
    applied: bool


def probe_reranker(config: RuntimeConfig) -> tuple[bool, str | None]:
    """Probe backend availability and return probe metadata.

    Args:
        config: Runtime configuration object.

    Returns:
        Tuple of `(supported, warning)` for reranker readiness.
    """
    cache_key = _cache_key(config)
    cached = _probe_cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        _ensure_reranker_service(config)
    except RuntimeError as exc:
        outcome = (False, str(exc))
        _probe_cache[cache_key] = outcome
        return outcome
    outcome = _probe_reranker_uncached(config)
    _probe_cache[cache_key] = outcome
    return outcome


def rerank_chunks(
    question: str,
    candidates: list[RankedChunk],
    config: RuntimeConfig,
) -> RerankOutcome:
    """Rerank candidate chunks for a question.

    Args:
        question: User question text.
        candidates: Candidate chunks to rerank.
        config: Runtime configuration object.

    Returns:
        Reranked chunk candidates and rerank status.
    """
    if not candidates:
        return RerankOutcome(ranked=[], warnings=[], applied=False)
    supported, warning = probe_reranker(config)
    if not supported:
        return RerankOutcome(
            ranked=candidates,
            warnings=[warning] if warning is not None else [],
            applied=False,
        )
    try:
        with _client(config) as client:
            response = client.post(
                _endpoint_path(config),
                json={
                    "model": config.models.rerank,
                    "query": question,
                    "documents": [candidate.text for candidate in candidates],
                    "top_n": len(candidates),
                },
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        return RerankOutcome(ranked=candidates, warnings=[str(exc)], applied=False)
    try:
        payload = response.json()
    except ValueError as exc:
        return RerankOutcome(
            ranked=candidates,
            warnings=[f"Invalid rerank response payload: {exc}"],
            applied=False,
        )
    reranked = _apply_rerank_payload(candidates, payload)
    if reranked is None:
        return RerankOutcome(
            ranked=candidates,
            warnings=["Rerank payload did not contain usable results."],
            applied=False,
        )
    return RerankOutcome(ranked=reranked, warnings=[], applied=True)


def _probe_reranker_uncached(config: RuntimeConfig) -> tuple[bool, str | None]:
    if config.reranker.url is None:
        return False, "reranker.url must be configured for the llama.cpp backend."
    return _probe_reranker_http(config)


def _probe_reranker_http(config: RuntimeConfig) -> tuple[bool, str | None]:
    try:
        with _client(config) as client:
            response = client.post(
                _endpoint_path(config),
                json={
                    "model": config.models.rerank,
                    "query": "probe query",
                    "documents": ["first probe document", "second probe document"],
                    "top_n": 2,
                },
            )
    except httpx.HTTPError as exc:
        return False, str(exc)
    if response.status_code == 404:
        return (
            False,
            (
                "Configured llama.cpp reranker is unavailable because the server does not expose "
                f"{_endpoint_path(config)}."
            ),
        )
    if response.status_code >= 400:
        return False, response.text
    try:
        payload = response.json()
    except ValueError as exc:
        return False, f"Invalid rerank response payload: {exc}"
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return False, "Rerank probe returned an unexpected payload."
    return True, None


def _ensure_reranker_service(config: RuntimeConfig) -> None:
    launch = getattr(config.reranker, "launch", None)
    if launch is None or not launch.auto_start:
        return
    if _probe_reranker_http(config)[0]:
        return

    runtime_dir = config.paths.data_path / _RUNTIME_DIRNAME
    runtime_dir.mkdir(parents=True, exist_ok=True)
    pid_path, log_path = _runtime_paths(config)

    running_pid = _load_running_pid(pid_path)
    if running_pid is not None:
        if _wait_for_reranker_ready(config, timeout_secs=launch.startup_timeout_secs):
            return
        raise RuntimeError(
            f"Configured llama.cpp reranker process pid={running_pid} did not become ready at "
            f"{config.reranker.url}{_endpoint_path(config)}. Check {log_path}."
        )

    command = _build_llama_server_command(config)
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    _write_pid_file(pid_path, log_path, process.pid, command)
    if _wait_for_reranker_ready(config, timeout_secs=launch.startup_timeout_secs):
        return
    exit_code = process.poll()
    if exit_code is None:
        raise RuntimeError(
            "Configured llama.cpp reranker was started but did not become ready within "
            f"{launch.startup_timeout_secs} seconds. Check {log_path}."
        )
    raise RuntimeError(
        f"Configured llama.cpp reranker exited with code {exit_code}. "
        f"Check {log_path}. Last log lines:\n{_tail_log(log_path)}"
    )


def _apply_rerank_payload(candidates: list[RankedChunk], payload: Any) -> list[RankedChunk] | None:
    if not isinstance(payload, dict):
        return None
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return None
    scored: list[tuple[float, RankedChunk]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        score = item.get("relevance_score")
        if not isinstance(index, int) or not isinstance(score, (int, float)):
            continue
        if index < 0 or index >= len(candidates):
            continue
        candidate = candidates[index]
        reranked_candidate = candidate.__class__(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            citation_label=candidate.citation_label,
            source_rel_path=candidate.source_rel_path,
            source_path=candidate.source_path,
            source_filename=candidate.source_filename,
            source_type=candidate.source_type,
            source_domain=candidate.source_domain,
            source_hash=candidate.source_hash,
            chunk_index=candidate.chunk_index,
            chunk_occurrence=candidate.chunk_occurrence,
            token_count=candidate.token_count,
            text=candidate.text,
            score_hint=candidate.score_hint,
            metadata_json=candidate.metadata_json,
            score=float(score),
        )
        scored.append((float(score), reranked_candidate))
    if not scored:
        return None
    return [candidate for _, candidate in sorted(scored, key=lambda item: item[0], reverse=True)]


def _endpoint_path(config: RuntimeConfig) -> str:
    endpoint = config.reranker.endpoint.strip()
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return endpoint


def _cache_key(config: RuntimeConfig) -> tuple[str, str, str, str, str]:
    launch = config.reranker.launch
    launch_signature = "none"
    if launch is not None:
        launch_signature = "|".join(
            [
                launch.executable,
                launch.model_source,
                str(launch.model_path) if launch.model_path is not None else "",
                launch.host,
                str(launch.port),
                str(launch.auto_start),
                ",".join(launch.extra_args),
            ]
        )
    return (
        config.reranker.backend,
        str(config.reranker.url) if config.reranker.url is not None else "",
        _endpoint_path(config),
        config.models.rerank,
        launch_signature,
    )


def _client(config: RuntimeConfig) -> httpx.Client:
    if config.reranker.url is None:
        raise ValueError("reranker.url must be configured.")
    return httpx.Client(
        base_url=str(config.reranker.url),
        timeout=float(config.reranker.timeout_secs),
    )


def _wait_for_reranker_ready(config: RuntimeConfig, *, timeout_secs: int) -> bool:
    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        ready, _ = _probe_reranker_http(config)
        if ready:
            return True
        time.sleep(0.5)
    return False


def _build_llama_server_command(config: RuntimeConfig) -> list[str]:
    launch = config.reranker.launch
    if launch is None:
        raise RuntimeError("reranker.launch must be configured to auto-start llama.cpp.")
    executable = _resolve_llama_server_executable(launch.executable)
    model_path = _resolve_reranker_model_path(config)
    return [
        executable,
        "--model",
        str(model_path),
        "--embedding",
        "--pooling",
        "rank",
        "--alias",
        config.models.rerank,
        "--host",
        launch.host,
        "--port",
        str(launch.port),
        "--reranking",
        *launch.extra_args,
    ]


def _resolve_llama_server_executable(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(f"Could not locate reranker executable '{executable}' on PATH.")
    return resolved


def _resolve_reranker_model_path(config: RuntimeConfig) -> Path:
    launch = config.reranker.launch
    if launch is None:
        raise RuntimeError("reranker.launch must be configured to resolve the reranker model.")
    if launch.model_source == "model_path":
        if launch.model_path is None:
            raise RuntimeError("reranker.launch.model_path must be configured.")
        if not launch.model_path.exists():
            raise RuntimeError(
                f"Configured reranker model path does not exist: {launch.model_path}"
            )
        return launch.model_path
    if launch.model_source == "ollama_blob":
        return _resolve_ollama_blob_model_path(config.models.rerank)
    raise RuntimeError(f"Unsupported reranker launch model_source: {launch.model_source}")


def _resolve_ollama_blob_model_path(model_name: str) -> Path:
    result = subprocess.run(
        ["ollama", "show", "--modelfile", model_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"Could not resolve local Ollama reranker model '{model_name}'. {stderr}".strip()
        )
    for line in result.stdout.splitlines():
        if not line.startswith("FROM "):
            continue
        candidate = Path(line.removeprefix("FROM ").strip())
        if not candidate.is_absolute():
            break
        if not candidate.exists():
            raise RuntimeError(
                f"Ollama modelfile for '{model_name}' points to a missing blob: {candidate}"
            )
        return candidate
    raise RuntimeError(
        f"Could not resolve a local model blob from 'ollama show --modelfile {model_name}'."
    )


def _write_pid_file(pid_path: Path, log_path: Path, pid: int, command: list[str]) -> None:
    pid_path.write_text(
        json.dumps(
            {
                "pid": pid,
                "command": command,
                "log_path": str(log_path),
                "started_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_running_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        payload = json.loads(pid_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pid_path.unlink(missing_ok=True)
        return None
    pid = payload.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        pid_path.unlink(missing_ok=True)
        return None
    if _process_is_running(pid):
        return pid
    pid_path.unlink(missing_ok=True)
    return None


def _process_is_running(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _tail_log(log_path: Path, *, lines: int = 20) -> str:
    if not log_path.exists():
        return "<no log output>"
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = content[-lines:]
    return "\n".join(tail) if tail else "<no log output>"


def _runtime_paths(config: RuntimeConfig) -> tuple[Path, Path]:
    runtime_dir = config.paths.data_path / _RUNTIME_DIRNAME
    runtime_dir.mkdir(parents=True, exist_ok=True)
    host = config.reranker.launch.host if config.reranker.launch is not None else "unknown-host"
    port = config.reranker.launch.port if config.reranker.launch is not None else 0
    model_slug = _slugify(config.models.rerank)
    stem = f"reranker-llama-server-{host}-{port}-{model_slug}"
    return runtime_dir / f"{stem}.json", runtime_dir / f"{stem}.log"


def _slugify(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")
