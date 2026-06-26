"""Tests for `estimate_run_cost` (B-STACK-10)."""

from pathlib import Path
from types import SimpleNamespace

from lxd.ingest.budget import estimate_run_cost
from lxd.ingest.scanner import ScannedCorpusFile


def _scanned(*, name: str, source_type: str, size_bytes: int) -> ScannedCorpusFile:
    return ScannedCorpusFile(
        absolute_path=Path(f"/tmp/{name}"),
        relative_path=name,
        source_type=source_type,
        file_size_bytes=size_bytes,
        content_hash=f"hash-{name}",
        source_domain="default",
    )


def _config(
    *,
    embed_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4o-mini",
    llm_cap: int | None = 1000,
) -> SimpleNamespace:
    return SimpleNamespace(
        models=SimpleNamespace(embed=embed_model, embed_dims=1536),
        relation_extraction=SimpleNamespace(openai_model=llm_model),
        ingest_budget=SimpleNamespace(max_llm_calls_per_run=llm_cap),
    )


def test_embedding_tokens_match_chars_per_4_round_up() -> None:
    files = [_scanned(name="a.md", source_type="markdown", size_bytes=4_000)]
    estimate = estimate_run_cost(files, _config())  # type: ignore[arg-type]

    assert estimate.embedding_tokens_est == 1_000


def test_embedding_tokens_round_up_for_partial_token() -> None:
    files = [_scanned(name="a.md", source_type="markdown", size_bytes=4_001)]
    estimate = estimate_run_cost(files, _config())  # type: ignore[arg-type]

    assert estimate.embedding_tokens_est == 1_001


def test_image_files_excluded_from_embedding_total() -> None:
    files = [
        _scanned(name="a.md", source_type="markdown", size_bytes=4_000),
        _scanned(name="b.png", source_type="image_png", size_bytes=1_000_000),
    ]
    estimate = estimate_run_cost(files, _config())  # type: ignore[arg-type]

    assert estimate.text_file_count == 1, "Image files must be excluded from text count."
    assert estimate.embedding_tokens_est == 1_000


def test_embedding_usd_uses_known_price_for_text_embedding_3_small() -> None:
    files = [_scanned(name="a.md", source_type="markdown", size_bytes=4_000_000)]
    estimate = estimate_run_cost(files, _config())  # type: ignore[arg-type]

    assert estimate.embedding_tokens_est == 1_000_000
    assert estimate.embedding_usd_est == 0.020


def test_llm_cap_drives_llm_total_tokens_and_usd() -> None:
    files = [_scanned(name="a.md", source_type="markdown", size_bytes=400)]
    estimate = estimate_run_cost(
        files,
        _config(llm_cap=1_000),  # type: ignore[arg-type]
        relation_prompt_tokens=2_000,
        relation_completion_tokens=500,
    )

    assert estimate.llm_call_cap == 1_000
    assert estimate.llm_total_tokens_est == 1_000 * (2_000 + 500)
    assert (
        estimate.llm_usd_est
        == (1_000 * 2_000 / 1_000_000) * 0.150 + (1_000 * 500 / 1_000_000) * 0.600
    )


def test_no_llm_cap_reports_zero_llm_total() -> None:
    files = [_scanned(name="a.md", source_type="markdown", size_bytes=400)]
    estimate = estimate_run_cost(files, _config(llm_cap=None))  # type: ignore[arg-type]

    assert estimate.llm_call_cap is None
    assert estimate.llm_total_tokens_est == 0
    assert estimate.llm_usd_est == 0.0


def test_total_usd_sums_embedding_and_llm() -> None:
    files = [_scanned(name="a.md", source_type="markdown", size_bytes=4_000_000)]
    estimate = estimate_run_cost(files, _config(llm_cap=1_000))  # type: ignore[arg-type]

    assert estimate.total_usd_est == estimate.embedding_usd_est + estimate.llm_usd_est


def test_unknown_embedding_model_reports_zero_usd_without_crash() -> None:
    files = [_scanned(name="a.md", source_type="markdown", size_bytes=4_000)]
    estimate = estimate_run_cost(
        files,
        _config(embed_model="unknown-model"),  # type: ignore[arg-type]
    )

    assert estimate.embedding_tokens_est == 1_000
    assert estimate.embedding_usd_est == 0.0
