"""Evaluate retrieval performance against labeled benchmark cases."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lxd.retrieval.query_pipeline import SearchOutcome, search_chunks
from lxd.settings.models import RuntimeConfig
from lxd.stores.sqlite.connection import build_store_paths, connect_sqlite, initialize_schema


@dataclass(frozen=True, slots=True)
class EvalCase:
    """Single labeled retrieval evaluation case."""

    question: str
    expected_source_files: list[str]
    domain: str | None
    category: str | None = None


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    """Per-question retrieval metrics and ranked outputs."""

    question: str
    recall_at_10: float
    mrr_at_10: float
    expected: list[str]
    ranked: list[str]
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class EvalSummary:
    """Aggregate retrieval evaluation metrics across cases."""

    question_count: int
    mean_recall_at_10: float
    mean_mrr_at_10: float
    cases: list[EvalCaseResult]


@dataclass(frozen=True, slots=True)
class RetrievalEvalCaseRecord:
    """Persisted per-case record — mirrors :class:`EvalCaseResult` but caps ``ranked`` at 10.

    Only the top-10 ranked sources are kept in the history file; the full
    ranked list is only needed for the live console report.
    """

    question: str
    recall_at_10: float
    mrr_at_10: float
    expected: list[str]
    ranked: list[str]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RetrievalEvalRun:
    """One persisted retrieval-eval run, mirroring the ``eval_quality_runs.jsonl`` pattern.

    Written as one JSONL line per run to ``<data_path>/retrieval_eval_runs.jsonl``
    so historical recall/MRR scores can be diffed between runs and gap
    tickets can be derived from any given run.
    """

    run_tag: str
    run_started_at: str
    run_finished_at: str
    question_count: int
    mean_recall_at_10: float
    mean_mrr_at_10: float
    cases: list[RetrievalEvalCaseRecord] = field(default_factory=list)


def recall_at_k(expected: set[str], ranked: list[str], k: int) -> float:
    """Compute recall@k for expected versus ranked sources.

    Args:
        expected: Expected relevant source paths.
        ranked: Ranked source paths from retrieval.
        k: Top-k cutoff.

    Returns:
        Recall score at cutoff `k`.
    """
    if not expected:
        return 0.0
    top_k = set(ranked[:k])
    return len(expected & top_k) / len(expected)


def mrr_at_k(expected: set[str], ranked: list[str], k: int) -> float:
    """Compute MRR@k for expected versus ranked sources.

    Args:
        expected: Expected relevant source paths.
        ranked: Ranked source paths from retrieval.
        k: Top-k cutoff.

    Returns:
        MRR score at cutoff `k`.
    """
    for index, item in enumerate(ranked[:k], start=1):
        if item in expected:
            return 1.0 / index
    return 0.0


def load_eval_cases(path: Path) -> list[EvalCase]:
    """Load evaluation cases from a JSON file.

    Args:
        path: Path to the source file or storage location.

    Returns:
        Validated evaluation cases from disk.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Eval set must be a JSON array.")
    cases: list[EvalCase] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each eval case must be an object.")
        question = item.get("question")
        expected_source_files = item.get("expected_source_files")
        domain = item.get("domain")
        category = item.get("category")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Each eval case requires a non-empty question.")
        if not isinstance(expected_source_files, list) or not all(
            isinstance(x, str) for x in expected_source_files
        ):
            raise ValueError("Each eval case requires expected_source_files as a list of strings.")
        if domain is not None and not isinstance(domain, str):
            raise ValueError("domain must be null or a string.")
        if category is not None and not isinstance(category, str):
            raise ValueError("category must be null or a string.")
        cases.append(
            EvalCase(
                question=question,
                expected_source_files=[str(path) for path in expected_source_files],
                domain=domain,
                category=category,
            )
        )
    return cases


def run_eval(
    cases: list[EvalCase],
    *,
    config: RuntimeConfig,
) -> EvalSummary:
    """Execute retrieval evaluation and aggregate metrics.

    Args:
        cases: Evaluation cases to execute.
        config: Runtime configuration object.

    Returns:
        Aggregate retrieval evaluation summary.
    """
    searchable_paths = _searchable_source_rel_paths(config)
    results: list[EvalCaseResult] = []
    for case in cases:
        outcome = search_chunks(
            question=case.question,
            config=config,
            domain=case.domain,
            limit=20,
        )
        ranked = _ranked_source_rel_paths(outcome)
        expected = _normalize_expected(case.expected_source_files, searchable_paths)
        results.append(
            EvalCaseResult(
                question=case.question,
                recall_at_10=recall_at_k(set(expected), ranked, 10),
                mrr_at_10=mrr_at_k(set(expected), ranked, 10),
                expected=expected,
                ranked=ranked,
                warnings=[*outcome.warnings, *outcome.config_drift_warnings],
            )
        )
    question_count = len(results)
    mean_recall = sum(item.recall_at_10 for item in results) / question_count if results else 0.0
    mean_mrr = sum(item.mrr_at_10 for item in results) / question_count if results else 0.0
    return EvalSummary(
        question_count=question_count,
        mean_recall_at_10=mean_recall,
        mean_mrr_at_10=mean_mrr,
        cases=results,
    )


def build_eval_run(
    summary: EvalSummary,
    *,
    run_started_at: str,
    run_finished_at: str,
    run_tag: str = "",
) -> RetrievalEvalRun:
    """Wrap an :class:`EvalSummary` into a persistable :class:`RetrievalEvalRun`.

    Args:
        summary: Aggregate result from :func:`run_eval`.
        run_started_at: ISO-8601 UTC timestamp the run started.
        run_finished_at: ISO-8601 UTC timestamp the run finished.
        run_tag: Optional caller-supplied label for this run (defaults to empty).

    Returns:
        A run record ready to append to ``retrieval_eval_runs.jsonl``.
    """
    return RetrievalEvalRun(
        run_tag=run_tag,
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
        question_count=summary.question_count,
        mean_recall_at_10=summary.mean_recall_at_10,
        mean_mrr_at_10=summary.mean_mrr_at_10,
        cases=[
            RetrievalEvalCaseRecord(
                question=case.question,
                recall_at_10=case.recall_at_10,
                mrr_at_10=case.mrr_at_10,
                expected=case.expected,
                ranked=case.ranked[:10],
                warnings=case.warnings,
            )
            for case in summary.cases
        ],
    )


def append_eval_run(run: RetrievalEvalRun, history_path: Path) -> None:
    """Append one JSONL row per retrieval-eval run, latest at EOF.

    Mirrors :func:`lxd.eval.report.append_run_to_history`: the history file
    lives next to the data store and is gitignored, and each line is a
    complete :class:`RetrievalEvalRun` so gap tickets can be rebuilt from
    any given run.
    """
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(run), sort_keys=False)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(payload)
        f.write("\n")


def _ranked_source_rel_paths(outcome: SearchOutcome) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for chunk in outcome.ranked:
        if chunk.source_rel_path in seen:
            continue
        seen.add(chunk.source_rel_path)
        ranked.append(chunk.source_rel_path)
    return ranked


def _normalize_expected(expected: list[str], searchable_paths: list[str]) -> list[str]:
    searchable_by_name: dict[str, set[str]] = {}
    for path in searchable_paths:
        searchable_by_name.setdefault(Path(path).name, set()).add(path)
    normalized: list[str] = []
    for item in expected:
        if "/" in item:
            normalized.append(item)
            continue
        matches = searchable_by_name.get(item, set())
        if not matches:
            normalized.append(item)
            continue
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous eval basename '{item}'. Use explicit source_rel_path instead."
            )
        normalized.append(next(iter(matches)))
    return normalized


def _searchable_source_rel_paths(config: RuntimeConfig) -> list[str]:
    store_paths = build_store_paths(config.paths.data_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        rows = connection.execute(
            """
            SELECT source_rel_path
            FROM corpus_manifest
            WHERE lifecycle_status != 'deleted'
              AND retrieval_status = 'searchable'
            ORDER BY source_rel_path
            """
        ).fetchall()
        return [str(row["source_rel_path"]) for row in rows]
    finally:
        connection.close()
