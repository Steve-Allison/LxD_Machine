"""Tests for ``LearnerBrief`` and its session-persistence wiring in query_pipeline."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from lxd.domain.brief import LearnerBrief
from lxd.retrieval import query_pipeline as _query_pipeline
from lxd.settings.models import RuntimeConfig
from lxd.stores.sqlite.connection import build_store_paths, connect_sqlite, initialize_schema
from lxd.stores.sqlite.sessions import load_session, load_turns
from lxd.synthesis.answering import AnswerEnvelope

# Private helpers exercised intentionally — same logical unit as
# query_pipeline; brief resolution / turn persistence have no public API.
_resolve_learner_brief = _query_pipeline._resolve_learner_brief  # pyright: ignore[reportPrivateUsage]
_persist_session_turn = _query_pipeline._persist_session_turn  # pyright: ignore[reportPrivateUsage]


def _config(data_path: Path) -> RuntimeConfig:
    return cast("RuntimeConfig", SimpleNamespace(paths=SimpleNamespace(data_path=data_path)))


def _init_store(data_path: Path) -> None:
    store_paths = build_store_paths(data_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# LearnerBrief model
# ---------------------------------------------------------------------------


def test_learner_brief_is_empty_true_for_bare_session_id() -> None:
    assert LearnerBrief(session_id="s1").is_empty()


def test_learner_brief_is_empty_false_when_any_field_set() -> None:
    assert not LearnerBrief(audience="admins").is_empty()
    assert not LearnerBrief(modality="ILT").is_empty()
    assert not LearnerBrief(bloom_target="apply").is_empty()
    assert not LearnerBrief(constraints="1 day").is_empty()


def test_merge_over_request_fields_take_precedence() -> None:
    request = LearnerBrief(audience="new admins", session_id="s1")
    stored = LearnerBrief(audience="old admins", modality="ILT", session_id="s1")
    merged = request.merge_over(stored)
    assert merged.audience == "new admins", "request-supplied field must win"
    assert merged.modality == "ILT", "unset request field must fall back to stored"
    assert merged.session_id == "s1"


def test_merge_over_falls_back_to_stored_session_id_when_request_unset() -> None:
    request = LearnerBrief(audience="admins")
    stored = LearnerBrief(session_id="s1")
    merged = request.merge_over(stored)
    assert merged.session_id == "s1"


# ---------------------------------------------------------------------------
# _resolve_learner_brief
# ---------------------------------------------------------------------------


def test_resolve_learner_brief_without_session_id_is_a_no_op(tmp_path: Path) -> None:
    config = _config(tmp_path)  # store not even initialised
    request = LearnerBrief(audience="admins")
    resolved = _resolve_learner_brief(config, request)
    assert resolved == request


def test_resolve_learner_brief_degrades_gracefully_when_store_missing(tmp_path: Path) -> None:
    config = _config(tmp_path)  # sqlite file does not exist
    request = LearnerBrief(audience="admins", session_id="s1")
    resolved = _resolve_learner_brief(config, request)
    assert resolved == request


def test_resolve_learner_brief_persists_first_seen_brief(tmp_path: Path) -> None:
    _init_store(tmp_path)
    config = _config(tmp_path)
    request = LearnerBrief(audience="new admins", modality="ILT", session_id="s1")

    resolved = _resolve_learner_brief(config, request)
    assert resolved.audience == "new admins"
    assert resolved.modality == "ILT"

    store_paths = build_store_paths(tmp_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        stored = load_session(connection, "s1")
    finally:
        connection.close()
    assert stored is not None
    assert stored.audience == "new admins"
    assert stored.modality == "ILT"


def test_resolve_learner_brief_merges_request_over_prior_session(tmp_path: Path) -> None:
    _init_store(tmp_path)
    config = _config(tmp_path)

    first_turn = LearnerBrief(audience="new admins", modality="ILT", session_id="s1")
    _resolve_learner_brief(config, first_turn)

    # Second turn only supplies bloom_target; audience/modality should
    # carry forward from the persisted brief.
    second_turn = LearnerBrief(bloom_target="analyze", session_id="s1")
    resolved = _resolve_learner_brief(config, second_turn)

    assert resolved.audience == "new admins"
    assert resolved.modality == "ILT"
    assert resolved.bloom_target == "analyze"


def test_resolve_learner_brief_request_field_overrides_stored(tmp_path: Path) -> None:
    _init_store(tmp_path)
    config = _config(tmp_path)

    _resolve_learner_brief(config, LearnerBrief(audience="new admins", session_id="s1"))
    resolved = _resolve_learner_brief(
        config, LearnerBrief(audience="revised audience", session_id="s1")
    )
    assert resolved.audience == "revised audience"


# ---------------------------------------------------------------------------
# _persist_session_turn
# ---------------------------------------------------------------------------


def _answer(status: str = "answered", text: str = "The answer.") -> AnswerEnvelope:
    from lxd.domain.status import QueryAnswerStatus

    return AnswerEnvelope(
        answer_status=QueryAnswerStatus(status),
        answer_text=text,
        citations=["a", "b"],
        warnings=[],
        metadata={},
    )


def test_persist_session_turn_without_session_id_is_a_no_op(tmp_path: Path) -> None:
    config = _config(tmp_path)  # store not even initialised
    _persist_session_turn(config, None, question="q", answer=_answer())  # must not raise


def test_persist_session_turn_degrades_gracefully_when_store_missing(tmp_path: Path) -> None:
    config = _config(tmp_path)  # sqlite file does not exist
    _persist_session_turn(config, "s1", question="q", answer=_answer())  # must not raise


def test_persist_session_turn_appends_user_and_assistant_turns(tmp_path: Path) -> None:
    _init_store(tmp_path)
    config = _config(tmp_path)
    _resolve_learner_brief(config, LearnerBrief(session_id="s1"))

    _persist_session_turn(config, "s1", question="What is ADDIE?", answer=_answer())

    store_paths = build_store_paths(tmp_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        turns = load_turns(connection, "s1")
        session = load_session(connection, "s1")
    finally:
        connection.close()

    assert [t.role for t in turns] == ["user", "assistant"]
    assert "What is ADDIE?" in turns[0].content_json
    assert session is not None
    assert '"citations"' in session.last_artefact_json


def test_persist_session_turn_is_safe_to_call_without_prior_brief_resolution(
    tmp_path: Path,
) -> None:
    """A caller could pass session_id straight through without ever calling
    ``_resolve_learner_brief`` first (e.g. a future direct MCP surface).
    ``append_turn``'s FK requires a parent ``sessions`` row — the store
    layer must degrade to a logged warning, not raise, per
    ``_persist_session_turn``'s graceful-degradation contract."""
    _init_store(tmp_path)
    config = _config(tmp_path)
    _persist_session_turn(config, "no-such-session", question="q", answer=_answer())
    store_paths = build_store_paths(tmp_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        session = load_session(connection, "no-such-session")
    finally:
        connection.close()
    assert session is None
