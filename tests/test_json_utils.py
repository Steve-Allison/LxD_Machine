"""Unit tests for lxd.domain.json_utils."""

from lxd.domain.json_utils import (
    extract_json_dict,
    extract_json_list,
    extract_json_payload,
    strip_thinking,
)


def test_strip_thinking() -> None:
    assert strip_thinking("") == ""
    assert strip_thinking("Hello world") == "Hello world"
    assert strip_thinking("<think>reasoning step</think>Final answer") == "Final answer"
    assert (
        strip_thinking(
            "<reasoning>\n1. check ontology\n2. check prompt\n</reasoning>\nActual response"
        )
        == "Actual response"
    )
    assert strip_thinking("<THINK>capitalized</THINK>done") == "done"


def test_extract_json_payload_clean() -> None:
    assert extract_json_payload('{"key": "value"}') == {"key": "value"}
    assert extract_json_payload("[1, 2, 3]") == [1, 2, 3]


def test_extract_json_with_thinking_blocks() -> None:
    raw = """<think>
    I need to structure the learning objectives.
    Objective 1: explain.
    Objective 2: apply.
    </think>
    {"objectives": ["Explain ADDIE", "Apply ADDIE"]}"""

    result = extract_json_dict(raw)
    assert result == {"objectives": ["Explain ADDIE", "Apply ADDIE"]}


def test_extract_json_with_markdown_fences() -> None:
    raw_with_json_tag = """Here is the plan:
```json
{
  "modality_plan": "Blended ILT"
}
```
Hope this helps!"""
    assert extract_json_dict(raw_with_json_tag) == {"modality_plan": "Blended ILT"}

    raw_without_tag = """```
{"outline": ["Module 1", "Module 2"]}
```"""
    assert extract_json_dict(raw_without_tag) == {"outline": ["Module 1", "Module 2"]}


def test_extract_json_with_preamble_no_fences() -> None:
    raw = 'Sure! The assessment items are: ["Quiz 1", "Scenario 2"]. Let me know if you need more.'
    assert extract_json_list(raw) == ["Quiz 1", "Scenario 2"]

    raw_dict = 'Understood. Output: {"score": 0.85, "feedback": ["Good"]} That is all.'
    assert extract_json_dict(raw_dict) == {"score": 0.85, "feedback": ["Good"]}


def test_extract_json_with_trailing_commas() -> None:
    raw_dict = '{"a": 1, "b": 2,}'
    assert extract_json_dict(raw_dict) == {"a": 1, "b": 2}

    raw_list = "[1, 2, 3, ]"
    assert extract_json_list(raw_list) == [1, 2, 3]


def test_extract_json_fallbacks() -> None:
    assert extract_json_payload("not json at all", default=None) is None
    assert extract_json_dict("broken {json", default={"fallback": True}) == {"fallback": True}
    assert extract_json_dict("") == {}
    assert extract_json_list("") == []
    # Enforces expected container types
    assert extract_json_dict("[1, 2, 3]") == {}
    assert extract_json_list('{"a": 1}') == []
