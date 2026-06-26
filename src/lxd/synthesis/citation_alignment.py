"""Per-sentence citation alignment for the synthesised answer.

The synthesis preamble asks the model to mark each assertion with one or
more ``[citation_label]`` markers inline. This module parses those markers
back out so the answer can be reported as a list of
``SentenceCitation(text, citation_labels)`` — one entry per sentence.

Why per-sentence (vs the existing bag-of-citations list):

  Bag-level citations let a client say "this answer is grounded in these
  N sources" but not "which source supports sentence 3". Per-sentence
  alignment makes hallucinations visible: any sentence with empty
  citations is making a claim the model couldn't (or didn't) attribute,
  which is exactly the failure mode citation discipline exists to surface.

Robustness:
  - Citation markers that don't match any known label are dropped silently
    (the model occasionally invents nearby-looking strings).
  - Sentences are split conservatively — we don't try to handle every
    edge case of English punctuation; the goal is "useful enough for the
    UI to render", not perfect linguistic segmentation.
  - The display text strips out the ``[label]`` markers so the user sees
    a clean answer; the markers' information is captured in
    ``citation_labels``.
"""

import re

from pydantic import BaseModel, ConfigDict, Field


class SentenceCitation(BaseModel):
    """One sentence of the answer plus the citation labels supporting it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(description="Sentence text with citation markers stripped.")
    citation_labels: list[str] = Field(
        default_factory=list,
        description=(
            "Citation labels the model attributed to this sentence. Empty "
            "means the model made a claim it could not attribute — flag for "
            "review."
        ),
    )


# Match anything resembling [label] where label is non-greedy and contains
# no newline or nested brackets. We validate against the known set later.
_CITATION_MARKER_RE = re.compile(r"\[([^\[\]\n]+)\]")

# Sentence splitter: end-of-sentence punctuation followed by whitespace and
# either end-of-text or a capital letter / opening bracket / digit. This
# misses edge cases (e.g. abbreviations), but is good enough for
# typical RAG-synthesised prose.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\(0-9])")


def align_citations(
    *,
    answer_text: str,
    valid_labels: list[str],
) -> list[SentenceCitation]:
    """Split ``answer_text`` into sentences and extract citation markers.

    Args:
        answer_text: The synthesised answer body.
        valid_labels: Citation labels known to be valid. Markers whose label
            is not in this set are dropped from the per-sentence labels (but
            kept in the text — we strip ALL markers regardless of validity
            so the display copy stays clean).

    Returns:
        One :class:`SentenceCitation` per detected sentence, in source
        order. Empty answer text returns an empty list.
    """
    text = answer_text.strip()
    if not text:
        return []

    valid_set = set(valid_labels)
    sentences = _SENTENCE_SPLIT_RE.split(text)

    results: list[SentenceCitation] = []
    for raw in sentences:
        sentence = raw.strip()
        if not sentence:
            continue
        labels = _extract_valid_labels(sentence, valid_set)
        cleaned = _strip_markers(sentence)
        results.append(SentenceCitation(text=cleaned, citation_labels=labels))
    return results


def _extract_valid_labels(sentence: str, valid_set: set[str]) -> list[str]:
    """Return citation labels from a sentence, deduplicated and validated."""
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _CITATION_MARKER_RE.finditer(sentence):
        label = match.group(1).strip()
        if label not in valid_set or label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return ordered


def _strip_markers(sentence: str) -> str:
    """Remove every ``[label]`` marker and tidy the surrounding whitespace.

    Preserves the sentence's final punctuation (``.``, ``!``, ``?``, ``;``)
    by leaving it untouched after the marker pass. Removes the orphan
    whitespace that markers leave behind (``"X [a] Y"`` → ``"X Y"``) and
    collapses runs of internal whitespace to a single space.
    """
    stripped = _CITATION_MARKER_RE.sub("", sentence)
    # Collapse internal whitespace runs (including those introduced by
    # marker removal) to a single space.
    collapsed = re.sub(r"\s+", " ", stripped)
    # Tighten spaces immediately before terminal punctuation that we just
    # exposed by stripping a marker (``"X . "`` -> ``"X."``).
    collapsed = re.sub(r"\s+([.,;:!?])", r"\1", collapsed)
    return collapsed.strip()
