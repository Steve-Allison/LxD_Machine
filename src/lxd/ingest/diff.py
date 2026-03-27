"""Compare corpus scans to detect created, updated, and deleted files."""

from __future__ import annotations

from dataclasses import dataclass

from lxd.ingest.scanner import ScannedCorpusFile


@dataclass(frozen=True)
class ScanDiff:
    """Set-wise diff between previous and current corpus scans."""
    new_paths: set[str]
    deleted_paths: set[str]
    unchanged_paths: set[str]


def diff_scans(previous: list[ScannedCorpusFile], current: list[ScannedCorpusFile]) -> ScanDiff:
    """Compare scan snapshots and classify path changes.

    Args:
        previous: Previously scanned corpus files.
        current: Current scanned corpus files.

    Returns:
        Path sets partitioned into new, deleted, and unchanged.
    """
    previous_paths = {item.relative_path for item in previous}
    current_paths = {item.relative_path for item in current}
    return ScanDiff(
        new_paths=current_paths - previous_paths,
        deleted_paths=previous_paths - current_paths,
        unchanged_paths=previous_paths & current_paths,
    )
