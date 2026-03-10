from __future__ import annotations

from dataclasses import dataclass

from lxd.ingest.scanner import ScannedCorpusFile


@dataclass(frozen=True)
class ScanDiff:
    new_paths: set[str]
    deleted_paths: set[str]
    unchanged_paths: set[str]


def diff_scans(previous: list[ScannedCorpusFile], current: list[ScannedCorpusFile]) -> ScanDiff:
    previous_paths = {item.relative_path for item in previous}
    current_paths = {item.relative_path for item in current}
    return ScanDiff(
        new_paths=current_paths - previous_paths,
        deleted_paths=previous_paths - current_paths,
        unchanged_paths=previous_paths & current_paths,
    )
