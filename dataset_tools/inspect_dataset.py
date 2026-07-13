"""Inspect local JSON/JSONL datasets without modifying source data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO


SUPPORTED_SUFFIXES = {".json", ".jsonl"}
SKIPPED_PARTS = {".cache", "__pycache__"}


@dataclass
class FileStats:
    path: str
    format: str
    records: int = 0
    malformed_records: int = 0
    field_counts: Counter[str] = field(default_factory=Counter)
    entity_type_counts: Counter[str] = field(default_factory=Counter)
    entities: int = 0
    invalid_entities: int = 0
    duplicate_entities: int = 0
    overlapping_entity_pairs: int = 0
    first_sample: Any = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["field_counts"] = dict(sorted(self.field_counts.items()))
        result["entity_type_counts"] = dict(
            sorted(self.entity_type_counts.items())
        )
        return result


def discover_files(raw_dir: Path) -> list[Path]:
    """Return source JSON files while excluding hidden download caches."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not any(part in SKIPPED_PARTS for part in path.parts)
    )


def _iter_jsonl(handle: TextIO, stats: FileStats) -> Iterator[tuple[int, Any]]:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            yield line_number, json.loads(line)
        except json.JSONDecodeError as exc:
            stats.malformed_records += 1
            stats.errors.append(f"line {line_number}: {exc.msg}")


def _iter_json(handle: TextIO, stats: FileStats) -> Iterator[tuple[int, Any]]:
    try:
        value = json.load(handle)
    except json.JSONDecodeError as exc:
        stats.malformed_records += 1
        stats.errors.append(f"JSON decode error at line {exc.lineno}: {exc.msg}")
        return
    if isinstance(value, list):
        yield from enumerate(value, start=1)
    else:
        yield 1, value


def iter_records(path: Path, stats: FileStats) -> Iterator[tuple[int, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        if path.suffix.lower() == ".jsonl":
            yield from _iter_jsonl(handle, stats)
        else:
            yield from _iter_json(handle, stats)


def _valid_entity(entity: Any, text: Any) -> bool:
    if not isinstance(entity, dict) or not isinstance(text, str):
        return False
    start, end, entity_text = (
        entity.get("start"),
        entity.get("end"),
        entity.get("text"),
    )
    return (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and isinstance(entity_text, str)
        and 0 <= start < end <= len(text)
        and text[start:end] == entity_text
    )


def inspect_record(record: Any, stats: FileStats, record_number: int) -> None:
    stats.records += 1
    if stats.first_sample is None:
        stats.first_sample = record
    if not isinstance(record, dict):
        stats.malformed_records += 1
        stats.errors.append(f"record {record_number}: expected object")
        return

    stats.field_counts.update(record.keys())
    text = record.get("text")
    entities = record.get("entities", [])
    if entities is None:
        entities = []
    if not isinstance(entities, list):
        stats.malformed_records += 1
        stats.errors.append(f"record {record_number}: entities is not a list")
        return

    seen: set[tuple[Any, Any, Any, Any]] = set()
    valid_spans: list[tuple[int, int]] = []
    for index, entity in enumerate(entities):
        stats.entities += 1
        entity_type = entity.get("type") if isinstance(entity, dict) else None
        stats.entity_type_counts[str(entity_type or "<missing>")] += 1
        if not _valid_entity(entity, text):
            stats.invalid_entities += 1
            if len(stats.errors) < 100:
                stats.errors.append(
                    f"record {record_number}, entity {index}: invalid start/end or text"
                )
            continue

        key = (entity_type, entity["start"], entity["end"], entity["text"])
        if key in seen:
            stats.duplicate_entities += 1
        seen.add(key)
        valid_spans.append((entity["start"], entity["end"]))

    valid_spans.sort()
    for current_index, (start, _end) in enumerate(valid_spans):
        for previous_start, previous_end in valid_spans[:current_index]:
            if previous_end > start and previous_start < _end:
                stats.overlapping_entity_pairs += 1


def inspect_file(path: Path, root: Path) -> FileStats:
    stats = FileStats(path=path.relative_to(root).as_posix(), format=path.suffix[1:])
    try:
        for record_number, record in iter_records(path, stats):
            inspect_record(record, stats, record_number)
    except (OSError, UnicodeError) as exc:
        stats.malformed_records += 1
        stats.errors.append(f"read error: {exc}")
    return stats


def build_summary(raw_dir: Path, file_stats: list[FileStats]) -> dict[str, Any]:
    entity_types: Counter[str] = Counter()
    fields: Counter[str] = Counter()
    for stats in file_stats:
        entity_types.update(stats.entity_type_counts)
        fields.update(stats.field_counts)
    return {
        "raw_directory": raw_dir.as_posix(),
        "files_scanned": len(file_stats),
        "totals": {
            "records": sum(item.records for item in file_stats),
            "malformed_records": sum(item.malformed_records for item in file_stats),
            "entities": sum(item.entities for item in file_stats),
            "invalid_entities": sum(item.invalid_entities for item in file_stats),
            "duplicate_entities": sum(item.duplicate_entities for item in file_stats),
            "overlapping_entity_pairs": sum(
                item.overlapping_entity_pairs for item in file_stats
            ),
        },
        "field_counts": dict(sorted(fields.items())),
        "entity_type_counts": dict(sorted(entity_types.items())),
        "files": [item.to_dict() for item in file_stats],
    }


def write_reports(
    summary: dict[str, Any], file_stats: list[FileStats], report_dir: Path
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    csv_path = report_dir / "dataset_by_type.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "entity_type", "count"])
        writer.writeheader()
        for stats in file_stats:
            for entity_type, count in sorted(stats.entity_type_counts.items()):
                writer.writerow(
                    {"path": stats.path, "entity_type": entity_type, "count": count}
                )


def print_results(file_stats: list[FileStats], summary: dict[str, Any]) -> None:
    for stats in file_stats:
        print("=" * 72)
        print(stats.path)
        print("First sample:")
        print(json.dumps(stats.first_sample, ensure_ascii=False, indent=2))
        print(f"Fields: {dict(sorted(stats.field_counts.items()))}")
        print(f"Entity types: {dict(sorted(stats.entity_type_counts.items()))}")
        print(
            "Counts: "
            f"records={stats.records}, entities={stats.entities}, "
            f"invalid={stats.invalid_entities}, duplicates={stats.duplicate_entities}, "
            f"overlaps={stats.overlapping_entity_pairs}, "
            f"malformed={stats.malformed_records}"
        )
    print("=" * 72)
    print("Overall summary:")
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2))
    print(f"Entity types: {summary['entity_type_counts']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    try:
        files = discover_files(args.raw_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2
    if not files:
        print(f"ERROR: no JSON or JSONL files found under {args.raw_dir}")
        return 2

    file_stats = [inspect_file(path, args.raw_dir) for path in files]
    summary = build_summary(args.raw_dir, file_stats)
    write_reports(summary, file_stats, args.report_dir)
    print_results(file_stats, summary)
    print(f"Reports written to {args.report_dir.resolve()}")
    return 1 if summary["totals"]["malformed_records"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
