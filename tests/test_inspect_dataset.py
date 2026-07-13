import json
from pathlib import Path

from dataset_tools.inspect_dataset import discover_files, inspect_file


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_discover_files_skips_download_cache(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    cached = tmp_path / ".cache" / "source.jsonl"
    cached.parent.mkdir()
    source.write_text("{}\n", encoding="utf-8")
    cached.write_text("{}\n", encoding="utf-8")

    assert discover_files(tmp_path) == [source]


def test_inspection_counts_invalid_duplicate_and_overlap(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "sample",
                "text": "AliceParis",
                "entities": [
                    {"type": "person", "text": "Alice", "start": 0, "end": 5},
                    {"type": "person", "text": "Alice", "start": 0, "end": 5},
                    {"type": "address", "text": "iceP", "start": 2, "end": 6},
                    {"type": "person", "text": "Bob", "start": 0, "end": 3},
                ],
            }
        ],
    )

    stats = inspect_file(path, tmp_path)

    assert stats.records == 1
    assert stats.entities == 4
    assert stats.invalid_entities == 1
    assert stats.duplicate_entities == 1
    assert stats.overlapping_entity_pairs == 3
    assert stats.entity_type_counts == {"person": 3, "address": 1}

