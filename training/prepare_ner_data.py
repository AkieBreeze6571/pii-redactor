"""Convert PII Bench ZH into validated, deduplicated, frozen NER splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SPLITS = ("train", "validation", "test_fixed")


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    source: str


def sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_fingerprint(text: str, normalize_outer_whitespace: bool = True) -> str:
    comparable = text.strip() if normalize_outer_whitespace else text
    return hashlib.sha256(comparable.encode("utf-8")).hexdigest()


def first_present(data: dict[str, Any], names: list[str]) -> tuple[str | None, Any]:
    for name in names:
        if name in data:
            return name, data[name]
    return None, None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def discover_sources(config: dict[str, Any]) -> list[SourceSpec]:
    raw = config["raw"]
    root = Path(raw["search_root"])
    sources: list[SourceSpec] = []
    seen: set[Path] = set()
    for kind, source in (("formal_patterns", "pii_bench_zh_formal"), ("chat_patterns", "pii_bench_zh_chat")):
        for pattern in raw[kind]:
            for path in sorted(root.glob(pattern)):
                resolved = path.resolve()
                if path.is_file() and resolved not in seen and ".cache" not in path.parts:
                    sources.append(SourceSpec(path=path, source=source))
                    seen.add(resolved)
    if not sources:
        raise FileNotFoundError(f"No PII Bench ZH files found below {root}")
    return sources


def load_label_map(path: Path) -> tuple[dict[str, str], set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    aliases = data.get("aliases", {})
    targets = set(data.get("target_labels", []))
    if not aliases or not targets:
        raise ValueError(f"Invalid label map: {path}")
    return aliases, targets


def stable_id(source: str, raw_id: Any, fingerprint: str) -> str:
    if raw_id is None or not str(raw_id).strip():
        suffix = fingerprint[:16]
    else:
        suffix = str(raw_id).strip()
        for prefix in ("zh_chat_", "zh_"):
            if suffix.startswith(prefix):
                suffix = suffix[len(prefix):]
                break
    return f"{source}_{suffix}"


def entity_signature(record: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (entity["type"], entity["start"], entity["end"], entity["text"])
        for entity in record["entities"]
    )


def count_overlaps(entities: list[dict[str, Any]]) -> int:
    spans = sorted((entity["start"], entity["end"]) for entity in entities)
    return sum(
        left_start < right_end and right_start < left_end
        for index, (right_start, right_end) in enumerate(spans)
        for left_start, left_end in spans[:index]
    )


def convert_record(
    raw_record: dict[str, Any],
    source: SourceSpec,
    row_number: int,
    config: dict[str, Any],
    aliases: dict[str, str],
    targets: set[str],
    issues: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    schema = config["raw"]["schema"]
    _, text = first_present(raw_record, schema["text_fields"])
    _, raw_entities = first_present(raw_record, schema["entities_fields"])
    _, raw_id = first_present(raw_record, schema["id_fields"])
    location = {"source_file": source.path.as_posix(), "row": row_number, "raw_id": raw_id}
    if not isinstance(text, str) or not text:
        issues["invalid"].append({**location, "reason": "empty or missing text"})
        return None
    if raw_entities is None:
        raw_entities = []
    if not isinstance(raw_entities, list):
        issues["invalid"].append({**location, "reason": "entities is not a list"})
        return None

    policy = config["labels"]["unknown_label_policy"]
    entities: list[dict[str, Any]] = []
    for entity_index, raw_entity in enumerate(raw_entities):
        if not isinstance(raw_entity, dict):
            issues["invalid"].append({**location, "entity_index": entity_index, "reason": "entity is not an object"})
            continue
        _, raw_type = first_present(raw_entity, schema["entity_type_fields"])
        _, entity_text = first_present(raw_entity, schema["entity_text_fields"])
        _, start = first_present(raw_entity, schema["start_fields"])
        _, end = first_present(raw_entity, schema["end_fields"])
        mapped_type = aliases.get(str(raw_type))
        if mapped_type is None:
            unknown = {**location, "entity_index": entity_index, "unknown_label": raw_type}
            issues["unknown"].append(unknown)
            if policy == "error":
                raise ValueError(f"Unknown label {raw_type!r} at {source.path}:{row_number}")
            if policy == "skip":
                continue
            mapped_type = str(raw_type).strip().lower()
        if mapped_type not in targets:
            issues["unknown"].append({**location, "entity_index": entity_index, "unknown_label": raw_type, "retained_as": mapped_type})
        valid = (
            isinstance(start, int) and not isinstance(start, bool)
            and isinstance(end, int) and not isinstance(end, bool)
            and isinstance(entity_text, str)
            and 0 <= start < end <= len(text)
            and text[start:end] == entity_text
        )
        if not valid:
            issues["invalid"].append({
                **location, "entity_index": entity_index, "reason": "invalid span or entity text",
                "entity": raw_entity,
            })
            continue
        entities.append({"type": mapped_type, "text": entity_text, "start": start, "end": end})

    entities.sort(key=lambda item: (item["start"], item["end"], item["type"]))
    fingerprint = text_fingerprint(text, config["deduplication"]["normalize_outer_whitespace"])
    return {
        "id": stable_id(source.source, raw_id, fingerprint),
        "source": source.source,
        "license": config["raw"].get("license", "unknown"),
        "split": "",
        "text": text,
        "entities": entities,
        "pseudo_label": False,
        "text_sha256": fingerprint,
        "template_sha256": template_fingerprint(text, entities),
    }


def template_fingerprint(text: str, entities: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    cursor = 0
    for entity in sorted(entities, key=lambda item: (item["start"], item["end"])):
        if entity["start"] < cursor:
            continue
        parts.append(text[cursor:entity["start"]])
        parts.append(f"<{entity['type'].upper()}>")
        cursor = entity["end"]
    parts.append(text[cursor:])
    normalized = "".join(parts).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate(
    records: list[dict[str, Any]], conflicts: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int, int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["text_sha256"]].append(record)
    selected = []
    conflict_count = 0
    for fingerprint, candidates in groups.items():
        candidates.sort(key=lambda item: (-len(item["entities"]), item["source"], item["id"]))
        signatures = {entity_signature(item) for item in candidates}
        if len(signatures) > 1:
            conflict_count += 1
            conflicts.append({
                "text_sha256": fingerprint,
                "sample_ids": [item["id"] for item in candidates],
                "entity_signatures": [[list(value) for value in entity_signature(item)] for item in candidates],
            })
        selected.append(candidates[0])
    selected.sort(key=lambda item: (item["source"], item["id"]))
    return selected, len(records) - len(selected), conflict_count


def _group_records(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["template_sha256"]].append(record)
    return list(groups.values())


def split_records(
    records: list[dict[str, Any]], ratios: dict[str, float], seed: int
) -> dict[str, list[dict[str, Any]]]:
    groups = _group_records(records)
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    targets = {split: len(records) * ratio for split, ratio in ratios.items()}
    result = {split: [] for split in SPLITS}
    eligible_splits = [split for split in SPLITS if ratios[split] > 0]
    if not eligible_splits:
        raise ValueError("At least one split ratio must be greater than zero")
    source_totals = Counter(record["source"] for record in records)
    source_counts = {split: Counter() for split in SPLITS}
    label_totals = Counter(
        entity["type"] for record in records for entity in record["entities"]
    )
    label_counts = {split: Counter() for split in SPLITS}
    for group in groups:
        group_sources = Counter(record["source"] for record in group)
        group_labels = Counter(
            entity["type"] for record in group for entity in record["entities"]
        )
        def score(split: str) -> tuple[float, float, int]:
            size_pressure = (len(result[split]) + len(group)) / max(targets[split], 1)
            source_pressure = sum(
                (source_counts[split][source] + count)
                / max(source_totals[source] * ratios[split], 1)
                for source, count in group_sources.items()
            )
            source_pressure /= max(len(group_sources), 1)
            label_pressure = sum(
                (label_counts[split][label] + count)
                / max(label_totals[label] * ratios[split], 1)
                for label, count in group_labels.items()
            ) / max(len(group_labels), 1)
            return (
                size_pressure + 0.3 * source_pressure + 0.7 * label_pressure,
                size_pressure,
                SPLITS.index(split),
            )
        chosen = min(eligible_splits, key=score)
        result[chosen].extend(group)
        source_counts[chosen].update(group_sources)
        label_counts[chosen].update(group_labels)
    return result


def assign_splits(
    records: list[dict[str, Any]], output_dir: Path, config: dict[str, Any], force_resplit: bool
) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    processed = config["processed"]
    ratios = {
        "train": float(processed["train_ratio"]),
        "validation": float(processed["validation_ratio"]),
        "test_fixed": float(processed["test_ratio"]),
    }
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    test_path = output_dir / "test_fixed.jsonl"
    frozen = test_path.exists() and processed.get("freeze_test", True) and not force_resplit
    if frozen:
        fixed = load_jsonl(test_path)
        fixed_fingerprints = {item["text_sha256"] for item in fixed}
        current = {item["text_sha256"] for item in records}
        missing = fixed_fingerprints - current
        if missing:
            raise ValueError(f"Frozen test set contains {len(missing)} records absent from current sources")
        remaining = [item for item in records if item["text_sha256"] not in fixed_fingerprints]
        train_ratio = ratios["train"] / (ratios["train"] + ratios["validation"])
        partial = split_records(
            remaining,
            {"train": train_ratio, "validation": 1 - train_ratio, "test_fixed": 0.0},
            int(processed["seed"]),
        )
        result = {"train": partial["train"], "validation": partial["validation"], "test_fixed": fixed}
    else:
        result = split_records(records, ratios, int(processed["seed"]))
    for split, items in result.items():
        for item in items:
            item["split"] = split
        items.sort(key=lambda item: item["id"])
    return result, frozen


def _split_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(entity["type"] for item in items for entity in item["entities"])
    sources = Counter(item["source"] for item in items)
    lengths = [len(item["text"]) for item in items]
    return {
        "samples": len(items),
        "entities": sum(labels.values()),
        "labels": dict(sorted(labels.items())),
        "sources": dict(sorted(sources.items())),
        "no_entity_samples": sum(not item["entities"] for item in items),
        "multi_entity_samples": sum(len(item["entities"]) > 1 for item in items),
        "average_text_length": round(sum(lengths) / len(lengths), 3) if lengths else 0,
        "max_text_length": max(lengths, default=0),
        "overlapping_entity_pairs": sum(count_overlaps(item["entities"]) for item in items),
    }


def write_csv_reports(report_dir: Path, summary: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "processed_dataset_by_type.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["split", "entity_type", "count"])
        for split in SPLITS:
            for label, count in summary["splits"][split]["labels"].items(): writer.writerow([split, label, count])
    with (report_dir / "processed_dataset_by_source.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["split", "source", "sample_count"])
        for split in SPLITS:
            for source, count in summary["splits"][split]["sources"].items(): writer.writerow([split, source, count])
    with (report_dir / "split_distribution.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["split", "samples", "entities", "no_entity_samples", "multi_entity_samples", "average_text_length", "max_text_length", "overlapping_entity_pairs"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for split in SPLITS: writer.writerow({"split": split, **{key: summary["splits"][split][key] for key in fields[1:]}})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    parser.add_argument("--formal", type=Path, help="Override the formal JSONL input")
    parser.add_argument("--chat", type=Path, help="Override the chat JSONL input")
    parser.add_argument("--output-dir", type=Path, help="Override processed output directory")
    parser.add_argument("--force-resplit", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config["processed"]["output_dir"])
    report_dir = Path(config["reports"]["output_dir"])
    aliases, targets = load_label_map(Path(config["labels"]["map_path"]))
    sources = discover_sources(config)
    if args.formal or args.chat:
        sources = [
            SourceSpec(path=path, source=source)
            for path, source in ((args.formal, "pii_bench_zh_formal"), (args.chat, "pii_bench_zh_chat"))
            if path is not None
        ]
    if args.force_resplit:
        print("WARNING: --force-resplit requested; the frozen test_fixed split will be replaced.", file=sys.stderr)

    issues: dict[str, list[dict[str, Any]]] = {"invalid": [], "unknown": [], "conflicts": []}
    converted = []
    raw_count = 0
    for source in sources:
        raw_records = load_jsonl(source.path); raw_count += len(raw_records)
        for row_number, raw_record in enumerate(raw_records, start=1):
            record = convert_record(raw_record, source, row_number, config, aliases, targets, issues)
            if record is not None: converted.append(record)
    if config["deduplication"].get("enabled", True):
        converted, duplicate_count, conflict_count = deduplicate(converted, issues["conflicts"])
    else:
        duplicate_count = conflict_count = 0

    splits, reused_frozen_test = assign_splits(converted, output_dir, config, args.force_resplit)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        if split == "test_fixed" and reused_frozen_test:
            continue
        write_jsonl(path, splits[split])
    write_jsonl(report_dir / "invalid_annotations.jsonl", issues["invalid"] + issues["unknown"])
    write_jsonl(report_dir / "conflicting_duplicates.jsonl", issues["conflicts"])

    split_stats = {split: _split_stats(splits[split]) for split in SPLITS}
    summary = {
        "source_files": [source.path.as_posix() for source in sources],
        "raw_samples": raw_count,
        "converted_samples": sum(len(items) for items in splits.values()),
        "deduplicated_samples": len(converted),
        "duplicate_samples": duplicate_count,
        "conflicting_duplicate_groups": conflict_count,
        "invalid_annotations": len(issues["invalid"]),
        "unknown_labels": len(issues["unknown"]),
        "frozen_test_reused": reused_frozen_test,
        "splits": split_stats,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "processed_dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv_reports(report_dir, summary)
    manifest = {
        "seed": int(config["processed"]["seed"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_files": summary["source_files"],
        "source_file_hashes": {source.path.as_posix(): sha256_bytes(source.path) for source in sources},
        "train_count": len(splits["train"]),
        "validation_count": len(splits["validation"]),
        "test_fixed_count": len(splits["test_fixed"]),
        "label_distribution": {split: split_stats[split]["labels"] for split in SPLITS},
        "source_distribution": {split: split_stats[split]["sources"] for split in SPLITS},
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_files": summary["source_files"], "raw_samples": raw_count,
        "converted_samples": summary["converted_samples"], "duplicates": duplicate_count,
        "invalid_annotations": summary["invalid_annotations"],
        "split_counts": {split: len(splits[split]) for split in SPLITS},
        "frozen_test_reused": reused_frozen_test,
    }, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
