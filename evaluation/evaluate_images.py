"""Evaluate OCR, entity detection, coordinate mapping, and coverage on images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.document_processor import DocumentProcessor


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, a in enumerate(left, 1):
        current = [row]
        for column, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (a != b)))
        previous = current
    return previous[-1]


def rectangle(box: list[Any]) -> tuple[float, float, float, float]:
    if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
        return tuple(float(value) for value in box)
    xs = [float(point[0]) for point in box]; ys = [float(point[1]) for point in box]
    return min(xs), min(ys), max(xs), max(ys)


def iou(first: list[Any], second: list[Any]) -> float:
    ax1, ay1, ax2, ay2 = rectangle(first); bx1, by1, bx2, by2 = rectangle(second)
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - intersection
    return intersection / union if union else 0.0


def coverage(truth: list[Any], predicted: list[list[Any]]) -> float:
    return max((iou(truth, box) for box in predicted), default=0.0)


def evaluate_rows(rows: list[dict], processor: DocumentProcessor) -> tuple[dict, dict, dict, list[dict]]:
    totals = Counter(images=0, truth=0, predicted=0, matched=0, coarse=0, boxes=0, complete=0)
    cer_numerator = cer_denominator = iou_sum = 0.0
    by_type = defaultdict(Counter); by_source = defaultdict(Counter); errors = []
    for row in rows:
        result = processor.process_image(row["image_path"])
        totals["images"] += 1
        cer_numerator += levenshtein(row.get("text", ""), result.get("ocr_text", "")); cer_denominator += max(len(row.get("text", "")), 1)
        truth_entities = row.get("entities", []); predicted = result.get("entities", [])
        totals["truth"] += len(truth_entities); totals["predicted"] += len(predicted)
        source = row.get("source", "unknown"); by_source[source]["truth"] += len(truth_entities); by_source[source]["predicted"] += len(predicted)
        for item in predicted: by_type[item["type"]]["predicted"] += 1
        used: set[int] = set()
        for truth in truth_entities:
            entity_type = truth["type"]; by_type[entity_type]["truth"] += 1
            match_index = next((index for index, item in enumerate(predicted) if index not in used and item["type"] == entity_type and item.get("text") == truth.get("text")), None)
            if match_index is None:
                errors.append({"id": row["id"], "error_type": "entity_missed", "type": entity_type, "text": truth.get("text")}); continue
            used.add(match_index); match = predicted[match_index]; totals["matched"] += 1; by_type[entity_type]["matched"] += 1; by_source[source]["matched"] += 1
            if match.get("mapping_quality") == "coarse": totals["coarse"] += 1
            predicted_boxes = match.get("boxes", [])
            for truth_box in truth.get("boxes", []):
                score = coverage(truth_box, predicted_boxes); iou_sum += score; totals["boxes"] += 1
                if score >= 0.95: totals["complete"] += 1
                else: errors.append({"id": row["id"], "error_type": "incomplete_coverage", "type": entity_type, "coverage": score})
        for index, item in enumerate(predicted):
            if index not in used: errors.append({"id": row["id"], "error_type": "false_positive", "type": item["type"]})
    precision = totals["matched"] / max(totals["predicted"], 1); recall = totals["matched"] / max(totals["truth"], 1)
    summary = {
        "images": totals["images"], "ocr_character_accuracy": max(0.0, 1 - cer_numerator / max(cer_denominator, 1)),
        "entity_precision": precision, "entity_recall": recall, "entity_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "mean_box_iou": iou_sum / max(totals["boxes"], 1), "complete_redaction_rate": totals["complete"] / max(totals["boxes"], 1),
        "coarse_mapping_rate": totals["coarse"] / max(totals["matched"], 1), "missed": totals["truth"] - totals["matched"],
        "false_positives": totals["predicted"] - totals["matched"],
    }
    return summary, by_type, by_source, errors


def write_outputs(summary: dict, by_type: dict, by_source: dict, errors: list[dict], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "image_evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, groups, key_name in (("image_evaluation_by_type.csv", by_type, "type"), ("image_evaluation_by_source.csv", by_source, "source")):
        with (report_dir / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle); writer.writerow([key_name, "truth", "predicted", "matched", "precision", "recall"])
            for key, counts in sorted(groups.items()):
                writer.writerow([key, counts["truth"], counts["predicted"], counts["matched"], counts["matched"] / max(counts["predicted"], 1), counts["matched"] / max(counts["truth"], 1)])
    (report_dir / "image_error_cases.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in errors), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--manifest", type=Path, default=Path("data/annotations/image_test.jsonl")); parser.add_argument("--limit", type=int); parser.add_argument("--report-dir", type=Path, default=Path("reports")); args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None: rows = rows[:args.limit]
    summary, by_type, by_source, errors = evaluate_rows(rows, DocumentProcessor()); write_outputs(summary, by_type, by_source, errors, args.report_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
