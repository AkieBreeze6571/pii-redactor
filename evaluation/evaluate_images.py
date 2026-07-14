"""Run real OCR/detection/mapping evaluation on a fixed image manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.error_analysis import draw_error_artifacts
from evaluation.image_metrics import best_coverage, iou, text_similarity
from services.document_processor import DocumentProcessor


def _group_update(group: Counter, truth: int = 0, predicted: int = 0, matched: int = 0) -> None:
    group["truth"] += truth; group["predicted"] += predicted; group["matched"] += matched


def _match_entities(truth: list[dict], predicted: list[dict]) -> tuple[list[tuple[int, int, str]], set[int], set[int]]:
    matches = []; used_truth: set[int] = set(); used_pred: set[int] = set()
    for truth_index, target in enumerate(truth):
        exact = next((index for index, item in enumerate(predicted) if index not in used_pred and item["type"] == target["type"] and item.get("text") == target.get("text")), None)
        if exact is not None:
            matches.append((truth_index, exact, "exact")); used_truth.add(truth_index); used_pred.add(exact)
    for truth_index, target in enumerate(truth):
        if truth_index in used_truth: continue
        partial = next((index for index, item in enumerate(predicted) if index not in used_pred and item["type"] == target["type"] and (item.get("text", "") in target.get("text", "") or target.get("text", "") in item.get("text", ""))), None)
        if partial is not None:
            matches.append((truth_index, partial, "partial")); used_truth.add(truth_index); used_pred.add(partial)
    return matches, used_truth, used_pred


def evaluate_rows(rows: list[dict], processor: DocumentProcessor, artifact_dir: Path) -> tuple[dict, dict, dict, dict, dict, list[dict]]:
    totals = Counter(); by_type = defaultdict(Counter); by_template = defaultdict(Counter); by_source = defaultdict(Counter); mapping = Counter(); errors = []
    ocr_similarity_sum = ocr_cer_chars = ocr_truth_chars = box_iou_sum = 0.0
    block_truth = block_found = complete_boxes = partial_boxes = 0
    source_detection = defaultdict(Counter); confidence_bins = defaultdict(Counter)
    for row in rows:
        result = processor.process_image(row["image_path"]); totals["images"] += 1
        truth_text, ocr_text = row.get("text", ""), result.get("ocr_text", "")
        similarity = text_similarity(truth_text, ocr_text); ocr_similarity_sum += similarity
        ocr_cer_chars += round((1 - similarity) * max(len(truth_text), len(ocr_text), 1)); ocr_truth_chars += max(len(truth_text), 1)
        template, source = row.get("template", "unknown"), row.get("source", "unknown")
        truth, predicted = row.get("entities", []), result.get("entities", [])
        _group_update(by_template[template], len(truth), len(predicted)); _group_update(by_source[source], len(truth), len(predicted))
        for item in truth: by_type[item["type"]]["truth"] += 1
        for item in predicted:
            by_type[item["type"]]["predicted"] += 1; source_detection[item.get("source", "unknown")]["predicted"] += 1
            quality = item.get("mapping_quality", "coarse"); mapping[quality] += 1
            confidence = float(item.get("confidence", 0)); confidence_bins["0.9-1.0" if confidence >= 0.9 else "0.7-0.9" if confidence >= 0.7 else "<0.7"]["predicted"] += 1
        matches, used_truth, used_pred = _match_entities(truth, predicted)
        totals["truth"] += len(truth); totals["predicted"] += len(predicted)
        image_errors = []
        for truth_index, pred_index, match_kind in matches:
            target, item = truth[truth_index], predicted[pred_index]
            totals["matched"] += 1; totals[f"{match_kind}_matches"] += 1
            by_type[target["type"]]["matched"] += 1; by_template[template]["matched"] += 1; by_source[source]["matched"] += 1
            source_detection[item.get("source", "unknown")]["matched"] += 1
            if match_kind == "partial": totals["boundary_errors"] += 1
            predicted_boxes = item.get("boxes", [])
            for truth_box in target.get("boxes", []):
                block_truth += 1
                if any(iou(truth_box, block["polygon"]) > 0.05 for block in result.get("ocr_blocks", [])): block_found += 1
                score = best_coverage(truth_box, predicted_boxes); box_iou_sum += score; totals["boxes"] += 1
                if score >= 0.95: complete_boxes += 1
                elif score > 0: partial_boxes += 1; image_errors.append({"error_type": "partial_coverage", "type": target["type"], "coverage": score})
                else: image_errors.append({"error_type": "mapping_missed", "type": target["type"], "coverage": score})
        for index, target in enumerate(truth):
            if index not in used_truth: image_errors.append({"error_type": "entity_missed", "type": target["type"], "text": target.get("text")})
        for index, item in enumerate(predicted):
            if index not in used_pred: image_errors.append({"error_type": "false_positive", "type": item["type"]})
        if not result.get("ocr_blocks"): image_errors.append({"error_type": "no_ocr_text"})
        if image_errors:
            artifacts = draw_error_artifacts(row, result, artifact_dir) if len({item.get("id") for item in errors}) < 20 else {}
            for error in image_errors: errors.append({"id": row["id"], "template": template, **error, **artifacts})
    precision = totals["matched"] / max(totals["predicted"], 1); recall = totals["matched"] / max(totals["truth"], 1)
    summary = {
        "images": totals["images"], "ocr_text_similarity": ocr_similarity_sum / max(totals["images"], 1),
        "ocr_character_error_rate": ocr_cer_chars / max(ocr_truth_chars, 1), "ocr_block_recall": block_found / max(block_truth, 1),
        "entity_precision": precision, "entity_recall": recall, "entity_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "exact_matches": totals["exact_matches"], "partial_matches": totals["partial_matches"], "boundary_errors": totals["boundary_errors"],
        "type_errors": totals["type_errors"], "mean_box_iou": box_iou_sum / max(totals["boxes"], 1),
        "complete_redaction_rate": complete_boxes / max(totals["boxes"], 1), "partial_coverage_boxes": partial_boxes,
        "missed_entities": totals["truth"] - totals["matched"], "false_redactions": totals["predicted"] - totals["matched"],
        "mapping_quality": dict(mapping), "mapping_quality_rates": {key: value / max(sum(mapping.values()), 1) for key, value in mapping.items()},
        "detection_sources": {key: dict(value) for key, value in source_detection.items()}, "ocr_confidence_bins": {key: dict(value) for key, value in confidence_bins.items()},
    }
    return summary, by_type, by_template, by_source, mapping, errors


def _write_group(path: Path, key_name: str, groups: dict) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle); writer.writerow([key_name, "truth", "predicted", "matched", "precision", "recall", "f1"])
        for key, counts in sorted(groups.items()):
            precision = counts["matched"] / max(counts["predicted"], 1); recall = counts["matched"] / max(counts["truth"], 1); f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            writer.writerow([key, counts["truth"], counts["predicted"], counts["matched"], precision, recall, f1])


def write_outputs(summary: dict, by_type: dict, by_template: dict, by_source: dict, mapping: Counter, errors: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "image_evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_group(output_dir / "image_evaluation_by_type.csv", "type", by_type); _write_group(output_dir / "image_evaluation_by_template.csv", "template", by_template); _write_group(output_dir / "image_evaluation_by_source.csv", "source", by_source)
    with (output_dir / "image_mapping_quality.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["mapping_quality", "count", "rate"]); writer.writerows((key, value, value / max(sum(mapping.values()), 1)) for key, value in sorted(mapping.items()))
    (output_dir / "image_error_cases.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in errors), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--annotations", "--manifest", dest="annotations", type=Path, default=Path("data/annotations/image_test_fixed.jsonl")); parser.add_argument("--limit", type=int); parser.add_argument("--output-dir", "--report-dir", dest="output_dir", type=Path, default=Path("reports")); args = parser.parse_args()
    rows = [json.loads(line) for line in args.annotations.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None: rows = rows[:args.limit]
    summary, by_type, by_template, by_source, mapping, errors = evaluate_rows(rows, DocumentProcessor(), args.output_dir / "image_errors")
    write_outputs(summary, by_type, by_template, by_source, mapping, errors, args.output_dir); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
