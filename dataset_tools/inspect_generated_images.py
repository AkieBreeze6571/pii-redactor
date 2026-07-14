"""Validate generated images and exact character/box annotations."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def main() -> int:
    annotation_files = sorted(Path("data/generated").glob("annotations_*.jsonl"))
    rows = [json.loads(line) for path in annotation_files for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    templates = Counter(); types = Counter(); errors = []; annotated_paths = set(); cross_line = negative = empty_boxes = 0
    for row in rows:
        templates[row.get("template", "unknown")] += 1
        image_path = Path(row["image"]); annotated_paths.add(image_path.as_posix())
        if not image_path.exists():
            errors.append({"id": row.get("id"), "error": "missing_image", "path": image_path.as_posix()}); continue
        try:
            with Image.open(image_path) as image: width, height = image.size
        except OSError as exc:
            errors.append({"id": row.get("id"), "error": "unreadable_image", "detail": str(exc)}); continue
        if not row.get("entities"): negative += 1
        for entity in row.get("entities", []):
            types[entity["type"]] += 1
            if row["text"][entity["start"]:entity["end"]] != entity["text"]:
                errors.append({"id": row.get("id"), "error": "span_mismatch", "entity": entity})
            boxes = entity.get("boxes", [])
            if not boxes: empty_boxes += 1; errors.append({"id": row.get("id"), "error": "empty_boxes", "entity": entity})
            if len(boxes) > 1: cross_line += 1
            for box in boxes:
                if len(box) != 4 or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                    errors.append({"id": row.get("id"), "error": "box_out_of_bounds", "box": box, "size": [width, height]})
    image_paths = {path.as_posix() for path in Path("data/generated/images").rglob("*.png")}
    for path in sorted(image_paths - annotated_paths): errors.append({"error": "image_without_annotation", "path": path})
    summary = {
        "images": len(image_paths), "annotations": len(rows), "templates": len(templates),
        "template_distribution": dict(sorted(templates.items())), "entity_type_distribution": dict(sorted(types.items())),
        "cross_line_entities": cross_line, "negative_samples": negative, "empty_boxes": empty_boxes,
        "missing_annotation_images": len(image_paths - annotated_paths), "missing_images": sum(item["error"] == "missing_image" for item in errors),
        "invalid_annotations": len(errors),
    }
    reports = Path("reports"); reports.mkdir(exist_ok=True)
    (reports / "generated_image_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, name, counts in (("generated_image_by_template.csv", "template", templates), ("generated_image_by_type.csv", "entity_type", types)):
        with (reports / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle); writer.writerow([name, "count"]); writer.writerows(sorted(counts.items()))
    (reports / "generated_image_errors.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in errors), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)); return 1 if errors else 0


if __name__ == "__main__": raise SystemExit(main())
