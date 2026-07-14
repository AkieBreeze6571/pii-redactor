"""Run privacy-safe local regression diagnostics for OCR coordinate mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.document_processor import DocumentProcessor


COLORS = {
    "weighted": (255, 165, 0),
    "projection": (0, 120, 255),
    "component": (40, 180, 80),
    "final": (220, 30, 30),
}


def _draw(image: Image.Image, polygons: list[list[list[float]]], color: tuple[int, int, int], path: Path) -> None:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for polygon in polygons:
        points = [(round(point[0]), round(point[1])) for point in polygon]
        if len(points) >= 3:
            draw.line(points + [points[0]], fill=color, width=3)
    output.save(path)


def _step_polygons(entities: list[dict[str, Any]], step: str) -> list[list[list[float]]]:
    output = []
    for entity in entities:
        for part in entity.get("mapping_steps", []):
            polygon = part.get(step)
            if polygon:
                output.append(polygon)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", choices=("strict", "balanced"), default="strict")
    parser.add_argument("--output-dir", type=Path, default=Path("data/local_regression/output"))
    args = parser.parse_args()
    if not args.input.is_file():
        print(json.dumps({"error": "local regression input does not exist"}, ensure_ascii=False))
        return 2

    processor = DocumentProcessor()
    processor.mapper.safety_mode = args.mode
    result = processor.process_image(args.input, redaction_mode="black")
    if not result.get("result_path"):
        print(json.dumps({"error": "document processing did not produce a result"}, ensure_ascii=False))
        return 3

    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = Image.open(args.input).convert("RGB")
    block_polygons = [block["polygon"] for block in result.get("ocr_blocks", [])]
    _draw(original, block_polygons, (120, 120, 120), args.output_dir / "ocr_blocks.png")
    for step, filename in (
        ("weighted", "initial_weighted.png"),
        ("projection", "projection_refined.png"),
        ("component", "component_refined.png"),
        ("final", "final_boxes.png"),
    ):
        _draw(original, _step_polygons(result.get("entities", []), step), COLORS[step], args.output_dir / filename)
    redacted_path = args.output_dir / "redacted.png"
    shutil.copy2(result["result_path"], redacted_path)

    sanitized_entities = []
    for entity in result.get("entities", []):
        sanitized_entities.append({
            "type": entity.get("type"),
            "entity_text_length": len(str(entity.get("text", ""))),
            "detector_confidence": entity.get("confidence"),
            "mapping_quality": entity.get("mapping_quality"),
            "mapping_confidence": entity.get("mapping_confidence"),
            "mapping_strategy": entity.get("mapping_strategy"),
            "fallback_reason": entity.get("fallback_reason"),
            "final_boxes": entity.get("boxes", []),
            "mapping_steps": entity.get("mapping_steps", []),
        })
    input_id = hashlib.sha256(args.input.read_bytes()).hexdigest()[:12]
    report = {
        "input_id": input_id,
        "safety_mode": args.mode,
        "ocr_blocks": [
            {"block_index": block.get("block_index"), "text_length": len(str(block.get("text", ""))), "confidence": block.get("confidence"), "polygon": block.get("polygon")}
            for block in result.get("ocr_blocks", [])
        ],
        "entities": sanitized_entities,
        "warnings": result.get("warnings", []),
        "outputs": {
            "ocr_blocks": str(args.output_dir / "ocr_blocks.png"),
            "initial_weighted": str(args.output_dir / "initial_weighted.png"),
            "projection_refined": str(args.output_dir / "projection_refined.png"),
            "component_refined": str(args.output_dir / "component_refined.png"),
            "final_boxes": str(args.output_dir / "final_boxes.png"),
            "redacted": str(redacted_path),
        },
    }
    report_path = Path("data/local_regression/report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "entity_count": len(sanitized_entities), "mode": args.mode}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
