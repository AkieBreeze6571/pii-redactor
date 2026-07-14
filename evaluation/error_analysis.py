"""Render bounded image-evaluation error artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from evaluation.image_metrics import rectangle


def draw_error_artifacts(row: dict[str, Any], result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(row["image_path"]) as source:
        original = source.convert("RGB")
    truth = original.copy(); truth_draw = ImageDraw.Draw(truth)
    for entity in row.get("entities", []):
        for box in entity.get("boxes", []): truth_draw.rectangle(rectangle(box), outline=(0, 180, 0), width=3)
    prediction = original.copy(); prediction_draw = ImageDraw.Draw(prediction)
    for entity in result.get("entities", []):
        for box in entity.get("boxes", []): prediction_draw.rectangle(rectangle(box), outline=(220, 20, 20), width=3)
    stem = row["id"]
    original_path = output_dir / f"{stem}_original.png"; truth_path = output_dir / f"{stem}_truth.png"; prediction_path = output_dir / f"{stem}_prediction.png"
    original.save(original_path); truth.save(truth_path); prediction.save(prediction_path)
    return {"original_artifact": str(original_path), "truth_artifact": str(truth_path), "prediction_artifact": str(prediction_path), "redacted_artifact": result.get("result_path")}
