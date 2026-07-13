"""Non-destructive black, mosaic, and blur redaction outputs."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

from services.image_service import load_image


def safe_stem(name: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", Path(name).stem).strip(" ._")
    return value[:80] or "document"


def clip_polygon(polygon: list[list[float]], size: tuple[int, int]) -> list[tuple[int, int]]:
    width, height = size
    return [(round(max(0, min(width - 1, x))), round(max(0, min(height - 1, y)))) for x, y in polygon]


class RedactionService:
    def __init__(self, output_dir: str | Path = "data/outputs") -> None:
        self.output_dir = Path(output_dir)

    def redact(
        self,
        image: Any,
        entities: list[dict[str, Any]],
        mode: str = "black",
        filename: str = "document.png",
        black_color: tuple[int, int, int] = (0, 0, 0),
        mosaic_block_size: int = 10,
        blur_kernel_size: int = 31,
    ) -> dict[str, str]:
        if mode not in {"black", "mosaic", "blur"}:
            raise ValueError(f"不支持的打码模式：{mode}")
        original = load_image(image).convert("RGB")
        result = original.copy()
        mask = Image.new("L", original.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        polygons = [clip_polygon(polygon, original.size) for entity in entities if entity.get("enabled", True) for polygon in entity.get("boxes", []) if len(polygon) >= 3]
        for polygon in polygons:
            mask_draw.polygon(polygon, fill=255)
            if mode == "black":
                ImageDraw.Draw(result).polygon(polygon, fill=black_color)
            else:
                self._filter_region(result, polygon, mode, mosaic_block_size, blur_kernel_size)
        preview = original.copy(); preview_draw = ImageDraw.Draw(preview)
        for entity in entities:
            if not entity.get("enabled", True): continue
            for polygon in entity.get("boxes", []):
                points = clip_polygon(polygon, original.size)
                if len(points) >= 3:
                    preview_draw.line(points + [points[0]], fill=(220, 30, 30), width=3)
                    preview_draw.text(points[0], str(entity.get("type", "entity")), fill=(220, 30, 30))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex[:10]; stem = safe_stem(filename)
        result_path = self.output_dir / f"{stem}_{token}_{mode}.png"
        preview_path = self.output_dir / f"{stem}_{token}_preview.png"
        mask_path = self.output_dir / f"{stem}_{token}_mask.png"
        result.save(result_path); preview.save(preview_path); mask.save(mask_path)
        return {"result_path": str(result_path), "preview_path": str(preview_path), "mask_path": str(mask_path)}

    @staticmethod
    def _filter_region(image: Image.Image, polygon: list[tuple[int, int]], mode: str, block_size: int, kernel_size: int) -> None:
        xs, ys = zip(*polygon); left, top, right, bottom = max(0, min(xs)), max(0, min(ys)), min(image.width, max(xs) + 1), min(image.height, max(ys) + 1)
        if right <= left or bottom <= top: return
        crop = image.crop((left, top, right, bottom))
        local_mask = Image.new("L", crop.size, 0)
        ImageDraw.Draw(local_mask).polygon([(x - left, y - top) for x, y in polygon], fill=255)
        if mode == "mosaic":
            block = max(1, min(int(block_size), crop.width, crop.height))
            filtered = crop.resize((max(1, crop.width // block), max(1, crop.height // block)), Image.Resampling.BOX).resize(crop.size, Image.Resampling.NEAREST)
        else:
            kernel = max(1, int(kernel_size)); kernel -= 1 - kernel % 2
            radius = max(0.5, min(kernel / 6, crop.width / 3, crop.height / 3))
            filtered = crop.filter(ImageFilter.GaussianBlur(radius=radius))
        image.paste(filtered, (left, top), local_mask)
