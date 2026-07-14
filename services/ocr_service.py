"""PaddleOCR 3.x/legacy compatibility, caching, and text reconstruction."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from statistics import median
from threading import RLock
from typing import Any

import cv2
import numpy as np

from services.image_service import PreprocessedImage, preprocess_image


LOGGER = logging.getLogger(__name__)


def parse_ocr_result(result: Any, scale_x: float = 1.0, scale_y: float = 1.0) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if not result:
        return blocks
    for page in result if isinstance(result, list) else [result]:
        value = page.json if hasattr(page, "json") else page
        if isinstance(value, dict) and "res" in value:
            value = value["res"]
        if isinstance(value, dict) and "rec_texts" in value:
            texts = value.get("rec_texts", [])
            scores = value.get("rec_scores", [])
            polygons = value.get("rec_polys") or value.get("dt_polys") or []
            for text, score, polygon in zip(texts, scores, polygons):
                blocks.append(_block(text, score, polygon, len(blocks), scale_x, scale_y))
            continue
        lines = value
        if (
            isinstance(lines, (list, tuple))
            and len(lines) >= 2
            and isinstance(lines[0], (list, tuple))
            and lines[0]
            and isinstance(lines[0][0], (list, tuple))
            and isinstance(lines[1], (list, tuple))
            and isinstance(lines[1][0], str)
        ):
            lines = [lines]
        if isinstance(lines, list):
            for line in lines:
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                polygon, recognition = line[0], line[1]
                if isinstance(recognition, (list, tuple)) and len(recognition) >= 2:
                    blocks.append(_block(recognition[0], recognition[1], polygon, len(blocks), scale_x, scale_y))
    return blocks


def _block(text: Any, score: Any, polygon: Any, index: int, scale_x: float, scale_y: float) -> dict[str, Any]:
    points = [[float(point[0]) * scale_x, float(point[1]) * scale_y] for point in polygon]
    return {"text": str(text), "confidence": float(score), "polygon": points, "block_index": index}


def sort_ocr_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not blocks:
        return []
    heights = [max(point[1] for point in block["polygon"]) - min(point[1] for point in block["polygon"]) for block in blocks]
    tolerance = max(4.0, median(heights) * 0.6)
    ordered = sorted(blocks, key=lambda block: (min(point[1] for point in block["polygon"]), min(point[0] for point in block["polygon"])))
    lines: list[list[dict[str, Any]]] = []
    for block in ordered:
        center_y = sum(point[1] for point in block["polygon"]) / len(block["polygon"])
        for line in lines:
            line_y = sum(sum(point[1] for point in item["polygon"]) / len(item["polygon"]) for item in line) / len(line)
            if abs(center_y - line_y) <= tolerance:
                line.append(block); break
        else:
            lines.append([block])
    output = []
    for line in sorted(lines, key=lambda items: min(min(point[1] for point in item["polygon"]) for item in items)):
        output.extend(sorted(line, key=lambda item: min(point[0] for point in item["polygon"])))
    for index, block in enumerate(output): block["block_index"] = index
    return output


def reconstruct_text(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sort_ocr_blocks([dict(block) for block in blocks])
    full_text = ""
    char_map: list[dict[str, int | None]] = []
    for block_position, block in enumerate(ordered):
        if block_position:
            full_text += "\n"
            char_map.append({"char_index": len(full_text) - 1, "block_index": None, "offset_in_block": None})
        for offset, character in enumerate(block["text"]):
            full_text += character
            char_map.append({"char_index": len(full_text) - 1, "block_index": block["block_index"], "offset_in_block": offset})
    return {"full_text": full_text, "blocks": ordered, "char_map": char_map}


class OcrService:
    _engine: Any = None
    _engine_lock = RLock()
    _engine_init_count = 0
    _cache: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = OrderedDict()
    _cache_lock = RLock()

    def __init__(self, config: dict[str, Any], cache_root: str | Path = "data/cache/ocr") -> None:
        self.config = config
        # Retained for constructor compatibility; OCR cache data now stays in bounded process memory.
        self.cache_root = Path(cache_root)
        self.last_warning = ""
        self.last_cache_hit = False

    @classmethod
    def initialization_count(cls) -> int:
        return cls._engine_init_count

    @classmethod
    def cache_size(cls) -> int:
        with cls._cache_lock:
            return len(cls._cache)

    @classmethod
    def clear_memory_cache(cls) -> None:
        with cls._cache_lock:
            cls._cache.clear()

    @classmethod
    def device_name(cls) -> str:
        if cls._engine is None:
            return "not_initialized"
        for attribute in ("device", "_device"):
            value = getattr(cls._engine, attribute, None)
            if value is not None:
                return str(value)
        try:
            import paddle
            return str(paddle.device.get_device())
        except (ImportError, AttributeError, RuntimeError):
            return "initialized_unknown"

    def _get_engine(self) -> Any:
        if self.__class__._engine is not None:
            return self.__class__._engine
        with self._engine_lock:
            if self.__class__._engine is None:
                from paddleocr import PaddleOCR
                engine = PaddleOCR(
                    lang=self.config.get("language", "ch"),
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=bool(self.config.get("use_angle_cls", True)),
                    enable_mkldnn=False,
                )
                self.__class__._engine = engine
                self.__class__._engine_init_count += 1
        return self.__class__._engine

    def recognize(self, image: Any) -> list[dict[str, Any]]:
        self.last_cache_hit = False
        try:
            prepared = preprocess_image(image, max_side=int(self.config.get("max_image_side", 2400)))
            rgb = np.asarray(prepared.image)
            digest = hashlib.sha256(rgb.tobytes() + json.dumps(self.config, sort_keys=True).encode()).hexdigest()
            cache_enabled = bool(self.config.get("cache_enabled", True))
            cache_ttl = max(1.0, float(self.config.get("cache_ttl_seconds", 600)))
            cache_limit = max(1, int(self.config.get("cache_max_entries", 32)))
            now = time.monotonic()
            if cache_enabled:
                with self._cache_lock:
                    expired = [key for key, (expires_at, _) in self._cache.items() if expires_at <= now]
                    for key in expired:
                        self._cache.pop(key, None)
                    cached = self._cache.get(digest)
                    if cached is not None:
                        self._cache.move_to_end(digest)
                        self.last_cache_hit = True
                        self.last_warning = ""
                        return copy.deepcopy(cached[1])
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            result = self._get_engine().predict(bgr)
            blocks = parse_ocr_result(result, prepared.scale_x, prepared.scale_y)
            if cache_enabled:
                with self._cache_lock:
                    self._cache[digest] = (now + cache_ttl, copy.deepcopy(blocks))
                    self._cache.move_to_end(digest)
                    while len(self._cache) > cache_limit:
                        self._cache.popitem(last=False)
            self.last_warning = ""
            return blocks
        except Exception as exc:
            LOGGER.error("OCR inference failed error_type=%s", type(exc).__name__)
            self.last_warning = f"OCR 识别失败：{type(exc).__name__}"
            return []
