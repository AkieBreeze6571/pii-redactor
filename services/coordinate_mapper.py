"""Map reconstructed character spans back to local OCR polygons."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _interpolate(left: list[float], right: list[float], ratio: float) -> list[float]:
    return [left[0] + (right[0] - left[0]) * ratio, left[1] + (right[1] - left[1]) * ratio]


def _local_polygon(polygon: list[list[float]], start: int, end: int, length: int, padding_x: float, padding_y: float) -> list[list[float]]:
    if len(polygon) < 4 or length <= 0:
        return polygon
    top_left, top_right, bottom_right, bottom_left = polygon[:4]
    start_ratio, end_ratio = start / length, end / length
    points = [
        _interpolate(top_left, top_right, start_ratio),
        _interpolate(top_left, top_right, end_ratio),
        _interpolate(bottom_left, bottom_right, end_ratio),
        _interpolate(bottom_left, bottom_right, start_ratio),
    ]
    points[0][0] -= padding_x; points[3][0] -= padding_x
    points[1][0] += padding_x; points[2][0] += padding_x
    points[0][1] -= padding_y; points[1][1] -= padding_y
    points[2][1] += padding_y; points[3][1] += padding_y
    return points


def _clip(polygon: list[list[float]], image_size: tuple[int, int] | None) -> list[list[float]]:
    if image_size is None:
        return polygon
    width, height = image_size
    return [[max(0.0, min(float(width), x)), max(0.0, min(float(height), y))] for x, y in polygon]


class CoordinateMapper:
    def __init__(self, horizontal_padding: float = 5, vertical_padding: float = 3) -> None:
        self.horizontal_padding = horizontal_padding
        self.vertical_padding = vertical_padding

    def map_entity(self, entity: dict[str, Any], blocks: list[dict[str, Any]], char_map: list[dict[str, Any]], image_size: tuple[int, int] | None = None) -> dict[str, Any]:
        by_index = {block["block_index"]: block for block in blocks}
        grouped: dict[int, list[int]] = defaultdict(list)
        expected = max(entity["end"] - entity["start"], 1)
        mapped = 0
        for entry in char_map[entity["start"]:entity["end"]]:
            block_index = entry.get("block_index")
            offset = entry.get("offset_in_block")
            if block_index is not None and offset is not None:
                grouped[int(block_index)].append(int(offset)); mapped += 1
        polygons = []
        for block_index, offsets in grouped.items():
            block = by_index.get(block_index)
            if not block or not block.get("text"):
                continue
            polygon = _local_polygon(block["polygon"], min(offsets), max(offsets) + 1, len(block["text"]), self.horizontal_padding, self.vertical_padding)
            polygons.append(_clip(polygon, image_size))
        quality = "exact" if polygons and mapped == expected else "estimated" if polygons else "coarse"
        if not polygons:
            for block in blocks:
                if entity.get("text") and entity["text"] in block.get("text", ""):
                    polygons = [_clip(block["polygon"], image_size)]; break
        return {**entity, "boxes": polygons, "mapping_quality": quality}

    def map_entities(self, entities: list[dict[str, Any]], blocks: list[dict[str, Any]], char_map: list[dict[str, Any]], image_size: tuple[int, int] | None = None) -> list[dict[str, Any]]:
        return [self.map_entity(entity, blocks, char_map, image_size) for entity in entities]
