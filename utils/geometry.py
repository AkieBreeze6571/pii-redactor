"""Geometry, text alignment, and ink-analysis helpers for safe OCR mapping."""

from __future__ import annotations

import unicodedata
from typing import Any

import cv2
import numpy as np
from PIL import Image


_PUNCTUATION_MAP = str.maketrans({"，": ",", "。": ".", "：": ":", "；": ";", "（": "(", "）": ")"})


def normalize_with_mapping(text: str) -> dict[str, Any]:
    """Normalize OCR text while retaining both index directions."""

    normalized: list[str] = []
    normalized_to_original: list[int] = []
    original_to_normalized: list[int | None] = [None] * len(text)
    for original_index, character in enumerate(text):
        value = unicodedata.normalize("NFKC", character).translate(_PUNCTUATION_MAP).lower()
        for output_character in value:
            if output_character.isspace():
                continue
            if original_to_normalized[original_index] is None:
                original_to_normalized[original_index] = len(normalized)
            normalized.append(output_character)
            normalized_to_original.append(original_index)
    return {
        "normalized_text": "".join(normalized),
        "normalized_to_original": normalized_to_original,
        "original_to_normalized": original_to_normalized,
    }


def get_character_visual_weight(character: str) -> float:
    if character == "\u3000":
        return 1.0
    if character.isspace():
        return 0.30
    if "\u4e00" <= character <= "\u9fff" or unicodedata.east_asian_width(character) in {"W", "F"}:
        return 1.0
    if character.isupper():
        return 0.72
    if character.islower() or character.isdigit():
        return 0.58
    if unicodedata.category(character).startswith("P"):
        return 0.35
    return 0.72


def weighted_span_ratios(text: str, start: int, end: int) -> tuple[float, float]:
    weights = [get_character_visual_weight(character) for character in text]
    total = max(sum(weights), 1e-9)
    return sum(weights[:start]) / total, sum(weights[:end]) / total


def polygon_dimensions(polygon: list[list[float]]) -> tuple[float, float]:
    if len(polygon) < 4:
        return 0.0, 0.0
    points = np.asarray(polygon[:4], dtype=np.float32)
    width = (np.linalg.norm(points[1] - points[0]) + np.linalg.norm(points[2] - points[3])) / 2
    height = (np.linalg.norm(points[3] - points[0]) + np.linalg.norm(points[2] - points[1])) / 2
    return float(width), float(height)


def _interpolate(left: list[float], right: list[float], ratio: float) -> list[float]:
    return [left[0] + (right[0] - left[0]) * ratio, left[1] + (right[1] - left[1]) * ratio]


def polygon_for_ratios(
    polygon: list[list[float]], start_ratio: float, end_ratio: float,
    padding_x: float = 0.0, padding_y: float = 0.0,
) -> list[list[float]]:
    """Interpolate along a possibly tilted quadrilateral's writing direction."""

    if len(polygon) < 4:
        return [list(point) for point in polygon]
    width, _ = polygon_dimensions(polygon)
    ratio_padding = padding_x / max(width, 1.0)
    start_ratio = max(0.0, min(1.0, start_ratio - ratio_padding))
    end_ratio = max(start_ratio, min(1.0, end_ratio + ratio_padding))
    top_left, top_right, bottom_right, bottom_left = polygon[:4]
    top_start = _interpolate(top_left, top_right, start_ratio)
    top_end = _interpolate(top_left, top_right, end_ratio)
    bottom_end = _interpolate(bottom_left, bottom_right, end_ratio)
    bottom_start = _interpolate(bottom_left, bottom_right, start_ratio)
    for top, bottom in ((top_start, bottom_start), (top_end, bottom_end)):
        direction = np.asarray(bottom, dtype=np.float32) - np.asarray(top, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            unit = direction / norm
            top[0] -= float(unit[0] * padding_y); top[1] -= float(unit[1] * padding_y)
            bottom[0] += float(unit[0] * padding_y); bottom[1] += float(unit[1] * padding_y)
    return [top_start, top_end, bottom_end, bottom_start]


def clip_polygon(polygon: list[list[float]], image_size: tuple[int, int] | None) -> list[list[float]]:
    if image_size is None:
        return polygon
    width, height = image_size
    return [[max(0.0, min(float(width - 1), float(x))), max(0.0, min(float(height - 1), float(y)))] for x, y in polygon]


def _as_rgb_array(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if array.shape[2] == 4:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    return array.astype(np.uint8)


def rectify_block(image: Any, polygon: list[list[float]]) -> dict[str, Any] | None:
    """Perspective-warp an OCR polygon and extract foreground ink."""

    try:
        width, height = polygon_dimensions(polygon)
        target_width, target_height = max(2, round(width)), max(2, round(height))
        source = np.asarray(polygon[:4], dtype=np.float32)
        target = np.asarray([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(source, target)
        warped = cv2.warpPerspective(_as_rgb_array(image), matrix, (target_width, target_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        border = np.concatenate((blurred[0], blurred[-1], blurred[:, 0], blurred[:, -1]))
        light_background = float(np.median(border)) >= 127
        threshold_mode = cv2.THRESH_BINARY_INV if light_background else cv2.THRESH_BINARY
        _, otsu = cv2.threshold(blurred, 0, 255, threshold_mode + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_mode, 15, 5)
        binary = cv2.bitwise_and(otsu, adaptive)
        return {"warped": warped, "gray": gray, "binary": binary, "matrix": matrix, "width": target_width, "height": target_height}
    except (cv2.error, ValueError, TypeError, IndexError):
        return None


def _nearest_projection_gap(projection: np.ndarray, estimate: float, radius: int) -> tuple[int, float]:
    center = int(round(estimate))
    left, right = max(0, center - radius), min(len(projection) - 1, center + radius)
    if right <= left:
        return center, 0.0
    local = projection[left:right + 1]
    maximum = max(float(local.max()), 1.0)
    distances = np.abs(np.arange(left, right + 1) - center) / max(radius, 1)
    scores = local / maximum + distances * 0.30
    index = left + int(np.argmin(scores))
    strength = max(0.0, min(1.0, 1.0 - float(projection[index]) / maximum))
    return index, strength


def refine_box_with_projection(
    image: Any, full_block_polygon: list[list[float]], start_ratio: float, end_ratio: float,
) -> dict[str, Any]:
    rectified = rectify_block(image, full_block_polygon)
    if not rectified or np.count_nonzero(rectified["binary"]) < 4:
        return {"success": False, "confidence": 0.0, "reason": "projection_no_ink"}
    projection = np.count_nonzero(rectified["binary"], axis=0).astype(np.float32)
    projection = np.convolve(projection, np.ones(3, dtype=np.float32) / 3, mode="same")
    width, height = rectified["width"], rectified["height"]
    radius = max(3, min(round(height * 0.8), round(width * 0.12)))
    left, left_strength = _nearest_projection_gap(projection, start_ratio * width, radius)
    right, right_strength = _nearest_projection_gap(projection, end_ratio * width, radius)
    if right <= left:
        return {"success": False, "confidence": 0.0, "reason": "projection_invalid_boundaries"}
    confidence = (left_strength + right_strength) / 2
    return {
        "success": confidence >= 0.25,
        "confidence": confidence,
        "start_ratio": left / width,
        "end_ratio": right / width,
        "polygon": polygon_for_ratios(full_block_polygon, left / width, right / width),
        "reason": None if confidence >= 0.25 else "projection_weak_boundaries",
    }


def _ink_groups(binary: np.ndarray, height: int) -> list[tuple[int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    minimum_area = max(2, round(binary.shape[0] * binary.shape[1] * 0.0003))
    components = []
    for index in range(1, count):
        x, _, width, component_height, area = stats[index]
        if area >= minimum_area and component_height >= max(2, round(height * 0.08)):
            components.append((int(x), int(x + width)))
    components.sort()
    groups: list[list[int]] = []
    maximum_internal_gap = max(1, round(height * 0.10))
    for left, right in components:
        if not groups or left - groups[-1][1] > maximum_internal_gap:
            groups.append([left, right])
        else:
            groups[-1][1] = max(groups[-1][1], right)
    return [(left, right) for left, right in groups]


def refine_box_with_components(
    image: Any,
    estimated_polygon: list[list[float]],
    full_block_polygon: list[list[float]],
    entity_text: str,
    block_text: str,
    component_gap_ratio: float = 0.45,
) -> dict[str, Any]:
    """Expand a local estimate to nearby ink groups without crossing a large gap."""

    rectified = rectify_block(image, full_block_polygon)
    if not rectified:
        return {"success": False, "confidence": 0.0, "reason": "component_rectification_failed"}
    groups = _ink_groups(rectified["binary"], rectified["height"])
    if not groups:
        return {"success": False, "confidence": 0.0, "reason": "component_no_groups"}
    matrix = rectified["matrix"]
    transformed = cv2.perspectiveTransform(np.asarray([estimated_polygon], dtype=np.float32), matrix)[0]
    estimated_left, estimated_right = float(transformed[:, 0].min()), float(transformed[:, 0].max())
    expected = max(1, len(normalize_with_mapping(entity_text)["normalized_text"]))
    selected = [index for index, (left, right) in enumerate(groups) if right >= estimated_left and left <= estimated_right]
    if not selected:
        nearest = min(range(len(groups)), key=lambda index: abs((groups[index][0] + groups[index][1]) / 2 - (estimated_left + estimated_right) / 2))
        selected = [nearest]
    typical_width = max((estimated_right - estimated_left) / expected, rectified["height"] * 0.35, 1.0)
    maximum_gap = min(rectified["height"] * component_gap_ratio, typical_width * component_gap_ratio)
    while len(selected) < expected:
        candidates: list[tuple[float, int]] = []
        if min(selected) > 0:
            index = min(selected) - 1; candidates.append((groups[min(selected)][0] - groups[index][1], index))
        if max(selected) + 1 < len(groups):
            index = max(selected) + 1; candidates.append((groups[index][0] - groups[max(selected)][1], index))
        if not candidates:
            break
        gap, index = min(candidates)
        if gap > maximum_gap:
            break
        selected.append(index); selected.sort()
    left, right = groups[min(selected)][0], groups[max(selected)][1]
    match_ratio = min(1.0, len(selected) / expected)
    confidence = 0.45 + 0.45 * match_ratio
    width = rectified["width"]
    return {
        "success": match_ratio >= 0.5,
        "confidence": confidence,
        "component_match_ratio": match_ratio,
        "groups": groups,
        "selected_groups": [groups[index] for index in selected],
        "start_ratio": left / width,
        "end_ratio": right / width,
        "polygon": polygon_for_ratios(full_block_polygon, left / width, right / width),
        "reason": None if match_ratio >= 0.5 else "component_count_mismatch",
    }


def validate_ink_coverage(
    image: Any,
    entity_polygon: list[list[float]],
    block_polygon: list[list[float]],
    entity_text: str = "",
    block_text: str = "",
) -> dict[str, Any]:
    """Detect suspiciously narrow local boxes or ink touching their edges."""

    rectified = rectify_block(image, block_polygon)
    if not rectified:
        return {"valid": False, "confidence": 0.0, "reason": "coverage_rectification_failed"}
    transformed = cv2.perspectiveTransform(np.asarray([entity_polygon], dtype=np.float32), rectified["matrix"])[0]
    left = max(0, round(float(transformed[:, 0].min())))
    right = min(rectified["width"], round(float(transformed[:, 0].max())))
    actual_width = max(0, right - left)
    block_weight = max(sum(get_character_visual_weight(character) for character in block_text), 1e-9)
    entity_weight = sum(get_character_visual_weight(character) for character in entity_text)
    expected_width = entity_weight / block_weight * rectified["width"]
    cjk_count = sum("\u4e00" <= character <= "\u9fff" for character in entity_text)
    if cjk_count:
        expected_width = max(expected_width, cjk_count * rectified["height"] * 0.70)
    strip = max(1, round(rectified["height"] * 0.10))
    projection = np.count_nonzero(rectified["binary"], axis=0)
    left_touch = left > 0 and bool(np.any(projection[max(0, left - strip):left] > 0))
    right_touch = right < len(projection) and bool(np.any(projection[right:min(len(projection), right + strip)] > 0))
    width_ratio = actual_width / max(expected_width, 1.0)
    too_narrow = actual_width < max(4, rectified["height"] * 0.45) or width_ratio < 0.72
    valid = not too_narrow and not (left_touch and right_touch)
    return {
        "valid": valid,
        "confidence": max(0.0, min(1.0, width_ratio)),
        "left_edge_ink": left_touch,
        "right_edge_ink": right_touch,
        "width_ratio": width_ratio,
        "reason": "possible_partial_entity_coverage" if not valid else None,
    }
