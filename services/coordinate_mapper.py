"""Map reconstructed OCR character spans to conservative image polygons."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from utils.geometry import (
    clip_polygon,
    normalize_with_mapping,
    polygon_dimensions,
    polygon_for_ratios,
    refine_box_with_components,
    refine_box_with_projection,
    validate_ink_coverage,
    weighted_span_ratios,
)


def _normalized(value: str) -> str:
    return normalize_with_mapping(value)["normalized_text"]


def _is_cjk(value: str) -> bool:
    return bool(value) and all("\u4e00" <= character <= "\u9fff" for character in value if not character.isspace())


class CoordinateMapper:
    def __init__(
        self,
        horizontal_padding: float = 8,
        vertical_padding: float = 5,
        *,
        strategy: str = "ink_refined",
        safety_mode: str = "strict",
        minimum_box_width: float = 4,
        minimum_box_height: float = 4,
        low_confidence_fallback: str = "full_block",
        minimum_mapping_confidence: float = 0.72,
        minimum_component_match_ratio: float = 0.70,
        ink_refinement_enabled: bool = True,
        projection_refinement_enabled: bool = True,
        component_gap_ratio: float = 0.45,
        maximum_overcover_ratio: float = 0.35,
    ) -> None:
        if safety_mode not in {"strict", "balanced"}:
            raise ValueError("mapping safety_mode must be strict or balanced")
        self.horizontal_padding = float(horizontal_padding)
        self.vertical_padding = float(vertical_padding)
        self.strategy = strategy
        self.safety_mode = safety_mode
        self.minimum_box_width = float(minimum_box_width)
        self.minimum_box_height = float(minimum_box_height)
        self.low_confidence_fallback = low_confidence_fallback
        self.minimum_mapping_confidence = float(minimum_mapping_confidence)
        self.minimum_component_match_ratio = float(minimum_component_match_ratio)
        self.ink_refinement_enabled = bool(ink_refinement_enabled)
        self.projection_refinement_enabled = bool(projection_refinement_enabled)
        self.component_gap_ratio = float(component_gap_ratio)
        self.maximum_overcover_ratio = float(maximum_overcover_ratio)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CoordinateMapper":
        return cls(**config)

    def _span_parts(
        self,
        entity: dict[str, Any],
        blocks: list[dict[str, Any]],
        char_map: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        by_index = {int(block["block_index"]): block for block in blocks}
        grouped: dict[int, list[int]] = defaultdict(list)
        for entry in char_map[max(0, int(entity.get("start", 0))):max(0, int(entity.get("end", 0)))]:
            if entry.get("block_index") is not None and entry.get("offset_in_block") is not None:
                grouped[int(entry["block_index"])].append(int(entry["offset_in_block"]))
        parts = []
        for block_index, offsets in grouped.items():
            block = by_index.get(block_index)
            if block and block.get("text") and offsets:
                parts.append({"block": block, "start": min(offsets), "end": max(offsets) + 1, "alignment": "direct"})
        if parts:
            return parts, None

        target = _normalized(str(entity.get("text", "")))
        if not target:
            return [], "empty_entity_text"
        candidates = []
        for block in blocks:
            normalized = normalize_with_mapping(str(block.get("text", "")))
            haystack = normalized["normalized_text"]
            position = haystack.find(target)
            while position >= 0:
                original = normalized["normalized_to_original"]
                candidates.append({
                    "block": block,
                    "start": original[position],
                    "end": original[position + len(target) - 1] + 1,
                    "alignment": "normalized",
                })
                position = haystack.find(target, position + 1)
        if len(candidates) == 1:
            return candidates, "normalized_alignment_without_span"
        if len(candidates) > 1:
            return candidates, "non_unique_normalized_match"

        fuzzy = []
        for block in blocks:
            block_text = str(block.get("text", ""))
            for start in range(len(block_text)):
                end = min(len(block_text), start + max(len(str(entity.get("text", ""))), 1) + 2)
                score = SequenceMatcher(None, target, _normalized(block_text[start:end])).ratio()
                if score >= 0.75:
                    fuzzy.append((score, {"block": block, "start": start, "end": end, "alignment": "fuzzy"}))
        if fuzzy:
            fuzzy.sort(key=lambda item: item[0], reverse=True)
            return [fuzzy[0][1]], "fuzzy_text_match"
        return [], "entity_not_found_in_ocr_blocks"

    def _dynamic_padding(self, block_polygon: list[list[float]], low_confidence: bool = False) -> tuple[float, float]:
        _, height = polygon_dimensions(block_polygon)
        if self.safety_mode == "strict":
            horizontal = max(self.horizontal_padding, height * (0.20 if low_confidence else 0.12))
            vertical = max(self.vertical_padding, height * 0.08)
        else:
            horizontal = max(self.horizontal_padding, height * 0.08)
            vertical = max(self.vertical_padding, height * 0.05)
        return horizontal, vertical

    def _map_part(
        self,
        entity: dict[str, Any],
        part: dict[str, Any],
        image: Any,
        image_size: tuple[int, int] | None,
        cross_block: bool,
        forced_reason: str | None,
    ) -> dict[str, Any]:
        block = part["block"]
        block_text = str(block["text"])
        start, end = int(part["start"]), int(part["end"])
        target_text = str(entity.get("text", ""))
        aligned_text = block_text[start:end]
        exact_alignment = _normalized(aligned_text) == _normalized(target_text) or cross_block
        start_ratio, end_ratio = weighted_span_ratios(block_text, start, end)
        initial_polygon = polygon_for_ratios(block["polygon"], start_ratio, end_ratio)
        mapping_steps: dict[str, Any] = {"weighted": initial_polygon}

        confidence = 0.10
        confidence += 0.25 if part["alignment"] == "direct" else 0.12
        confidence += 0.20 if exact_alignment else 0.0
        confidence += 0.15 * max(0.0, min(1.0, float(block.get("confidence", 0.0))))
        confidence += 0.15 if not cross_block else 0.0
        confidence += 0.10
        reasons = [forced_reason] if forced_reason else []

        projection = {"success": False, "confidence": 0.0, "reason": "projection_disabled"}
        components = {"success": False, "confidence": 0.0, "reason": "component_disabled"}
        refined_polygon = initial_polygon
        mapping_strategy = "weighted"
        if image is not None and self.projection_refinement_enabled:
            projection = refine_box_with_projection(image, block["polygon"], start_ratio, end_ratio)
            if projection.get("polygon"):
                mapping_steps["projection"] = projection["polygon"]
            if projection.get("success"):
                refined_polygon = projection["polygon"]
                mapping_strategy = "projection"
                confidence += 0.08 * float(projection.get("confidence", 0.0))
        pre_component_polygon = refined_polygon
        pre_component_strategy = mapping_strategy
        if image is not None and self.ink_refinement_enabled:
            components = refine_box_with_components(
                image, refined_polygon, block["polygon"], target_text, block_text, self.component_gap_ratio,
            )
            if components.get("polygon"):
                mapping_steps["component"] = components["polygon"]
            if components.get("success") and float(components.get("component_match_ratio", 0.0)) >= self.minimum_component_match_ratio:
                refined_polygon = components["polygon"]
                mapping_strategy = "component"
                confidence += 0.10 * float(components.get("confidence", 0.0))

        initial_width, _ = polygon_dimensions(initial_polygon)
        refined_width, _ = polygon_dimensions(refined_polygon)
        overcover = max(0.0, refined_width - initial_width) / max(initial_width, 1.0)
        if overcover > self.maximum_overcover_ratio:
            refined_polygon = pre_component_polygon
            mapping_strategy = pre_component_strategy
            confidence = min(confidence - 0.15, 0.65)
            reasons.append("refinement_change_too_large")

        adjacent_digit = end < len(block_text) and block_text[end].isdigit() and _is_cjk(aligned_text)
        boundary_reliable = (
            float(projection.get("confidence", 0.0)) >= 0.35
            or float(components.get("component_match_ratio", 0.0)) >= self.minimum_component_match_ratio
        )
        if adjacent_digit and not boundary_reliable:
            confidence -= 0.18
            reasons.append("unreliable_name_number_boundary")
        if cross_block:
            confidence = min(confidence, 0.60)
            reasons.append("cross_block_entity")
        if part["alignment"] == "fuzzy":
            confidence = min(confidence, 0.55)
            reasons.append("fuzzy_text_match")
        if image is not None and not projection.get("success") and not components.get("success"):
            confidence -= 0.15
            reasons.append("ink_refinement_failed")

        confidence = max(0.0, min(1.0, confidence))
        horizontal, vertical = self._dynamic_padding(block["polygon"], confidence < self.minimum_mapping_confidence)
        final_polygon = polygon_for_ratios(block["polygon"], start_ratio, end_ratio, horizontal, vertical)
        if mapping_strategy in {"projection", "component"}:
            width, _ = polygon_dimensions(block["polygon"])
            refined_points = components if mapping_strategy == "component" else projection
            final_polygon = polygon_for_ratios(
                block["polygon"], float(refined_points["start_ratio"]), float(refined_points["end_ratio"]), horizontal, vertical,
            )

        coverage = {"valid": True, "reason": None}
        if image is not None:
            coverage = validate_ink_coverage(image, final_polygon, block["polygon"], aligned_text, block_text)
            if not coverage.get("valid"):
                confidence = min(confidence, 0.60)
                reasons.append("possible_partial_entity_coverage")

        width, height = polygon_dimensions(final_polygon)
        if width < self.minimum_box_width or height < self.minimum_box_height:
            confidence = min(confidence, 0.40)
            reasons.append("box_below_minimum_size")
        if confidence < self.minimum_mapping_confidence:
            reasons.append("mapping_confidence_below_threshold")

        fallback = self.safety_mode == "strict" and self.low_confidence_fallback == "full_block" and bool(reasons)
        if fallback:
            final_polygon = [list(point) for point in block["polygon"]]
            mapping_strategy = "full_block"
            mapping_quality = "coarse"
        else:
            mapping_quality = "refined" if mapping_strategy in {"projection", "component"} else "estimated"
        final_polygon = clip_polygon(final_polygon, image_size)
        mapping_steps["final"] = final_polygon
        fallback_reason = ";".join(dict.fromkeys(reason for reason in reasons if reason)) or None
        return {
            "polygon": final_polygon,
            "mapping_quality": mapping_quality,
            "mapping_confidence": round(confidence, 4),
            "mapping_strategy": mapping_strategy,
            "fallback_reason": fallback_reason,
            "mapping_warning": "possible_partial_entity_coverage" if fallback or "possible_partial_entity_coverage" in reasons else None,
            "mapping_steps": mapping_steps,
        }

    def map_entity(
        self,
        entity: dict[str, Any],
        blocks: list[dict[str, Any]],
        char_map: list[dict[str, Any]],
        image_size: tuple[int, int] | None = None,
        image: Any = None,
    ) -> dict[str, Any]:
        parts, forced_reason = self._span_parts(entity, blocks, char_map)
        if not parts:
            return {
                **entity,
                "boxes": [],
                "mapping_quality": "coarse",
                "mapping_confidence": 0.0,
                "mapping_strategy": "full_block",
                "fallback_reason": forced_reason or "no_ocr_block",
                "mapping_warning": "possible_partial_entity_coverage",
                "mapping_steps": {},
            }
        cross_block = len(parts) > 1
        mapped = [self._map_part(entity, part, image, image_size, cross_block, forced_reason) for part in parts]
        strategies = {item["mapping_strategy"] for item in mapped}
        qualities = {item["mapping_quality"] for item in mapped}
        reasons = [item["fallback_reason"] for item in mapped if item["fallback_reason"]]
        return {
            **entity,
            "boxes": [item["polygon"] for item in mapped],
            "mapping_quality": "coarse" if "coarse" in qualities else "refined" if "refined" in qualities else "estimated",
            "mapping_confidence": round(min(item["mapping_confidence"] for item in mapped), 4),
            "mapping_strategy": strategies.pop() if len(strategies) == 1 else "multi_block",
            "fallback_reason": ";".join(dict.fromkeys(reasons)) or None,
            "mapping_warning": "possible_partial_entity_coverage" if any(item["mapping_warning"] for item in mapped) else None,
            "mapping_steps": [item["mapping_steps"] for item in mapped],
        }

    def map_entities(
        self,
        entities: list[dict[str, Any]],
        blocks: list[dict[str, Any]],
        char_map: list[dict[str, Any]],
        image_size: tuple[int, int] | None = None,
        image: Any = None,
    ) -> list[dict[str, Any]]:
        return [self.map_entity(entity, blocks, char_map, image_size, image) for entity in entities]
