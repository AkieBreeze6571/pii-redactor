from __future__ import annotations

from PIL import Image, ImageDraw

from services.coordinate_mapper import CoordinateMapper
from services.ocr_service import reconstruct_text
from utils.geometry import get_character_visual_weight, normalize_with_mapping, polygon_dimensions


NAME_THREE = "赵明宇"
NAME_TWO = "李华"
NUMBER = "20263301014"


def synthetic_line(
    name: str = NAME_THREE,
    *,
    include_space: bool = True,
    dark: bool = False,
    third_width: int = 44,
    number_gap: int = 28,
) -> tuple[Image.Image, dict, dict]:
    image = Image.new("RGB", (330, 80), "black" if dark else "white")
    ink = "white" if dark else "black"
    draw = ImageDraw.Draw(image)
    positions = [(15, 40), (50, 78)]
    if len(name) == 3:
        positions.append((88, 88 + third_width))
    for left, right in positions:
        draw.rectangle((left, 15, right, 62), fill=ink)
    name_right = positions[-1][1]
    digit_left = name_right + number_gap
    for index in range(len(NUMBER)):
        left = digit_left + index * 12
        draw.rectangle((left, 22, left + 7, 58), fill=ink)
    text = name + (" " if include_space else "") + NUMBER
    block = {"text": text, "confidence": 0.98, "polygon": [[5, 8], [325, 8], [325, 70], [5, 70]], "block_index": 0}
    rebuilt = reconstruct_text([block])
    entity = {"type": "person", "text": name, "start": 0, "end": len(name), "confidence": 0.99, "source": "ner"}
    return image, rebuilt, entity


def map_name(image: Image.Image, rebuilt: dict, entity: dict, mode: str = "strict") -> dict:
    mapper = CoordinateMapper(horizontal_padding=8, vertical_padding=5, safety_mode=mode)
    return mapper.map_entity(entity, rebuilt["blocks"], rebuilt["char_map"], image.size, image)


def polygon_right(polygon: list[list[float]]) -> float:
    return max(point[0] for point in polygon)


def test_normalize_with_mapping_preserves_index_directions() -> None:
    result = normalize_with_mapping("Ａ  B　，赵")
    assert result["normalized_text"] == "ab,赵"
    assert result["normalized_to_original"] == [0, 3, 5, 6]
    assert result["original_to_normalized"][1] is None


def test_character_visual_weights_distinguish_cjk_digits_and_spaces() -> None:
    assert get_character_visual_weight("赵") == 1.0
    assert get_character_visual_weight("A") == 0.72
    assert get_character_visual_weight("a") == get_character_visual_weight("2") == 0.58
    assert get_character_visual_weight(" ") == 0.30


def test_three_character_name_before_digits_is_fully_covered() -> None:
    image, rebuilt, entity = synthetic_line()
    result = map_name(image, rebuilt, entity)
    initial = result["mapping_steps"][0]["weighted"]
    assert polygon_right(initial) < 132
    assert polygon_right(result["boxes"][0]) >= 132
    assert result["mapping_strategy"] in {"component", "full_block"}


def test_two_character_name_before_digits_stays_safe() -> None:
    image, rebuilt, entity = synthetic_line(NAME_TWO)
    result = map_name(image, rebuilt, entity)
    assert polygon_right(result["boxes"][0]) >= 78


def test_tiny_name_number_gap_never_loses_third_character() -> None:
    image, rebuilt, entity = synthetic_line(number_gap=5)
    result = map_name(image, rebuilt, entity)
    assert polygon_right(result["boxes"][0]) >= 132


def test_missing_ocr_space_still_maps_name() -> None:
    image, rebuilt, entity = synthetic_line(include_space=False)
    result = map_name(image, rebuilt, entity)
    assert result["boxes"] and polygon_right(result["boxes"][0]) >= 132


def test_extra_ocr_space_keeps_direct_span_alignment() -> None:
    image, rebuilt, entity = synthetic_line(include_space=True)
    rebuilt["blocks"][0]["text"] = NAME_THREE + "  " + NUMBER
    rebuilt = reconstruct_text(rebuilt["blocks"])
    result = map_name(image, rebuilt, entity)
    assert result["mapping_confidence"] > 0
    assert result["fallback_reason"] is None or result["mapping_strategy"] == "full_block"


def test_wide_third_character_is_refined_or_full_block() -> None:
    image, rebuilt, entity = synthetic_line(third_width=58)
    result = map_name(image, rebuilt, entity)
    assert polygon_right(result["boxes"][0]) >= 146


def test_balanced_mode_keeps_local_box_when_ink_is_unreliable() -> None:
    image, rebuilt, entity = synthetic_line()
    blank = Image.new("RGB", image.size, "white")
    result = map_name(blank, rebuilt, entity, "balanced")
    assert result["mapping_strategy"] == "weighted"
    assert polygon_right(result["boxes"][0]) < 325


def test_balanced_mode_rejects_component_overexpansion_into_number() -> None:
    image, rebuilt, entity = synthetic_line()
    ImageDraw.Draw(image).rectangle((15, 30, 310, 42), fill="black")
    result = map_name(image, rebuilt, entity, "balanced")
    assert result["mapping_strategy"] != "component"
    assert polygon_right(result["boxes"][0]) < 250


def test_strict_mode_uses_full_block_when_ink_is_unreliable() -> None:
    image, rebuilt, entity = synthetic_line()
    blank = Image.new("RGB", image.size, "white")
    result = map_name(blank, rebuilt, entity, "strict")
    assert result["mapping_strategy"] == "full_block"
    assert result["boxes"][0] == rebuilt["blocks"][0]["polygon"]
    assert "ink_refinement_failed" in result["fallback_reason"]


def test_dynamic_padding_uses_block_height() -> None:
    image, rebuilt, entity = synthetic_line()
    mapper = CoordinateMapper(horizontal_padding=1, vertical_padding=1, safety_mode="balanced")
    result = mapper.map_entity(entity, rebuilt["blocks"], rebuilt["char_map"], image.size)
    initial_width, _ = polygon_dimensions(result["mapping_steps"][0]["weighted"])
    final_width, _ = polygon_dimensions(result["boxes"][0])
    # The left edge is clipped to the OCR block; the right side still receives dynamic padding.
    assert final_width >= initial_width + (62 * 0.08) - 1


def test_tilted_weighted_polygon_follows_main_direction() -> None:
    block = {"text": NAME_THREE + NUMBER, "confidence": 1.0, "polygon": [[5, 5], [305, 25], [300, 75], [0, 55]], "block_index": 0}
    rebuilt = reconstruct_text([block])
    entity = {"type": "person", "text": NAME_THREE, "start": 0, "end": 3}
    result = CoordinateMapper(safety_mode="balanced").map_entity(entity, rebuilt["blocks"], rebuilt["char_map"], (320, 90))
    initial = result["mapping_steps"][0]["weighted"]
    assert initial[1][1] > initial[0][1]


def test_dark_background_projection_does_not_crash_or_undercover() -> None:
    image, rebuilt, entity = synthetic_line(dark=True)
    result = map_name(image, rebuilt, entity)
    assert result["boxes"] and polygon_right(result["boxes"][0]) >= 132


def test_final_box_is_clipped_to_image() -> None:
    image, rebuilt, entity = synthetic_line()
    rebuilt["blocks"][0]["polygon"] = [[-20, -10], [350, -10], [350, 90], [-20, 90]]
    result = map_name(image, rebuilt, entity)
    assert all(0 <= x < image.width and 0 <= y < image.height for box in result["boxes"] for x, y in box)


def test_mapping_metadata_is_always_present() -> None:
    image, rebuilt, entity = synthetic_line()
    result = map_name(image, rebuilt, entity)
    assert {"mapping_quality", "mapping_confidence", "mapping_strategy", "fallback_reason"} <= result.keys()
