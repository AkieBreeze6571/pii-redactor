from services.coordinate_mapper import CoordinateMapper
from services.ocr_service import reconstruct_text


def block(text: str, polygon: list[list[float]], index: int = 0) -> dict:
    return {"text": text, "confidence": 1.0, "polygon": polygon, "block_index": index}


def test_single_block_local_mapping() -> None:
    blocks = [block("收件人张三电话号", [[0, 0], [80, 0], [80, 20], [0, 20]])]
    rebuilt = reconstruct_text(blocks)
    result = CoordinateMapper(0, 0).map_entity({"type": "person", "text": "张三", "start": 3, "end": 5}, rebuilt["blocks"], rebuilt["char_map"])
    assert result["boxes"][0] == [[30.0, 0.0], [50.0, 0.0], [50.0, 20.0], [30.0, 20.0]]
    assert result["mapping_quality"] == "exact"


def test_cross_block_entity_gets_multiple_boxes() -> None:
    blocks = [
        block("四川省", [[0, 0], [60, 0], [60, 20], [0, 20]], 0),
        block("成都市", [[0, 30], [60, 30], [60, 50], [0, 50]], 1),
    ]
    rebuilt = reconstruct_text(blocks)
    result = CoordinateMapper(0, 0).map_entity({"type": "address", "text": rebuilt["full_text"], "start": 0, "end": len(rebuilt["full_text"])}, rebuilt["blocks"], rebuilt["char_map"])
    assert len(result["boxes"]) == 2
    assert result["mapping_quality"] == "estimated"


def test_tilted_polygon_interpolation() -> None:
    blocks = [block("1234", [[0, 0], [40, 10], [38, 30], [-2, 20]])]
    rebuilt = reconstruct_text(blocks)
    result = CoordinateMapper(0, 0).map_entity({"type": "phone", "text": "23", "start": 1, "end": 3}, rebuilt["blocks"], rebuilt["char_map"])
    assert result["boxes"][0][0] == [10.0, 2.5]
    assert result["boxes"][0][1] == [30.0, 7.5]


def test_boxes_are_clipped_to_image() -> None:
    blocks = [block("张三", [[-10, -10], [110, -10], [110, 50], [-10, 50]])]
    rebuilt = reconstruct_text(blocks)
    result = CoordinateMapper(5, 5).map_entity({"type": "person", "text": "张三", "start": 0, "end": 2}, rebuilt["blocks"], rebuilt["char_map"], (100, 40))
    assert all(0 <= x <= 100 and 0 <= y <= 40 for x, y in result["boxes"][0])


def test_repeated_value_uses_character_position() -> None:
    blocks = [block("123-123", [[0, 0], [70, 0], [70, 20], [0, 20]])]
    rebuilt = reconstruct_text(blocks)
    result = CoordinateMapper(0, 0).map_entity({"type": "postal_code", "text": "123", "start": 4, "end": 7}, rebuilt["blocks"], rebuilt["char_map"])
    assert result["boxes"][0][0][0] == 40


def test_coarse_fallback_and_empty_ocr() -> None:
    mapper = CoordinateMapper()
    entity = {"type": "person", "text": "张三", "start": 0, "end": 2}
    assert mapper.map_entity(entity, [], [])["mapping_quality"] == "coarse"
    blocks = [block("联系人张三", [[0, 0], [100, 0], [100, 20], [0, 20]])]
    result = mapper.map_entity(entity, blocks, [])
    assert result["boxes"] == [blocks[0]["polygon"]]
    assert result["mapping_quality"] == "coarse"
