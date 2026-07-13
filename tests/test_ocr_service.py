from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from services.image_service import preprocess_image
from services.ocr_service import OcrService, parse_ocr_result, reconstruct_text


def test_parse_paddleocr_3_result() -> None:
    raw = [{"res": {"rec_texts": ["联系人张三"], "rec_scores": [0.98], "rec_polys": [[[10, 20], [110, 20], [110, 40], [10, 40]]]}}]
    blocks = parse_ocr_result(raw, scale_x=2, scale_y=3)
    assert blocks[0]["text"] == "联系人张三"
    assert blocks[0]["polygon"][0] == [20.0, 60.0]


def test_parse_legacy_result() -> None:
    raw = [[[[[0, 0], [20, 0], [20, 10], [0, 10]], ("张三", 0.9)]]]
    assert parse_ocr_result(raw)[0]["confidence"] == 0.9


def test_reconstruct_text_and_inserted_char_map() -> None:
    blocks = [
        {"text": "第二行", "confidence": 1.0, "polygon": [[0, 30], [60, 30], [60, 50], [0, 50]], "block_index": 1},
        {"text": "第一行", "confidence": 1.0, "polygon": [[0, 0], [60, 0], [60, 20], [0, 20]], "block_index": 0},
    ]
    result = reconstruct_text(blocks)
    assert result["full_text"] == "第一行\n第二行"
    assert result["char_map"][3]["block_index"] is None


def test_preprocess_rgba_and_resize() -> None:
    image = Image.new("RGBA", (400, 200), (0, 0, 0, 0))
    result = preprocess_image(image, max_side=100)
    assert result.image.mode == "RGB"
    assert result.image.size == (100, 50)
    assert result.scale_x == 4


def test_preprocess_grayscale_numpy() -> None:
    result = preprocess_image(np.zeros((20, 30), dtype=np.uint8))
    assert result.image.size == (30, 20)


def test_ocr_failure_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    service = OcrService({"cache_enabled": False, "max_image_side": 100})
    monkeypatch.setattr(service, "_get_engine", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    assert service.recognize(Image.new("RGB", (20, 20), "white")) == []
    assert "OCR 识别失败" in service.last_warning


@pytest.mark.integration
def test_real_paddleocr_on_generated_image() -> None:
    images = list(Path("data/generated/images/test").glob("*.png"))
    if not images:
        pytest.skip("没有本地合成测试图")
    service = OcrService({"language": "ch", "max_image_side": 2400, "use_angle_cls": False, "cache_enabled": True, "config_version": 1})
    blocks = service.recognize(images[0])
    assert blocks
    assert all(block["text"] and len(block["polygon"]) == 4 for block in blocks)
