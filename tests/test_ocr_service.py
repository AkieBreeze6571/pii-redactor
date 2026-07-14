import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

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


def test_ocr_engine_is_initialized_once_under_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    created = []

    class FakePaddleOcr:
        def __init__(self, **kwargs) -> None:
            time.sleep(0.02)
            created.append(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakePaddleOcr))
    monkeypatch.setattr(OcrService, "_engine", None)
    monkeypatch.setattr(OcrService, "_engine_init_count", 0)
    service = OcrService({"cache_enabled": False})
    with ThreadPoolExecutor(max_workers=8) as executor:
        engines = list(executor.map(lambda _: service._get_engine(), range(8)))
    assert len({id(engine) for engine in engines}) == 1
    assert len(created) == 1
    assert OcrService.initialization_count() == 1


def test_memory_cache_is_bounded_and_does_not_write_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEngine:
        def predict(self, image):
            return []

    monkeypatch.setattr(OcrService, "_cache", OrderedDict())
    service = OcrService({"cache_enabled": True, "cache_max_entries": 2, "cache_ttl_seconds": 60, "max_image_side": 100}, cache_root=tmp_path / "disk-cache")
    monkeypatch.setattr(service, "_get_engine", lambda: FakeEngine())
    for value in (10, 20, 30):
        service.recognize(Image.new("RGB", (20, 20), (value, value, value)))
    assert OcrService.cache_size() == 2
    service.recognize(Image.new("RGB", (20, 20), (30, 30, 30)))
    assert service.last_cache_hit is True
    assert not (tmp_path / "disk-cache").exists()


@pytest.mark.integration
def test_real_paddleocr_on_generated_image() -> None:
    images = list(Path("data/generated/images/test").glob("*.png"))
    if not images:
        pytest.skip("没有本地合成测试图")
    service = OcrService({"language": "ch", "max_image_side": 2400, "use_angle_cls": False, "cache_enabled": True, "config_version": 1})
    blocks = service.recognize(images[0])
    assert blocks
    assert all(block["text"] and len(block["polygon"]) == 4 for block in blocks)
