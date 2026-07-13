from pathlib import Path

from PIL import Image

from services.document_processor import DocumentProcessor
from services.redaction_service import RedactionService


class FakeOcr:
    last_warning = ""

    def __init__(self, blocks: list[dict]) -> None:
        self.blocks = blocks

    def recognize(self, image):
        return [dict(block) for block in self.blocks]


class FakeDetector:
    model_source = "local_finetuned"

    def __init__(self, entities: list[dict]) -> None:
        self.entities = entities

    def detect(self, text, enabled_types=None, thresholds=None):
        return [dict(item) for item in self.entities]


def block(text: str = "电话13812345678") -> dict:
    return {"text": text, "confidence": 1.0, "polygon": [[10, 10], [190, 10], [190, 40], [10, 40]], "block_index": 0}


def test_complete_mocked_pipeline(tmp_path: Path) -> None:
    text = "电话13812345678"
    phone = {"type": "phone", "text": "13812345678", "normalized_text": "13812345678", "start": 2, "end": 13, "confidence": 0.99, "source": "rule"}
    processor = DocumentProcessor(ocr_service=FakeOcr([block(text)]), detector=FakeDetector([phone]), redactor=RedactionService(tmp_path))
    result = processor.process_image(Image.new("RGB", (200, 60), "white"), redaction_mode="black")
    assert result["ocr_text"] == text
    assert result["entities"][0]["mapping_quality"] == "exact"
    assert Path(result["result_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert set(result["processing_time"]) == {"preprocess", "ocr", "detection", "mapping", "redaction", "total"}


def test_no_text_image_still_returns_result(tmp_path: Path) -> None:
    processor = DocumentProcessor(ocr_service=FakeOcr([]), detector=FakeDetector([]), redactor=RedactionService(tmp_path))
    result = processor.process_image(Image.new("RGB", (30, 30), "white"))
    assert result["entities"] == []
    assert any("未识别到文字" in warning for warning in result["warnings"])
    assert Path(result["result_path"]).exists()


def test_corrupt_image_returns_clear_warning(tmp_path: Path) -> None:
    damaged = tmp_path / "broken.png"; damaged.write_bytes(b"not an image")
    processor = DocumentProcessor(ocr_service=FakeOcr([]), detector=FakeDetector([]), redactor=RedactionService(tmp_path))
    result = processor.process_image(damaged)
    assert result["result_path"] is None
    assert any("文档处理失败" in warning for warning in result["warnings"])


def test_output_does_not_overwrite_input_and_coarse_warning(tmp_path: Path) -> None:
    source = tmp_path / "source.png"; Image.new("RGB", (200, 60), "white").save(source)
    original = source.read_bytes()
    bad_span = {"type": "person", "text": "张三", "start": 100, "end": 102, "confidence": 0.8, "source": "context"}
    processor = DocumentProcessor(ocr_service=FakeOcr([block("联系人张三")]), detector=FakeDetector([bad_span]), redactor=RedactionService(tmp_path / "out"))
    result = processor.process_image(source)
    assert source.read_bytes() == original
    assert result["result_path"] != str(source)
    assert any("粗略坐标映射" in warning for warning in result["warnings"])
