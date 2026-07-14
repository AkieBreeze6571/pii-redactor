import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from services.document_processor import DocumentProcessor
from services.hybrid_detector import HybridDetector
from services.redaction_service import RedactionService


class FakeOcr:
    last_warning = ""

    def __init__(self, blocks: list[dict]) -> None:
        self.blocks = blocks
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
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
    assert result["entities"][0]["mapping_quality"] == "coarse"
    assert result["entities"][0]["mapping_strategy"] == "full_block"
    assert "possible_partial_entity_coverage" in result["warnings"]
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


def test_single_pipeline_has_one_call_per_heavy_stage(tmp_path: Path) -> None:
    class Rule:
        def detect(self, text): return []

    class Ner:
        model_source = "local_finetuned"
        predictor = None
        def __init__(self): self.calls = 0
        def detect(self, text, thresholds=None): self.calls += 1; return []
        @staticmethod
        def initialization_count(): return 1

    class Context:
        def enhance(self, text, entities): return entities
        def detect(self, text): return []

    ocr = FakeOcr([block("SYNTHETIC")]); ner = Ner()
    detector = HybridDetector(rule_detector=Rule(), ner_detector=ner, context_detector=Context())
    processor = DocumentProcessor(ocr_service=ocr, detector=detector, redactor=RedactionService(tmp_path))
    mapping_calls = 0; redaction_calls = 0
    original_map = processor.mapper.map_entities; original_redact = processor.redactor.redact

    def counted_map(*args, **kwargs):
        nonlocal mapping_calls
        mapping_calls += 1
        return original_map(*args, **kwargs)

    def counted_redact(*args, **kwargs):
        nonlocal redaction_calls
        redaction_calls += 1
        return original_redact(*args, **kwargs)

    processor.mapper.map_entities = counted_map
    processor.redactor.redact = counted_redact
    performance = {"stages": {}, "counters": {}, "details": {}}
    result = processor.process_image(Image.new("RGB", (240, 80), "white"), performance=performance)
    assert result["result_path"]
    assert ocr.calls == ner.calls == mapping_calls == redaction_calls == 1
    assert performance["counters"] == {"ocr_calls": 1, "ner_calls": 1, "mapping_calls": 1, "redaction_calls": 1}
    assert set(result["processing_time"]) == {"preprocess", "ocr", "detection", "mapping", "redaction", "total"}


def test_concurrent_pipeline_requests_are_serialized(tmp_path: Path) -> None:
    active = 0; maximum_active = 0; lock = threading.Lock()

    class SlowOcr(FakeOcr):
        def recognize(self, image):
            nonlocal active, maximum_active
            with lock:
                active += 1; maximum_active = max(maximum_active, active)
            try:
                time.sleep(0.04)
                return super().recognize(image)
            finally:
                with lock: active -= 1

    processor = DocumentProcessor(ocr_service=SlowOcr([]), detector=FakeDetector([]), redactor=RedactionService(tmp_path))
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: processor.process_image(Image.new("RGB", (40, 40), "white")), range(2)))
    assert maximum_active == 1
    assert all(result["result_path"] for result in results)
