from services.context_detector import ContextDetector
from services.hybrid_detector import HybridDetector, fuse_entities
from services.rule_detector import RuleDetector


class FakeNer:
    model_source = "local_finetuned"

    def __init__(self, entities: list[dict]) -> None:
        self.entities = entities

    def detect(self, text: str, thresholds=None) -> list[dict]:
        return [dict(item) for item in self.entities]


def test_rule_and_ner_exact_duplicate_merge_sources() -> None:
    text = "电话13812345678"
    ner = FakeNer([{"type": "phone", "text": "13812345678", "start": 2, "end": 13, "confidence": 0.7, "source": "ner"}])
    result = HybridDetector(RuleDetector(), ner, ContextDetector()).detect(text)
    assert len(result) == 1
    assert result[0]["sources"] == ["ner", "rule"]


def test_valid_id_wins_over_contained_phone() -> None:
    identity = {"type": "id_number", "text": "x" * 18, "start": 0, "end": 18, "confidence": 0.99, "source": "rule", "validation": "passed"}
    phone = {"type": "phone", "text": "x" * 11, "start": 3, "end": 14, "confidence": 0.99, "source": "rule", "validation": "passed"}
    assert fuse_entities([phone, identity])[0]["type"] == "id_number"


def test_ner_address_wins_context_overlap() -> None:
    ner = {"type": "address", "text": "四川省成都市", "start": 3, "end": 10, "confidence": 0.95, "source": "ner"}
    context = {"type": "person", "text": "成都市", "start": 7, "end": 10, "confidence": 0.7, "source": "context"}
    assert fuse_entities([context, ner])[0]["type"] == "address"


def test_low_confidence_ner_filtered() -> None:
    ner = FakeNer([{"type": "person", "text": "张三", "start": 0, "end": 2, "confidence": 0.4, "source": "ner"}])
    result = HybridDetector(RuleDetector(), ner, ContextDetector()).detect("张三", thresholds={"person": 0.7})
    assert result == []


def test_enabled_types_filter() -> None:
    detector = HybridDetector(RuleDetector(), FakeNer([]), ContextDetector())
    result = detector.detect("电话13812345678，邮箱a@example.com", enabled_types={"email"})
    assert [item["type"] for item in result] == ["email"]
