from services.context_detector import ContextDetector, classify_organization


def test_name_context_is_conservative() -> None:
    detector = ContextDetector()
    result = detector.detect("收件人：张三，电话另附。")
    assert result[0]["type"] == "person"
    assert result[0]["text"] == "张三"
    assert detector.detect("今天下午开会") == []


def test_address_context() -> None:
    result = ContextDetector().detect("家庭住址：四川省成都市武侯区人民南路四段。")
    assert result[0]["type"] == "address"
    assert result[0]["text"].endswith("四段")


def test_address_extension_does_not_cross_period() -> None:
    text = "地址四川省成都市武侯区人民南路四段。联系电话另附"
    original = [{"type": "address", "text": "四川省成都市", "start": 2, "end": 9, "confidence": 0.9, "source": "ner"}]
    result = ContextDetector().enhance(text, original)[0]
    assert result["text"].endswith("人民南路")
    assert "联系电话" not in result["text"]


def test_organization_subtypes() -> None:
    assert classify_organization("星河实验学校") == "school"
    assert classify_organization("星河人民医院") == "hospital"
    assert classify_organization("星河科技有限公司") == "company"
    assert classify_organization("星河研究中心") == "organization"


def test_refinement_keeps_original_type() -> None:
    item = {"type": "organization", "text": "星河大学", "start": 0, "end": 4, "confidence": 0.9, "source": "ner"}
    result = ContextDetector().enhance("星河大学", [item])[0]
    assert result["type"] == "school"
    assert result["original_type"] == "organization"
    assert result["sources"] == ["ner", "context"]
