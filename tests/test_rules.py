import pytest

from services.rule_detector import Candidate, RuleDetector, resolve_conflicts
from tests.test_validators import make_id


def by_type(text: str, entity_type: str) -> list[dict[str, object]]:
    return [item for item in RuleDetector().detect(text) if item["type"] == entity_type]


@pytest.mark.parametrize(
    ("value", "normalized"),
    [
        ("13812345678", "13812345678"),
        ("138 1234 5678", "13812345678"),
        ("138-1234-5678", "13812345678"),
        ("+86 13812345678", "13812345678"),
        ("+86-138-1234-5678", "13812345678"),
        ("86 138 1234 5678", "13812345678"),
    ],
)
def test_phone_formats(value: str, normalized: str) -> None:
    text = f"联系电话：{value}。"
    result = by_type(text, "phone")
    assert len(result) == 1
    assert result[0]["text"] == value
    assert result[0]["normalized_text"] == normalized
    assert text[result[0]["start"]:result[0]["end"]] == value


def test_phone_not_extracted_from_id() -> None:
    text = f"身份证号：{make_id()}"
    assert not by_type(text, "phone")


def test_valid_and_invalid_id_candidates() -> None:
    valid = make_id()
    passed = by_type(f"身份证号：{valid}", "id_number")
    assert passed and passed[0]["validation"] == "passed"
    invalid = valid[:-1] + ("0" if valid[-1] != "0" else "1")
    failed = by_type(f"身份证号：{invalid}", "id_number")
    assert failed and failed[0]["validation"] == "failed"


def test_bank_card_luhn_handling() -> None:
    valid = by_type("银行卡：4532 0151 1283 0366", "bank_card")
    assert valid and valid[0]["validation"] == "passed"
    invalid = by_type("银行卡：4532-0151-1283-0367", "bank_card")
    assert invalid and invalid[0]["validation"] == "failed"


def test_email_drops_trailing_punctuation() -> None:
    text = "邮箱test.user@example.com，备用。"
    result = by_type(text, "email")
    assert result[0]["text"] == "test.user@example.com"
    assert text[result[0]["start"]:result[0]["end"]] == result[0]["text"]


def test_passport_requires_context() -> None:
    assert by_type("护照号码E12345678", "passport")
    assert not by_type("订单E12345678已完成", "passport")


@pytest.mark.parametrize("plate", ["京A12345", "粤BD12345"])
def test_license_plates(plate: str) -> None:
    assert by_type(f"车牌：{plate}", "license_plate")[0]["normalized_text"] == plate


def test_ipv4_and_ipv6() -> None:
    assert by_type("服务器 192.168.1.1", "ip_address")
    assert by_type("IPv6 2001:db8::1", "ip_address")
    assert not by_type("服务器 999.999.999.999", "ip_address")


def test_url() -> None:
    result = by_type("访问 https://example.com/path。", "url")
    assert result[0]["text"] == "https://example.com/path"


def test_contextual_qq_wechat_and_postal_code() -> None:
    assert by_type("QQ：12345678", "qq_number")
    wechat = by_type("WeChat: wx_test01", "wechat_id")
    assert len(wechat) == 1 and wechat[0]["text"] == "wx_test01"
    assert by_type("邮政编码：100000", "postal_code")


def test_overlap_resolution_uses_priority_not_input_order() -> None:
    phone = Candidate("phone", "13812345678", "13812345678", 3, 14, 0.99)
    identity = Candidate("id_number", "x" * 18, "x" * 18, 0, 18, 0.99)
    assert resolve_conflicts([phone, identity]) == [identity]
    assert resolve_conflicts([identity, phone]) == [identity]


def test_empty_and_non_sensitive_text() -> None:
    detector = RuleDetector()
    assert detector.detect("") == []
    assert detector.detect("今天下午开会，请准时参加。") == []
