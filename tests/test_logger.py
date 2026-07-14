import logging

from utils.logger import PrivacyFilter, redact_message


def test_redact_message_masks_common_sensitive_values() -> None:
    value = redact_message("联系人张三 手机13812345678 身份证11010519491231002X 地址四川省成都市武侯区人民南路 test@example.com")
    assert "张三" not in value
    assert "13812345678" not in value and "138****5678" in value
    assert "11010519491231002X" not in value and "110***********002X" in value
    assert "武侯区人民南路" not in value
    assert "test@example.com" not in value


def test_privacy_filter_replaces_formatted_record_message() -> None:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "电话 %s", ("13900001234",), None)
    assert PrivacyFilter().filter(record)
    assert record.getMessage() == "电话 139****1234"
