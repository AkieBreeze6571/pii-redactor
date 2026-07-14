"""Central logging configuration with lightweight PII redaction."""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping


_PHONE = re.compile(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)")
_ID_NUMBER = re.compile(r"(?<![0-9Xx])(\d{3})\d{11}(\d{3}[0-9Xx])(?![0-9Xx])")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_NAME_FIELD = re.compile(r"((?:姓名|收件人|联系人)\s*[:：]?\s*)([\u4e00-\u9fff]{2,4})")
_ADDRESS_FIELD = re.compile(r"((?:地址|住址)\s*[:：]?\s*)([\u4e00-\u9fff]{6,})")

PERFORMANCE_STAGES = (
    "callback_wait",
    "image_decode",
    "image_preprocess",
    "ocr",
    "detection",
    "ner",
    "fusion",
    "coordinate_mapping",
    "redaction",
    "preview_generation",
    "json_generation",
    "database_write",
    "frontend_payload_prepare",
    "total",
)


def redact_message(message: str) -> str:
    """Mask common PII while retaining enough context for diagnostics."""

    message = _PHONE.sub(r"\1****\3", message)
    message = _ID_NUMBER.sub(r"\1***********\2", message)
    message = _EMAIL.sub("[EMAIL]", message)
    message = _NAME_FIELD.sub(lambda match: f"{match.group(1)}{match.group(2)[0]}*", message)
    message = _ADDRESS_FIELD.sub(lambda match: f"{match.group(1)}{match.group(2)[:6]}****", message)
    return message


class PrivacyFilter(logging.Filter):
    """Prevent common sensitive values from being written to log files."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_message(record.getMessage())
        record.args = ()
        return True


def configure_logging(filename: str, level: int = logging.INFO) -> Path:
    """Configure console and size-limited file logging once per process."""

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    privacy_filter = PrivacyFilter()

    root = logging.getLogger()
    root.setLevel(level)
    if not any(getattr(handler, "baseFilename", None) == str(log_path.resolve()) for handler in root.handlers):
        file_handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(privacy_filter)
        root.addHandler(file_handler)
    if not any(type(handler) is logging.StreamHandler for handler in root.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(privacy_filter)
        root.addHandler(console_handler)
    return log_path


def log_performance(
    logger: logging.Logger,
    request_id: str,
    stages: Mapping[str, float],
    *,
    counters: Mapping[str, int] | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    """Write fixed-shape timing diagnostics without OCR text or entity values."""

    for stage in PERFORMANCE_STAGES:
        logger.info(
            "timing request_id=%s stage=%s seconds=%.6f",
            request_id,
            stage,
            max(0.0, float(stages.get(stage, 0.0))),
        )
    counts = counters or {}
    logger.info(
        "pipeline_calls request_id=%s ocr_calls=%d ner_calls=%d mapping_calls=%d redaction_calls=%d",
        request_id,
        int(counts.get("ocr_calls", 0)),
        int(counts.get("ner_calls", 0)),
        int(counts.get("mapping_calls", 0)),
        int(counts.get("redaction_calls", 0)),
    )
    safe = details or {}
    logger.info(
        "request_summary request_id=%s image_width=%d image_height=%d ocr_blocks=%d entities=%d "
        "device=%s ocr_device=%s model_reused=%s cache_hit=%s model_init_count=%d ocr_init_count=%d",
        request_id,
        int(safe.get("image_width", 0)),
        int(safe.get("image_height", 0)),
        int(safe.get("ocr_blocks", 0)),
        int(safe.get("entities", 0)),
        str(safe.get("device", "unknown")),
        str(safe.get("ocr_device", "unknown")),
        bool(safe.get("model_reused", False)),
        bool(safe.get("cache_hit", False)),
        int(safe.get("model_init_count", 0)),
        int(safe.get("ocr_init_count", 0)),
    )
