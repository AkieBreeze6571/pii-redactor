"""Context-aware rule detector for structured sensitive information."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

from utils.validators import (
    luhn_checksum,
    normalize_phone,
    remove_separators,
    validate_ip_address,
    validate_mainland_id,
    validate_phone,
)


PHONE_RE = re.compile(r"(?<!\d)(?:(?:\+?86)[ -]?)?1[3-9]\d[ -]?\d{4}[ -]?\d{4}(?!\d)")
ID_RE = re.compile(r"(?<![\dXx])\d(?: ?\d){16} ?[\dXx](?![\dXx])")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){15,18}\d(?!\d)")
EMAIL_RE = re.compile(r"(?<![A-Za-z0-9_.+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?![A-Za-z0-9_-])")
PASSPORT_RE = re.compile(r"(?<![A-Za-z0-9])(?:[EGPDS]\d{8}|E[A-Z]\d{7})(?![A-Za-z0-9])", re.IGNORECASE)
PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领"
PLATE_RE = re.compile(rf"[{PROVINCES}][A-HJ-NP-Z](?:[A-HJ-NP-Z0-9]{{5,6}})", re.IGNORECASE)
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6_RE = re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,}[0-9A-Fa-f:]{0,4}(?![0-9A-Fa-f:])")
URL_RE = re.compile(r"(?i)(?:https?://|www\.)[^\s<>\u3000\u3002\uff0c\uff1b\uff01\uff1f]+")
QQ_RE = re.compile(r"(?<!\d)[1-9]\d{4,11}(?!\d)")
WECHAT_RE = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9_-]{5,19}(?![A-Za-z0-9_-])")
POSTAL_RE = re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)")


CONTEXT = {
    "phone": re.compile(r"电话|手机号|手机|联系电话|TEL|Phone", re.IGNORECASE),
    "id_number": re.compile(r"身份证(?:号|号码)?|居民身份证", re.IGNORECASE),
    "bank_card": re.compile(r"银行卡|卡号|账户|账号", re.IGNORECASE),
    "passport": re.compile(r"护照|Passport|证件号码", re.IGNORECASE),
    "qq_number": re.compile(r"QQ|企鹅号", re.IGNORECASE),
    "wechat_id": re.compile(r"微信(?:号)?|WeChat", re.IGNORECASE),
    "postal_code": re.compile(r"邮编|邮政编码", re.IGNORECASE),
}

PRIORITY = {
    "id_number": 100,
    "bank_card": 90,
    "phone": 80,
    "email": 70,
    "license_plate": 60,
    "passport": 50,
    "ip_address": 40,
    "url": 30,
    "qq_number": 20,
    "wechat_id": 10,
    "postal_code": 5,
}


@dataclass(frozen=True)
class Candidate:
    type: str
    text: str
    normalized_text: str
    start: int
    end: int
    confidence: float
    validation: str = "passed"

    def as_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "source": "rule",
            "validation": self.validation,
        }


def _has_context(text: str, start: int, end: int, entity_type: str, radius: int = 16) -> bool:
    pattern = CONTEXT.get(entity_type)
    if pattern is None:
        return False
    return bool(pattern.search(text[max(0, start - radius):min(len(text), end + radius)]))


def _candidate(match: re.Match[str], entity_type: str, normalized: str | None = None, confidence: float = 0.95, validation: str = "passed") -> Candidate:
    return Candidate(entity_type, match.group(), normalized if normalized is not None else match.group(), match.start(), match.end(), confidence, validation)


class RuleDetector:
    def detect(self, text: str) -> list[dict[str, object]]:
        if not text:
            return []
        candidates: list[Candidate] = []
        candidates.extend(self._detect_ids(text))
        candidates.extend(self._detect_cards(text))
        candidates.extend(self._detect_phones(text))
        candidates.extend(_candidate(match, "email", confidence=0.99) for match in EMAIL_RE.finditer(text))
        candidates.extend(_candidate(match, "license_plate", match.group().upper(), 0.97) for match in PLATE_RE.finditer(text))
        candidates.extend(self._detect_passports(text))
        candidates.extend(self._detect_ips(text))
        candidates.extend(self._detect_urls(text))
        candidates.extend(self._detect_contextual(text, QQ_RE, "qq_number", 0.93))
        candidates.extend(self._detect_contextual(text, WECHAT_RE, "wechat_id", 0.92))
        candidates.extend(self._detect_contextual(text, POSTAL_RE, "postal_code", 0.92))
        return [candidate.as_dict() for candidate in resolve_conflicts(candidates)]

    def _detect_phones(self, text: str) -> Iterable[Candidate]:
        for match in PHONE_RE.finditer(text):
            if validate_phone(match.group()):
                confidence = 0.99 if _has_context(text, match.start(), match.end(), "phone") else 0.97
                yield _candidate(match, "phone", normalize_phone(match.group()), confidence)

    def _detect_ids(self, text: str) -> Iterable[Candidate]:
        for match in ID_RE.finditer(text):
            normalized = remove_separators(match.group()).upper()
            if validate_mainland_id(normalized):
                yield _candidate(match, "id_number", normalized, 0.995)
            elif _has_context(text, match.start(), match.end(), "id_number"):
                yield _candidate(match, "id_number", normalized, 0.45, "failed")

    def _detect_cards(self, text: str) -> Iterable[Candidate]:
        for match in CARD_RE.finditer(text):
            normalized = remove_separators(match.group(), allow_hyphen=True)
            if validate_mainland_id(normalized) or validate_phone(normalized):
                continue
            if luhn_checksum(normalized):
                yield _candidate(match, "bank_card", normalized, 0.99)
            elif _has_context(text, match.start(), match.end(), "bank_card"):
                yield _candidate(match, "bank_card", normalized, 0.4, "failed")

    def _detect_passports(self, text: str) -> Iterable[Candidate]:
        for match in PASSPORT_RE.finditer(text):
            if _has_context(text, match.start(), match.end(), "passport"):
                yield _candidate(match, "passport", match.group().upper(), 0.94)

    def _detect_ips(self, text: str) -> Iterable[Candidate]:
        for pattern in (IPV4_RE, IPV6_RE):
            for match in pattern.finditer(text):
                if validate_ip_address(match.group()):
                    yield _candidate(match, "ip_address", match.group().lower(), 0.99)

    def _detect_urls(self, text: str) -> Iterable[Candidate]:
        trailing = ".,;:!?)]}'\""
        for match in URL_RE.finditer(text):
            value = match.group().rstrip(trailing)
            if not value:
                continue
            yield Candidate("url", value, value, match.start(), match.start() + len(value), 0.98)

    def _detect_contextual(self, text: str, pattern: re.Pattern[str], entity_type: str, confidence: float) -> Iterable[Candidate]:
        for match in pattern.finditer(text):
            if entity_type == "wechat_id" and match.group().lower() == "wechat":
                continue
            if _has_context(text, match.start(), match.end(), entity_type):
                yield _candidate(match, entity_type, confidence=confidence)


def _priority(candidate: Candidate) -> tuple[int, float, int]:
    if candidate.validation != "passed":
        return 0, candidate.confidence, candidate.end - candidate.start
    return PRIORITY[candidate.type], candidate.confidence, candidate.end - candidate.start


def resolve_conflicts(candidates: Iterable[Candidate]) -> list[Candidate]:
    unique: dict[tuple[str, int, int, str], Candidate] = {}
    for candidate in candidates:
        key = (candidate.type, candidate.start, candidate.end, candidate.normalized_text)
        previous = unique.get(key)
        if previous is None or candidate.confidence > previous.confidence:
            unique[key] = candidate
    ranked = sorted(unique.values(), key=lambda item: _priority(item), reverse=True)
    accepted: list[Candidate] = []
    for candidate in ranked:
        if any(candidate.start < item.end and item.start < candidate.end for item in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda item: (item.start, item.end, item.type))


def detect_sensitive_info(text: str) -> list[dict[str, object]]:
    return RuleDetector().detect(text)
