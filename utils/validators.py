"""Validation and normalization helpers for structured PII."""

from __future__ import annotations

import ipaddress
import re
from datetime import date


ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
ID_CHECK_CODES = "10X98765432"


def remove_separators(value: str, allow_hyphen: bool = False) -> str:
    pattern = r"[\s-]+" if allow_hyphen else r"\s+"
    return re.sub(pattern, "", value)


def validate_mainland_id(value: str) -> bool:
    """Validate an 18-character mainland China resident ID using MOD 11-2."""
    normalized = remove_separators(value).upper()
    if not re.fullmatch(r"\d{17}[\dX]", normalized):
        return False
    try:
        date(
            int(normalized[6:10]),
            int(normalized[10:12]),
            int(normalized[12:14]),
        )
    except ValueError:
        return False
    remainder = sum(int(number) * weight for number, weight in zip(normalized[:17], ID_WEIGHTS)) % 11
    return normalized[-1] == ID_CHECK_CODES[remainder]


def luhn_checksum(value: str) -> bool:
    """Validate a payment card number after removing spaces and hyphens."""
    normalized = remove_separators(value, allow_hyphen=True)
    if not normalized.isdigit() or not 16 <= len(normalized) <= 19:
        return False
    total = 0
    parity = len(normalized) % 2
    for index, character in enumerate(normalized):
        digit = int(character)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def validate_ip_address(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def normalize_phone(value: str) -> str:
    normalized = re.sub(r"[\s-]+", "", value)
    if normalized.startswith("+86"):
        normalized = normalized[3:]
    elif normalized.startswith("86") and len(normalized) == 13:
        normalized = normalized[2:]
    return normalized


def validate_phone(value: str) -> bool:
    return bool(re.fullmatch(r"1[3-9]\d{9}", normalize_phone(value)))
