from utils.validators import (
    ID_CHECK_CODES,
    ID_WEIGHTS,
    luhn_checksum,
    validate_ip_address,
    validate_mainland_id,
)


def make_id(prefix17: str = "11010519900101123") -> str:
    remainder = sum(int(number) * weight for number, weight in zip(prefix17, ID_WEIGHTS)) % 11
    return prefix17 + ID_CHECK_CODES[remainder]


def test_valid_mainland_id() -> None:
    assert validate_mainland_id(make_id())


def test_invalid_id_checksum() -> None:
    valid = make_id()
    replacement = "0" if valid[-1] != "0" else "1"
    assert not validate_mainland_id(valid[:-1] + replacement)


def test_invalid_id_birth_date() -> None:
    assert not validate_mainland_id(make_id("11010519901301123"))


def test_id_allows_spaces() -> None:
    valid = make_id()
    assert validate_mainland_id(" ".join((valid[:6], valid[6:14], valid[14:])))


def test_luhn_valid_card() -> None:
    assert luhn_checksum("4532 0151 1283 0366")


def test_luhn_invalid_card() -> None:
    assert not luhn_checksum("4532-0151-1283-0367")


def test_ip_validation() -> None:
    assert validate_ip_address("192.168.1.1")
    assert validate_ip_address("2001:db8::1")
    assert not validate_ip_address("999.999.999.999")
