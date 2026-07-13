import pytest

from training.label_alignment import align_entities_to_tokens
from training.metrics import bio_to_spans, compute_entity_metrics


LABELS = ["O", "B-PERSON", "I-PERSON", "B-ADDRESS", "I-ADDRESS", "B-ORGANIZATION", "I-ORGANIZATION"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


def offsets(text: str, padded: int = 0) -> list[tuple[int, int]]:
    return [(0, 0)] + [(index, index + 1) for index in range(len(text))] + [(0, 0)] * (1 + padded)


def align(text: str, entities: list[dict], padded: int = 0) -> list[int]:
    return align_entities_to_tokens(offsets(text, padded), entities, LABEL2ID)[0]


def entity(entity_type: str, text: str, start: int, end: int) -> dict:
    return {"type": entity_type, "text": text[start:end], "start": start, "end": end}


def test_single_person() -> None:
    text = "联系张三"
    labels = align(text, [entity("person", text, 2, 4)])
    assert labels[3:5] == [LABEL2ID["B-PERSON"], LABEL2ID["I-PERSON"]]


def test_single_long_address() -> None:
    text = "地址四川省成都市武侯区"
    labels = align(text, [entity("address", text, 2, len(text))])
    assert labels[3] == LABEL2ID["B-ADDRESS"]
    assert all(value == LABEL2ID["I-ADDRESS"] for value in labels[4:-1])


def test_multiple_entities_and_punctuation() -> None:
    text = "张三，住北京"
    labels = align(text, [entity("person", text, 0, 2), entity("address", text, 4, 6)])
    assert labels[1:3] == [1, 2]
    assert labels[4] == LABEL2ID["O"]
    assert labels[5:7] == [3, 4]


def test_adjacent_entities() -> None:
    text = "张三北京"
    labels = align(text, [entity("person", text, 0, 2), entity("address", text, 2, 4)])
    assert labels[1:5] == [1, 2, 3, 4]


def test_entity_at_sentence_end() -> None:
    text = "现居北京"
    labels = align(text, [entity("address", text, 2, 4)])
    assert labels[-2] == LABEL2ID["I-ADDRESS"]


def test_wordpiece_partial_overlap() -> None:
    token_offsets = [(0, 0), (0, 2), (2, 4), (0, 0)]
    labels, _ = align_entities_to_tokens(token_offsets, [entity("person", "张三李四", 1, 4)], LABEL2ID)
    assert labels[1:3] == [LABEL2ID["B-PERSON"], LABEL2ID["I-PERSON"]]


def test_truncated_entity_is_fully_ignored() -> None:
    token_offsets = [(0, 0), (0, 1), (1, 2), (0, 0)]
    labels, stats = align_entities_to_tokens(token_offsets, [entity("address", "四川省", 0, 3)], LABEL2ID)
    assert labels == [-100, 0, 0, -100]
    assert stats.truncated_entities == 1


def test_special_and_padding_tokens_are_ignored() -> None:
    labels = align("张三", [entity("person", "张三", 0, 2)], padded=3)
    assert labels[0] == -100
    assert labels[-4:] == [-100, -100, -100, -100]


def test_unsupported_rule_label_is_ignored() -> None:
    labels, stats = align_entities_to_tokens(offsets("13812345678"), [entity("phone", "13812345678", 0, 11)], LABEL2ID)
    assert set(labels[1:-1]) == {LABEL2ID["O"]}
    assert stats.ignored_labels == 1


def test_overlapping_supervised_entities_raise() -> None:
    text = "张三北京"
    with pytest.raises(ValueError):
        align(text, [entity("person", text, 0, 3), entity("address", text, 2, 4)])


def test_entity_metrics_exact_and_partial() -> None:
    truth = [[-100, 1, 2, 0, 3, 4, -100]]
    prediction = [[0, 1, 2, 0, 3, 0, 0]]
    metrics = compute_entity_metrics(prediction, truth, ID2LABEL)
    assert metrics["exact_matches"] == 1
    assert metrics["partial_overlaps"] == 1
    assert metrics["by_type"]["person"]["f1"] == 1.0
