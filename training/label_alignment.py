"""Align character spans with fast-tokenizer offsets using BIO labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AlignmentStats:
    aligned_entities: int = 0
    ignored_labels: int = 0
    truncated_entities: int = 0
    zero_length_tokens: int = 0


def align_entities_to_tokens(
    offsets: list[tuple[int, int]],
    entities: list[dict[str, Any]],
    label2id: dict[str, int],
) -> tuple[list[int], AlignmentStats]:
    """Return token labels; unsupported entity types are intentionally ignored."""
    stats = AlignmentStats()
    labels = [label2id["O"] if end > start else -100 for start, end in offsets]
    stats.zero_length_tokens = sum(start == end for start, end in offsets)
    supported = []
    for entity in sorted(entities, key=lambda item: (item["start"], item["end"])):
        entity_type = str(entity["type"]).upper()
        if f"B-{entity_type}" not in label2id or f"I-{entity_type}" not in label2id:
            stats.ignored_labels += 1
            continue
        if supported and entity["start"] < supported[-1]["end"]:
            raise ValueError("Overlapping supervised entities are not supported")
        supported.append(entity)

    for entity in supported:
        token_indexes = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and start < entity["end"] and end > entity["start"]
        ]
        fully_visible = (
            token_indexes
            and offsets[token_indexes[0]][0] <= entity["start"]
            and offsets[token_indexes[-1]][1] >= entity["end"]
        )
        if not fully_visible:
            stats.truncated_entities += 1
            continue
        entity_type = str(entity["type"]).upper()
        for position, token_index in enumerate(token_indexes):
            prefix = "B" if position == 0 else "I"
            labels[token_index] = label2id[f"{prefix}-{entity_type}"]
        stats.aligned_entities += 1
    return labels, stats
