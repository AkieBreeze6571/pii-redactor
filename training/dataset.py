"""PyTorch dataset for unified character-span NER records."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from training.label_alignment import AlignmentStats, align_entities_to_tokens


def load_records(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    return records


class NerDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, label2id: dict[str, int], max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.stats = Counter()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        encoded = self.tokenizer(
            record["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
        )
        offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")]
        labels, stats = align_entities_to_tokens(offsets, record.get("entities", []), self.label2id)
        for key, value in vars(stats).items():
            self.stats[key] += value
        tensors = {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in encoded.items()
        }
        tensors["labels"] = torch.tensor(labels, dtype=torch.long)
        tensors["sample_index"] = torch.tensor(index, dtype=torch.long)
        return tensors


def count_bio_classes(records: list[dict[str, Any]], label2id: dict[str, int]) -> Counter[str]:
    supported = {label[2:].lower() for label in label2id if label.startswith("B-")}
    return Counter(entity["type"] for record in records for entity in record.get("entities", []) if entity["type"] in supported)
