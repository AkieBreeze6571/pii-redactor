"""Entity-level BIO metrics independent of token accuracy."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable


Span = tuple[str, int, int]


def bio_to_spans(labels: list[str]) -> list[Span]:
    spans: list[Span] = []
    current_type: str | None = None
    start = 0
    for index, label in enumerate(labels + ["O"]):
        if label.startswith("B-") or (label.startswith("I-") and label[2:] != current_type):
            if current_type is not None:
                spans.append((current_type, start, index))
            current_type = label[2:]
            start = index
        elif label == "O":
            if current_type is not None:
                spans.append((current_type, start, index))
                current_type = None
        elif label.startswith("I-") and current_type is None:
            current_type = label[2:]
            start = index
    return spans


def compute_entity_metrics(predictions: Iterable[list[int]], references: Iterable[list[int]], id2label: dict[int, str]) -> dict:
    per_type = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
    exact = partial = false_positive = missed = 0
    for pred_ids, ref_ids in zip(predictions, references):
        pairs = [(pred, ref) for pred, ref in zip(pred_ids, ref_ids) if ref != -100]
        pred_spans = set(bio_to_spans([id2label[pred] for pred, _ in pairs]))
        ref_spans = set(bio_to_spans([id2label[ref] for _, ref in pairs]))
        matches = pred_spans & ref_spans
        exact += len(matches)
        for entity_type, *_ in matches:
            per_type[entity_type]["tp"] += 1
        unmatched_pred = pred_spans - matches
        unmatched_ref = ref_spans - matches
        for span in unmatched_pred:
            per_type[span[0]]["fp"] += 1
        for span in unmatched_ref:
            per_type[span[0]]["fn"] += 1
        partial += sum(
            pred[1] < truth[2] and truth[1] < pred[2]
            for pred in unmatched_pred
            for truth in unmatched_ref
        )
        false_positive += len(unmatched_pred)
        missed += len(unmatched_ref)

    by_type = {}
    total = Counter(tp=0, fp=0, fn=0)
    for entity_type, counts in sorted(per_type.items()):
        total.update(counts)
        precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        by_type[entity_type.lower()] = {"precision": precision, "recall": recall, "f1": f1, **dict(counts)}
    micro_p = total["tp"] / max(total["tp"] + total["fp"], 1)
    micro_r = total["tp"] / max(total["tp"] + total["fn"], 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-12)
    macro_f1 = sum(item["f1"] for item in by_type.values()) / max(len(by_type), 1)
    return {
        "precision": micro_p, "recall": micro_r, "f1": micro_f1,
        "micro_f1": micro_f1, "macro_f1": macro_f1, "by_type": by_type,
        "exact_matches": exact, "partial_overlaps": partial,
        "missed": missed, "false_positives": false_positive,
    }
