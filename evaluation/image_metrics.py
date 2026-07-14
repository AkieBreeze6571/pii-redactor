"""Pure image/OCR metric helpers."""

from __future__ import annotations

from typing import Any


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, a in enumerate(left, 1):
        current = [row]
        for column, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (a != b)))
        previous = current
    return previous[-1]


def text_similarity(left: str, right: str) -> float:
    return max(0.0, 1 - levenshtein(left, right) / max(len(left), len(right), 1))


def rectangle(box: list[Any]) -> tuple[float, float, float, float]:
    if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
        return tuple(float(value) for value in box)
    xs = [float(point[0]) for point in box]; ys = [float(point[1]) for point in box]
    return min(xs), min(ys), max(xs), max(ys)


def iou(first: list[Any], second: list[Any]) -> float:
    ax1, ay1, ax2, ay2 = rectangle(first); bx1, by1, bx2, by2 = rectangle(second)
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - intersection
    return intersection / union if union else 0.0


def best_coverage(truth: list[Any], predicted: list[list[Any]]) -> float:
    return max((iou(truth, box) for box in predicted), default=0.0)
