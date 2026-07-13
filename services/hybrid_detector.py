"""Fuse rule, local NER, and conservative context detections."""

from __future__ import annotations

from typing import Any

from services.context_detector import ContextDetector
from services.ner_detector import NerDetector
from services.rule_detector import RuleDetector


PRIORITY = {
    "id_number": 120, "bank_card": 110, "email": 100, "phone": 90,
    "license_plate": 80, "passport": 70, "address": 60, "person": 55,
    "company": 52, "school": 52, "hospital": 52, "organization": 50,
    "ip_address": 45, "url": 40, "qq_number": 35, "wechat_id": 30, "postal_code": 25,
}


def normalize_entity(item: dict[str, Any]) -> dict[str, Any]:
    value = dict(item)
    value.setdefault("normalized_text", value.get("text", ""))
    value.setdefault("confidence", 0.0)
    value.setdefault("source", "unknown")
    value.setdefault("sources", [value["source"]])
    value.setdefault("validation", "not_applicable")
    return value


def _rank(item: dict[str, Any]) -> tuple[int, float, int]:
    if item.get("validation") == "failed":
        return 0, float(item["confidence"]), item["end"] - item["start"]
    base = PRIORITY.get(item["type"], 1)
    if item["type"] in {"address", "person", "organization"} and "ner" not in item.get("sources", [item["source"]]):
        base -= 15
    return base, float(item["confidence"]), item["end"] - item["start"]


def fuse_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exact: dict[tuple[str, int, int], dict[str, Any]] = {}
    for raw in entities:
        item = normalize_entity(raw)
        key = (item["type"], item["start"], item["end"])
        if key in exact:
            previous = exact[key]
            previous["sources"] = sorted(set(previous["sources"]) | set(item["sources"]))
            previous["confidence"] = max(float(previous["confidence"]), float(item["confidence"]))
        else:
            exact[key] = item
    accepted = []
    for item in sorted(exact.values(), key=_rank, reverse=True):
        if any(item["start"] < kept["end"] and kept["start"] < item["end"] for kept in accepted):
            continue
        accepted.append(item)
    return sorted(accepted, key=lambda item: (item["start"], item["end"], item["type"]))


class HybridDetector:
    def __init__(self, rule_detector: RuleDetector | None = None, ner_detector: NerDetector | None = None, context_detector: ContextDetector | None = None) -> None:
        self.rule_detector = rule_detector or RuleDetector()
        self.ner_detector = ner_detector or NerDetector()
        self.context_detector = context_detector or ContextDetector()

    @property
    def model_source(self) -> str:
        return self.ner_detector.model_source

    def detect(self, text: str, enabled_types: set[str] | None = None, thresholds: dict[str, float] | None = None) -> list[dict[str, Any]]:
        if not text:
            return []
        rules = self.rule_detector.detect(text)
        ner_raw = self.ner_detector.detect(text, thresholds)
        if thresholds:
            ner_raw = [item for item in ner_raw if float(item.get("confidence", 0)) >= thresholds.get(item["type"], 0.0)]
        ner = self.context_detector.enhance(text, ner_raw)
        context = self.context_detector.detect(text)
        fused = fuse_entities(rules + ner + context)
        return [item for item in fused if enabled_types is None or item["type"] in enabled_types]
