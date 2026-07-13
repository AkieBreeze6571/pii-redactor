"""Run sliding-window inference with a locally fine-tuned NER model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForTokenClassification, AutoTokenizer


def merge_window_predictions(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, int, int], dict[str, Any]] = {}
    for item in candidates:
        key = (item["type"], item["start"], item["end"])
        if key not in unique or item["confidence"] > unique[key]["confidence"]:
            unique[key] = item
    return sorted(unique.values(), key=lambda item: (item["start"], item["end"], item["type"]))


class NerPredictor:
    def __init__(self, model_path: str | Path, max_length: int = 256, stride: int = 64, device: str | None = None) -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path).to(self.device).eval()
        self.max_length = max_length
        self.stride = stride
        self.id2label = {int(key): value for key, value in self.model.config.id2label.items()}

    def predict(self, text: str, thresholds: dict[str, float] | None = None, default_threshold: float = 0.6) -> list[dict[str, Any]]:
        if not text:
            return []
        encoded = self.tokenizer(
            text, max_length=self.max_length, truncation=True, stride=self.stride,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            padding=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            probabilities = self.model(**inputs).logits.softmax(dim=-1).cpu()
        candidates = []
        for window_index, window_offsets in enumerate(offsets.tolist()):
            active: dict[str, Any] | None = None
            for token_index, (start, end) in enumerate(window_offsets):
                if end <= start:
                    if active: candidates.append(active); active = None
                    continue
                confidence, label_id = probabilities[window_index, token_index].max(dim=-1)
                label = self.id2label[int(label_id)]
                if label == "O":
                    if active: candidates.append(active); active = None
                    continue
                prefix, entity_type = label.split("-", 1)
                entity_type = entity_type.lower()
                score = float(confidence)
                if prefix == "B" or active is None or active["type"] != entity_type or start > active["end"]:
                    if active: candidates.append(active)
                    active = {"type": entity_type, "start": start, "end": end, "scores": [score]}
                else:
                    active["end"] = end; active["scores"].append(score)
            if active: candidates.append(active)
        thresholds = thresholds or {}
        output = []
        for item in candidates:
            scores = item.pop("scores", [])
            confidence = sum(scores) / max(len(scores), 1)
            if confidence < thresholds.get(item["type"], default_threshold):
                continue
            item.update({"text": text[item["start"]:item["end"]], "confidence": confidence, "source": "ner"})
            output.append(item)
        return merge_window_predictions(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/inference_config.yaml"))
    parser.add_argument("--model-path")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8")); ner = config["ner"]
    model_path = Path(args.model_path or ner["model_path"])
    if not model_path.exists():
        print(json.dumps({"error": f"本地 NER 模型不存在：{model_path}", "entities": []}, ensure_ascii=False, indent=2)); return 2
    try:
        predictor = NerPredictor(model_path, stride=int(ner["window_stride"]))
        result = predictor.predict(args.text, ner.get("thresholds"), float(ner["default_threshold"]))
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": f"NER 推理失败：{exc}", "entities": []}, ensure_ascii=False, indent=2)); return 2
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
