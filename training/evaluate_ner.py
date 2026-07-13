"""Evaluate a local NER model on a unified JSONL split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset import NerDataset, load_records
from training.train_ner import evaluate_model, load_labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("checkpoints/best"))
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    labels, label2id, id2label = load_labels(Path("configs/label_map.json"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    records = load_records(args.data_path); dataset = NerDataset(records, tokenizer, label2id, args.max_length)
    loss, metrics, _, _ = evaluate_model(model, DataLoader(dataset, batch_size=args.batch_size), device, id2label)
    print(json.dumps({"loss": loss, **metrics}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
