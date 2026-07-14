"""Read training run metadata without loading model weights."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def read_run(path: Path) -> dict[str, Any] | None:
    required = [path / "config.yaml", path / "model" / "config.json", path / "tokenizer", path / "label_map.json", path / "validation_metrics.json", path / "test_metrics.json"]
    if not all(item.exists() for item in required): return None
    config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8")); validation = json.loads((path / "validation_metrics.json").read_text(encoding="utf-8")); test = json.loads((path / "test_metrics.json").read_text(encoding="utf-8")); runtime = config.get("runtime", {})
    return {
        "run_name": path.name, "model_name": config["model"]["pretrained_model_name"],
        "learning_rate": config["training"]["learning_rate"], "batch_size": config["training"]["batch_size"],
        "max_length": config["model"]["max_length"], "epochs": config["training"]["num_epochs"],
        "best_epoch": runtime.get("best_epoch"), "validation_macro_f1": validation.get("macro_f1"), "test_macro_f1": test.get("macro_f1"),
        "person_f1": test.get("by_type", {}).get("person", {}).get("f1"), "address_f1": test.get("by_type", {}).get("address", {}).get("f1"),
        "training_time": runtime.get("training_seconds"), "device": runtime.get("device"),
        "created_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
    }


def list_runs(root: Path = Path("checkpoints/runs")) -> list[dict[str, Any]]:
    return [value for path in sorted(root.glob("*")) if path.is_dir() and (value := read_run(path)) is not None]
