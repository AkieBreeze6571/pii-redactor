"""Run only explicitly listed NER experiments; no implicit Cartesian product."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml")); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    data = yaml.safe_load(args.config.read_text(encoding="utf-8")); experiments = data.get("experiments", [])
    if not experiments: print("没有显式配置 experiments，未启动任何训练。") ; return 0
    for experiment in experiments:
        command = [sys.executable, "training/train_ner.py", "--config", str(data.get("train_config", "configs/train_config.yaml")), "--run-name", experiment["name"], "--pretrained-model-name", experiment["model"], "--learning-rate", str(experiment["learning_rate"]), "--batch-size", str(experiment["batch_size"]), "--max-length", str(experiment["max_length"])]
        if args.overwrite: command.append("--overwrite-run")
        result = subprocess.run(command, check=False)
        if result.returncode: return result.returncode
    return subprocess.run([sys.executable, "scripts/list_runs.py"], check=False).returncode


if __name__ == "__main__": raise SystemExit(main())
