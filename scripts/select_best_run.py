"""Validate and copy a run to checkpoints/best while retaining a backup."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_utils import read_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-name", required=True); args = parser.parse_args()
    run = Path("checkpoints/runs") / args.run_name; metadata = read_run(run)
    if metadata is None: print(f"Run 不完整或不存在：{run}", file=sys.stderr); return 2
    best = Path("checkpoints/best"); timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if best.exists():
        backup = Path("checkpoints/best_backups") / f"best_{timestamp}"; backup.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(best, backup)
    temporary = Path("checkpoints/best.tmp")
    if temporary.exists(): shutil.rmtree(temporary)
    shutil.copytree(run / "model", temporary)
    for path in (run / "tokenizer").iterdir():
        if path.is_file(): shutil.copy2(path, temporary / path.name)
    shutil.copy2(run / "label_map.json", temporary / "label_map.json"); shutil.copy2(run / "config.yaml", temporary / "train_config.yaml")
    metadata.update({"selected_at": datetime.now(timezone.utc).isoformat(), "selection_basis": "manual; validation metrics are primary"})
    (temporary / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if best.exists(): shutil.rmtree(best)
    temporary.replace(best); print(json.dumps(metadata, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
