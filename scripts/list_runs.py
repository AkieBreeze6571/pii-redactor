"""List complete training runs and write comparison reports."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_utils import list_runs


def main() -> int:
    runs = list_runs(); reports = Path("reports"); reports.mkdir(exist_ok=True)
    (reports / "experiment_comparison.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    if runs:
        with (reports / "experiment_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(runs[0])); writer.writeheader(); writer.writerows(runs)
    print(json.dumps(runs, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
