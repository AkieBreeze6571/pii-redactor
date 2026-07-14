"""Fault-isolated batch processing with ZIP/JSON/CSV exports."""

from __future__ import annotations

import csv
import json
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable


class BatchService:
    def __init__(self, processor: Any, output_dir: str | Path = "data/outputs/batches", max_files: int = 20, max_file_size_mb: int = 20) -> None:
        self.processor = processor; self.output_dir = Path(output_dir); self.max_files = max_files; self.max_file_size = max_file_size_mb * 1024 * 1024

    def process(self, files: list[str | Path], enabled_types: set[str] | None, redaction_mode: str, thresholds: dict[str, float] | None = None, progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
        files = list(files or [])
        if len(files) > self.max_files: raise ValueError(f"一次最多处理 {self.max_files} 张图片")
        batch_dir = self.output_dir / f"batch_{int(time.time())}_{uuid.uuid4().hex[:8]}"; batch_dir.mkdir(parents=True, exist_ok=False)
        rows = []
        for index, value in enumerate(files, start=1):
            path = Path(value); row = {"filename": path.name, "status": "failed", "entity_count": 0, "result_path": None, "error": None}
            try:
                if not path.is_file(): raise FileNotFoundError("文件不存在")
                if path.stat().st_size > self.max_file_size: raise ValueError("文件超过大小限制")
                result = self.processor.process_image(path, enabled_types, redaction_mode, thresholds)
                if not result.get("result_path"): raise RuntimeError("处理未生成结果图片")
                row.update({"status": "success", "entity_count": len(result.get("entities", [])), "result_path": result["result_path"], "report_path": result.get("report_path")})
            except Exception as exc: row["error"] = str(exc)
            rows.append(row)
            if progress: progress(index / max(len(files), 1), f"已处理 {index}/{len(files)}")
        json_path = batch_dir / "batch_report.json"; csv_path = batch_dir / "batch_report.csv"; zip_path = batch_dir / "results.zip"
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "status", "entity_count", "result_path", "error"], extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(json_path, json_path.name); archive.write(csv_path, csv_path.name)
            for row in rows:
                for field in ("result_path", "report_path"):
                    path = Path(row[field]) if row.get(field) else None
                    if path and path.exists(): archive.write(path, f"{Path(row['filename']).stem}/{path.name}")
        return {"rows": rows, "success": sum(row["status"] == "success" for row in rows), "failed": sum(row["status"] == "failed" for row in rows), "zip_path": str(zip_path), "json_path": str(json_path), "csv_path": str(csv_path)}

    def cleanup(self, older_than_seconds: int = 86400) -> int:
        removed = 0; cutoff = time.time() - older_than_seconds
        if not self.output_dir.exists(): return 0
        for path in self.output_dir.glob("batch_*"):
            if path.is_dir() and path.stat().st_mtime < cutoff: shutil.rmtree(path); removed += 1
        return removed
