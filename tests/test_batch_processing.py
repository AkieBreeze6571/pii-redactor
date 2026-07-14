from pathlib import Path
from zipfile import ZipFile

import pytest

from services.batch_service import BatchService


class FakeProcessor:
    def __init__(self, output: Path) -> None: self.output = output

    def process_image(self, path, enabled_types, mode, thresholds):
        if "bad" in Path(path).name: raise RuntimeError("synthetic failure")
        result = self.output / f"{Path(path).stem}_result.png"; report = self.output / f"{Path(path).stem}.json"
        result.write_bytes(b"result"); report.write_text("{}", encoding="utf-8")
        return {"result_path": str(result), "report_path": str(report), "entities": [{"type": "phone"}]}


def test_batch_failure_is_isolated_and_exports(tmp_path: Path) -> None:
    good = tmp_path / "good.png"; bad = tmp_path / "bad.png"; good.write_bytes(b"a"); bad.write_bytes(b"b")
    service = BatchService(FakeProcessor(tmp_path), tmp_path / "batches")
    result = service.process([good, bad], None, "black")
    assert result["success"] == 1 and result["failed"] == 1
    assert Path(result["json_path"]).exists() and Path(result["csv_path"]).exists()
    with ZipFile(result["zip_path"]) as archive:
        assert "batch_report.json" in archive.namelist()
        assert any(name.endswith("_result.png") for name in archive.namelist())


def test_batch_limits_count_and_size(tmp_path: Path) -> None:
    service = BatchService(FakeProcessor(tmp_path), tmp_path / "batches", max_files=1, max_file_size_mb=0)
    file = tmp_path / "one.png"; file.write_bytes(b"x")
    with pytest.raises(ValueError): service.process([file, file], None, "black")
    result = service.process([file], None, "black")
    assert result["failed"] == 1


def test_cleanup_old_batches(tmp_path: Path) -> None:
    service = BatchService(FakeProcessor(tmp_path), tmp_path / "batches")
    old = service.output_dir / "batch_old"; old.mkdir(parents=True)
    assert service.cleanup(older_than_seconds=-1) == 1
