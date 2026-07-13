from pathlib import Path

import yaml

from services.ner_detector import NerDetector


def test_missing_model_degrades_without_crashing(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/inference_config.yaml").read_text(encoding="utf-8"))
    config["ner"]["model_path"] = str(tmp_path / "missing")
    path = tmp_path / "inference.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    detector = NerDetector(path)
    assert detector.model_source == "rule_only"
    assert detector.detect("收件人张三") == []
    assert "降级" in detector.status_message


def test_local_best_model_exists() -> None:
    assert (Path("checkpoints/best") / "config.json").exists()
