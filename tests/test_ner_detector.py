import time
from concurrent.futures import ThreadPoolExecutor
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


def test_model_is_initialized_once_under_concurrency(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "model"; model_path.mkdir(); (model_path / "config.json").write_text("{}", encoding="utf-8")
    config = yaml.safe_load(Path("configs/inference_config.yaml").read_text(encoding="utf-8"))
    config["ner"]["model_path"] = str(model_path)
    config_path = tmp_path / "inference.yaml"; config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    created = []

    class FakePredictor:
        def __init__(self, path, stride):
            time.sleep(0.02)
            created.append((path, stride))

    monkeypatch.setattr("services.ner_detector.NerPredictor", FakePredictor)
    monkeypatch.setattr(NerDetector, "_cache", {})
    monkeypatch.setattr(NerDetector, "_model_init_count", 0)
    with ThreadPoolExecutor(max_workers=6) as executor:
        detectors = list(executor.map(lambda _: NerDetector(config_path), range(6)))
    assert len(created) == 1
    assert len({id(detector.predictor) for detector in detectors}) == 1
    assert NerDetector.initialization_count() == 1
