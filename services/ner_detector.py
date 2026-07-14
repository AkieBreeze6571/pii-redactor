"""Cached local NER service with explicit rule-only degradation."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

from training.predict_ner import NerPredictor


LOGGER = logging.getLogger(__name__)


class NerDetector:
    _cache: dict[str, NerPredictor] = {}
    _lock = RLock()
    _model_init_count = 0

    def __init__(self, config_path: str | Path = "configs/inference_config.yaml", model_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        configured = model_path or self.config["ner"]["model_path"]
        self.model_path = Path(configured)
        self.model_source = "rule_only"
        self.status_message = "本地微调 NER 模型不存在，已降级为规则检测。"
        self.predictor: NerPredictor | None = None
        self.load()

    @classmethod
    def initialization_count(cls) -> int:
        return cls._model_init_count

    def load(self, force: bool = False) -> bool:
        if not self.model_path.is_dir() or not (self.model_path / "config.json").exists():
            return False
        key = str(self.model_path.resolve())
        try:
            with self._lock:
                if force:
                    self._cache.pop(key, None)
                if key not in self._cache:
                    self._cache[key] = NerPredictor(
                        self.model_path,
                        stride=int(self.config["ner"].get("window_stride", 64)),
                    )
                    self.__class__._model_init_count += 1
                self.predictor = self._cache[key]
            self.model_source = "local_finetuned"
            self.status_message = "本地微调 NER 模型已加载。"
            return True
        except (OSError, ValueError, RuntimeError) as exc:
            LOGGER.error("NER model initialization failed error_type=%s", type(exc).__name__)
            self.status_message = f"NER 模型加载失败，已降级为规则检测：{type(exc).__name__}"
            self.model_source = "rule_only"
            self.predictor = None
            return False

    def detect(self, text: str, thresholds: dict[str, float] | None = None) -> list[dict[str, Any]]:
        if self.predictor is None or not text:
            return []
        configured = dict(self.config["ner"].get("thresholds", {}))
        if thresholds:
            configured.update(thresholds)
        return self.predictor.predict(text, configured, float(self.config["ner"].get("default_threshold", 0.6)))

    def reload(self, model_path: str | Path | None = None) -> bool:
        if model_path is not None:
            self.model_path = Path(model_path)
        return self.load(force=True)
