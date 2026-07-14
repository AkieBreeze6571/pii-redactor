"""End-to-end image sensitive-information detection and redaction."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from services.coordinate_mapper import CoordinateMapper
from services.hybrid_detector import HybridDetector
from services.image_service import preprocess_image
from services.ocr_service import OcrService, reconstruct_text
from services.redaction_service import RedactionService, safe_stem
from utils.logger import log_performance


LOGGER = logging.getLogger(__name__)


class DocumentProcessor:
    _init_count = 0
    _init_lock = threading.Lock()

    def __init__(self, config_path: str | Path = "configs/inference_config.yaml", ocr_service: OcrService | None = None, detector: HybridDetector | None = None, redactor: RedactionService | None = None) -> None:
        with self._init_lock:
            self.__class__._init_count += 1
        self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.ocr = ocr_service or OcrService(self.config["ocr"])
        self.detector = detector or HybridDetector()
        self.redactor = redactor or RedactionService()
        self.mapper = CoordinateMapper.from_config(self.config["mapping"])
        # The Gradio queue also serializes heavy events; this lock protects non-UI callers.
        self._pipeline_lock = threading.Lock()

    @classmethod
    def initialization_count(cls) -> int:
        return cls._init_count

    def process_image(
        self,
        image: Any,
        enabled_types: set[str] | None = None,
        redaction_mode: str | None = None,
        thresholds: dict[str, float] | None = None,
        *,
        request_id: str | None = None,
        performance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_id = request_id or uuid.uuid4().hex[:12]
        owns_performance = performance is None
        metrics = performance if performance is not None else {}
        stages = metrics.setdefault("stages", {})
        counters = metrics.setdefault("counters", {})
        details = metrics.setdefault("details", {})
        stages.setdefault("callback_wait", 0.0)
        for name in ("ocr_calls", "ner_calls", "mapping_calls", "redaction_calls"):
            counters.setdefault(name, 0)
        started = time.perf_counter(); timing = {}; warnings: list[str] = []
        output = {"original_path": str(image) if isinstance(image, (str, Path)) else None, "preview_path": None, "result_path": None, "mask_path": None, "ocr_text": "", "ocr_blocks": [], "entities": [], "warnings": warnings, "processing_time": timing}
        try:
            with self._pipeline_lock:
                phase = time.perf_counter()
                prepared = preprocess_image(
                    image,
                    max_side=int(self.config["ocr"]["max_image_side"]),
                    stage_timings=stages,
                )
                timing["preprocess"] = time.perf_counter() - phase
                details["image_width"], details["image_height"] = prepared.original_size
                warnings.extend(prepared.warnings)

                phase = time.perf_counter(); counters["ocr_calls"] += 1
                blocks = self.ocr.recognize(prepared.image)
                timing["ocr"] = stages["ocr"] = time.perf_counter() - phase
                if prepared.scale_x != 1.0 or prepared.scale_y != 1.0:
                    for block in blocks:
                        block["polygon"] = [[x * prepared.scale_x, y * prepared.scale_y] for x, y in block["polygon"]]
                if self.ocr.last_warning: warnings.append(self.ocr.last_warning)
                rebuilt = reconstruct_text(blocks); output["ocr_text"] = rebuilt["full_text"]; output["ocr_blocks"] = rebuilt["blocks"]
                details["ocr_blocks"] = len(blocks)
                details["cache_hit"] = bool(getattr(self.ocr, "last_cache_hit", False))
                if not blocks: warnings.append("图片中未识别到文字。")

                phase = time.perf_counter()
                if isinstance(self.detector, HybridDetector):
                    detection = self.detector.detect_with_details(rebuilt["full_text"], enabled_types, thresholds)
                    entities = detection["final"]
                    stages.update(detection["timing"])
                    counters["ner_calls"] += int(detection["counts"]["ner_calls"])
                else:
                    entities = self.detector.detect(rebuilt["full_text"], enabled_types, thresholds)
                    stages.setdefault("ner", 0.0); stages.setdefault("fusion", 0.0)
                timing["detection"] = time.perf_counter() - phase
                stages.setdefault("detection", timing["detection"])

                phase = time.perf_counter(); counters["mapping_calls"] += 1
                mapped = self.mapper.map_entities(entities, rebuilt["blocks"], rebuilt["char_map"], prepared.original_size, prepared.original_image or prepared.image)
                timing["mapping"] = stages["coordinate_mapping"] = time.perf_counter() - phase
                for item in mapped:
                    item.setdefault("enabled", True)
                    if item["mapping_quality"] == "coarse" or item.get("mapping_strategy") == "full_block":
                        warnings.append(f"{item['type']} 实体为粗略坐标映射，已使用完整 OCR block 安全遮挡：{item.get('fallback_reason') or 'low_mapping_quality'}")
                    if item.get("mapping_warning") == "possible_partial_entity_coverage":
                        warnings.append("possible_partial_entity_coverage")
                output["entities"] = mapped; details["entities"] = len(mapped)

                phase = time.perf_counter(); counters["redaction_calls"] += 1
                mode = redaction_mode or self.config["redaction"]["default_mode"]
                filename = Path(image).name if isinstance(image, (str, Path)) else "uploaded.png"
                redaction_kwargs = {
                    "mosaic_block_size": int(self.config["redaction"]["mosaic_block_size"]),
                    "blur_kernel_size": int(self.config["redaction"]["blur_kernel_size"]),
                }
                if isinstance(self.redactor, RedactionService):
                    redaction_kwargs["stage_timings"] = stages
                paths = self.redactor.redact(prepared.original_image or prepared.image, mapped, mode, filename, **redaction_kwargs)
                timing["redaction"] = time.perf_counter() - phase
                stages.setdefault("redaction", timing["redaction"]); stages.setdefault("preview_generation", 0.0)
                output.update(paths)

                phase = time.perf_counter()
                report_path = Path(paths["result_path"]).with_suffix(".json")
                report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
                stages["json_generation"] = time.perf_counter() - phase
                output["report_path"] = str(report_path)
        except Exception as exc:
            LOGGER.error("Document processing failed request_id=%s error_type=%s", request_id, type(exc).__name__)
            warnings.append(f"文档处理失败：{type(exc).__name__}")
        timing["total"] = time.perf_counter() - started
        stages.setdefault("database_write", 0.0); stages.setdefault("frontend_payload_prepare", 0.0)
        stages["total"] = timing["total"]
        predictor = getattr(getattr(self.detector, "ner_detector", None), "predictor", None)
        details.setdefault("device", str(getattr(predictor, "device", "unavailable")))
        details.setdefault("ocr_device", getattr(self.ocr, "device_name", lambda: "unknown")())
        details.setdefault("model_reused", getattr(getattr(self.detector, "ner_detector", None), "initialization_count", lambda: 0)() == 1)
        details.setdefault("model_init_count", getattr(getattr(self.detector, "ner_detector", None), "initialization_count", lambda: 0)())
        details.setdefault("ocr_init_count", getattr(self.ocr, "initialization_count", lambda: 0)())
        if owns_performance:
            log_performance(LOGGER, request_id, stages, counters=counters, details=details)
        return output

    def regenerate(self, original_image: Any, entities: list[dict[str, Any]], redaction_mode: str, filename: str = "document.png") -> dict[str, str]:
        return self.redactor.redact(original_image, entities, redaction_mode, filename, mosaic_block_size=int(self.config["redaction"]["mosaic_block_size"]), blur_kernel_size=int(self.config["redaction"]["blur_kernel_size"]))
