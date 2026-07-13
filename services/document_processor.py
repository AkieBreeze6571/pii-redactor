"""End-to-end image sensitive-information detection and redaction."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from services.coordinate_mapper import CoordinateMapper
from services.hybrid_detector import HybridDetector
from services.image_service import preprocess_image
from services.ocr_service import OcrService, reconstruct_text
from services.redaction_service import RedactionService, safe_stem


LOGGER = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, config_path: str | Path = "configs/inference_config.yaml", ocr_service: OcrService | None = None, detector: HybridDetector | None = None, redactor: RedactionService | None = None) -> None:
        self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.ocr = ocr_service or OcrService(self.config["ocr"])
        self.detector = detector or HybridDetector()
        self.redactor = redactor or RedactionService()
        mapping = self.config["mapping"]
        self.mapper = CoordinateMapper(mapping["horizontal_padding"], mapping["vertical_padding"])

    def process_image(self, image: Any, enabled_types: set[str] | None = None, redaction_mode: str | None = None, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
        started = time.perf_counter(); timing = {}; warnings: list[str] = []
        output = {"original_path": str(image) if isinstance(image, (str, Path)) else None, "preview_path": None, "result_path": None, "mask_path": None, "ocr_text": "", "ocr_blocks": [], "entities": [], "warnings": warnings, "processing_time": timing}
        try:
            phase = time.perf_counter(); prepared = preprocess_image(image, max_side=int(self.config["ocr"]["max_image_side"])); timing["preprocess"] = time.perf_counter() - phase; warnings.extend(prepared.warnings)
            phase = time.perf_counter(); blocks = self.ocr.recognize(prepared.image); timing["ocr"] = time.perf_counter() - phase
            if prepared.scale_x != 1.0 or prepared.scale_y != 1.0:
                for block in blocks:
                    block["polygon"] = [[x * prepared.scale_x, y * prepared.scale_y] for x, y in block["polygon"]]
            if self.ocr.last_warning: warnings.append(self.ocr.last_warning)
            rebuilt = reconstruct_text(blocks); output["ocr_text"] = rebuilt["full_text"]; output["ocr_blocks"] = rebuilt["blocks"]
            if not blocks: warnings.append("图片中未识别到文字。")
            phase = time.perf_counter(); entities = self.detector.detect(rebuilt["full_text"], enabled_types, thresholds); timing["detection"] = time.perf_counter() - phase
            phase = time.perf_counter(); mapped = self.mapper.map_entities(entities, rebuilt["blocks"], rebuilt["char_map"], prepared.original_size); timing["mapping"] = time.perf_counter() - phase
            for item in mapped:
                item.setdefault("enabled", True)
                if item["mapping_quality"] == "coarse": warnings.append(f"{item['type']} 实体只能进行粗略坐标映射，请人工复核。")
            output["entities"] = mapped
            phase = time.perf_counter(); mode = redaction_mode or self.config["redaction"]["default_mode"]
            filename = Path(image).name if isinstance(image, (str, Path)) else "uploaded.png"
            paths = self.redactor.redact(prepared.original_image or prepared.image, mapped, mode, filename, mosaic_block_size=int(self.config["redaction"]["mosaic_block_size"]), blur_kernel_size=int(self.config["redaction"]["blur_kernel_size"])); timing["redaction"] = time.perf_counter() - phase
            output.update(paths)
            report_path = Path(paths["result_path"]).with_suffix(".json")
            report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
            output["report_path"] = str(report_path)
        except Exception as exc:
            LOGGER.exception("Document processing failed")
            warnings.append(f"文档处理失败：{exc}")
        timing["total"] = time.perf_counter() - started
        return output

    def regenerate(self, original_image: Any, entities: list[dict[str, Any]], redaction_mode: str, filename: str = "document.png") -> dict[str, str]:
        return self.redactor.redact(original_image, entities, redaction_mode, filename, mosaic_block_size=int(self.config["redaction"]["mosaic_block_size"]), blur_kernel_size=int(self.config["redaction"]["blur_kernel_size"]))
