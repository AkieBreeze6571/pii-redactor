"""UI handlers kept separate from Gradio component declarations."""

from __future__ import annotations

import html
import importlib.metadata
import json
import logging
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import torch
from PIL import Image

from utils.logger import log_performance


ENTITY_HEADERS = ["enabled", "type", "text", "normalized_text", "confidence", "source", "validation", "mapping_quality", "mapping_confidence", "mapping_strategy", "fallback_reason", "boxes"]
HEAVY_CONCURRENCY_ID = "pii-model-pipeline"
DISPLAY_MAX_SIDE = 1440
LOGGER = logging.getLogger(__name__)


def begin_request(button_label: str) -> tuple[dict[str, Any], Any, str]:
    """Immediately acknowledge a request before its queued heavy callback starts."""

    context = {"request_id": uuid.uuid4().hex[:12], "submitted_at": time.perf_counter()}
    return context, gr.update(value="处理中…", interactive=False), "已进入队列，等待处理…"


def restore_button(button_label: str) -> Any:
    return gr.update(value=button_label, interactive=True)


def failed_request(button_label: str) -> tuple[Any, str]:
    return restore_button(button_label), "处理失败，请查看日志中的 request_id。"


def make_display_image(path: str | Path | None, max_side: int = DISPLAY_MAX_SIDE) -> Image.Image | None:
    """Create a short-lived browser preview without changing the downloadable file."""

    if not path:
        return None
    with Image.open(path) as source:
        image = source.copy()
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def entities_to_rows(entities: list[dict[str, Any]]) -> list[list[Any]]:
    return [[item.get("enabled", True), item.get("type", ""), item.get("text", ""), item.get("normalized_text", item.get("text", "")), round(float(item.get("confidence", 0)), 4), item.get("source", ""), item.get("validation", "not_applicable"), item.get("mapping_quality", ""), round(float(item.get("mapping_confidence", 0)), 4), item.get("mapping_strategy", ""), item.get("fallback_reason") or "", json.dumps(item.get("boxes", []), ensure_ascii=False)] for item in entities]


def apply_table_edits(table: Any, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = table.values.tolist() if isinstance(table, pd.DataFrame) else table or []
    output = [dict(item) for item in entities]
    for index, row in enumerate(rows[:len(output)]):
        output[index]["enabled"] = bool(row[0]); output[index]["type"] = str(row[1])
    return output


def highlight_entities(text: str, entities: list[dict[str, Any]]) -> str:
    cursor = 0; parts = []
    for item in sorted(entities, key=lambda value: (value["start"], value["end"])):
        if item["start"] < cursor: continue
        parts.append(html.escape(text[cursor:item["start"]]))
        label = html.escape(item["type"]); value = html.escape(text[item["start"]:item["end"]])
        parts.append(f'<mark title="{label}">{value}<small> {label}</small></mark>'); cursor = item["end"]
    parts.append(html.escape(text[cursor:])); return "".join(parts).replace("\n", "<br>")


def process_image_ui(processor: Any, database: Any, image_path: str, enabled_types: list[str], mode: str, default_threshold: float, person: float, address: float, organization: float, horizontal_padding: float, vertical_padding: float, safety_mode: str = "严格模式", request_context: dict[str, Any] | None = None) -> tuple:
    if not image_path: return {}, None, None, None, [], "请先上传图片。", {}, None, None, None
    callback_started = time.perf_counter()
    context = request_context or {"request_id": uuid.uuid4().hex[:12], "submitted_at": callback_started}
    request_id = str(context.get("request_id") or uuid.uuid4().hex[:12])
    submitted_at = float(context.get("submitted_at") or callback_started)
    performance: dict[str, Any] = {
        "stages": {"callback_wait": max(0.0, callback_started - submitted_at)},
        "counters": {},
        "details": {},
    }
    thresholds = {"person": person, "address": address, "organization": organization}
    for entity_type in processor.config["ner"].get("thresholds", {}):
        thresholds.setdefault(entity_type, default_threshold)
    processor.mapper.horizontal_padding = horizontal_padding
    processor.mapper.vertical_padding = vertical_padding
    processor.mapper.safety_mode = "strict" if safety_mode in {"严格模式", "strict"} else "balanced"
    result = processor.process_image(
        image_path,
        set(enabled_types or []),
        mode,
        thresholds,
        request_id=request_id,
        performance=performance,
    )

    database_started = time.perf_counter()
    if result.get("result_path") and database.available:
        database.insert_document({"original_filename": Path(image_path).name, "original_path": image_path, "preview_path": result.get("preview_path"), "result_path": result.get("result_path"), "mask_path": result.get("mask_path"), "report_path": result.get("report_path"), "ocr_text": result.get("ocr_text"), "detected_entities": result.get("entities"), "redaction_mode": mode, "model_name": "chinese-macbert-base", "model_source": processor.detector.model_source, "processing_time": result.get("processing_time"), "warnings": result.get("warnings")})
    performance["stages"]["database_write"] = time.perf_counter() - database_started

    payload_started = time.perf_counter()
    state = {"original_path": image_path, "entities": result.get("entities", []), "mode": mode}
    preview_image = make_display_image(result.get("preview_path"))
    result_image = make_display_image(result.get("result_path"))
    mask_image = make_display_image(result.get("mask_path"))
    rows = entities_to_rows(result.get("entities", []))
    performance["stages"]["frontend_payload_prepare"] = time.perf_counter() - payload_started
    performance["stages"]["preview_generation"] = performance["stages"].get("preview_generation", 0.0) + performance["stages"]["frontend_payload_prepare"]
    performance["stages"]["total"] = time.perf_counter() - submitted_at
    log_performance(LOGGER, request_id, performance["stages"], counters=performance["counters"], details=performance["details"])
    return state, preview_image, result_image, mask_image, rows, "\n".join(result.get("warnings", [])), result.get("processing_time", {}), result.get("result_path"), result.get("report_path"), result.get("mask_path")


def process_image_ui_stream(processor: Any, database: Any, *args: Any):
    yield (*([gr.skip()] * 10), "处理中…")
    try:
        values = process_image_ui(processor, database, *args)
        warning = str(values[5] or "")
        if "请先上传" in warning:
            status = "请先上传图片。"
        else:
            status = "处理失败，请检查警告信息。" if "文档处理失败" in warning else "处理完成。"
        yield (*values, status)
    except Exception as exc:
        context = args[-1] if args and isinstance(args[-1], dict) else {}
        LOGGER.error("UI image callback failed request_id=%s error_type=%s", context.get("request_id", "unknown"), type(exc).__name__)
        yield ({}, None, None, None, [], "处理失败，请查看日志中的 request_id。", {}, None, None, None, "处理失败，请查看日志中的 request_id。")


def clear_image_ui() -> tuple:
    return {}, None, None, None, None, [], "", {}, None, None, None, "尚未处理。"


def regenerate_ui(processor: Any, state: dict, table: Any, mode: str) -> tuple:
    if not state or not state.get("original_path"): return state, None, None, "没有可重新生成的检测结果。"
    entities = apply_table_edits(table, state.get("entities", [])); paths = processor.regenerate(state["original_path"], entities, mode, Path(state["original_path"]).name)
    updated = dict(state); updated["entities"] = entities; updated["mode"] = mode
    return updated, paths["result_path"], paths["mask_path"], "已仅重新执行打码，未重复 OCR 或 NER。"


def regenerate_ui_for_display(processor: Any, state: dict, table: Any, mode: str) -> tuple:
    updated, result_path, mask_path, message = regenerate_ui(processor, state, table, mode)
    return updated, make_display_image(result_path), make_display_image(mask_path), message, result_path, mask_path


def detect_text_ui(detector: Any, text: str, enabled_types: list[str], threshold: float, performance: dict[str, Any] | None = None) -> tuple:
    thresholds = {key: threshold for key in ("person", "address", "organization")}
    if hasattr(detector, "detect_with_details"):
        details = detector.detect_with_details(text, set(enabled_types or []), thresholds)
        rules, ner, context, final = (details[key] for key in ("rules", "ner", "context", "final"))
        if performance is not None:
            performance["stages"].update(details["timing"])
            performance["counters"].update(details["counts"])
    else:
        rules = detector.rule_detector.detect(text)
        ner = detector.ner_detector.detect(text, thresholds)
        context = detector.context_detector.detect(text)
        final = detector.detect(text, set(enabled_types or []), thresholds)
    json_started = time.perf_counter()
    output_dir = Path("data/outputs"); output_dir.mkdir(parents=True, exist_ok=True); report = output_dir / f"text_detection_{uuid.uuid4().hex[:10]}.json"
    report.write_text(json.dumps({"text": text, "rule": rules, "ner": ner, "context": context, "final": final}, ensure_ascii=False, indent=2), encoding="utf-8")
    if performance is not None:
        performance["stages"]["json_generation"] = time.perf_counter() - json_started
        performance["details"]["entities"] = len(final)
    return rules, ner, context, entities_to_rows(final), highlight_entities(text, final), str(report)


def detect_text_ui_stream(detector: Any, text: str, enabled_types: list[str], threshold: float, request_context: dict[str, Any]):
    yield (*([gr.skip()] * 6), "处理中…")
    started = time.perf_counter(); submitted_at = float(request_context.get("submitted_at") or started)
    request_id = str(request_context.get("request_id") or uuid.uuid4().hex[:12])
    performance = {"stages": {"callback_wait": max(0.0, started - submitted_at)}, "counters": {"ocr_calls": 0, "mapping_calls": 0, "redaction_calls": 0}, "details": {}}
    try:
        values = detect_text_ui(detector, text, enabled_types, threshold, performance)
        performance["stages"].setdefault("database_write", 0.0); performance["stages"].setdefault("frontend_payload_prepare", 0.0)
        performance["stages"]["total"] = time.perf_counter() - submitted_at
        ner_detector = getattr(detector, "ner_detector", None)
        performance["details"].update({
            "device": str(getattr(getattr(ner_detector, "predictor", None), "device", "unavailable")),
            "model_reused": getattr(ner_detector, "initialization_count", lambda: 0)() == 1,
            "model_init_count": getattr(ner_detector, "initialization_count", lambda: 0)(),
            "ocr_init_count": 0,
        })
        log_performance(LOGGER, request_id, performance["stages"], counters=performance["counters"], details=performance["details"])
        yield (*values, "检测完成。")
    except Exception as exc:
        LOGGER.error("UI text callback failed request_id=%s error_type=%s", request_id, type(exc).__name__)
        yield (None, None, None, [], "", None, "处理失败，请查看日志中的 request_id。")


def system_status(processor: Any, database: Any) -> dict[str, Any]:
    def version(package: str) -> str:
        try: return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: return "未安装"
    return {
        "python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无",
        "transformers": version("transformers"), "paddleocr": version("paddleocr"), "gradio": version("gradio"),
        "ocr_loaded": processor.ocr.__class__._engine is not None, "ner_loaded": processor.detector.ner_detector.predictor is not None,
        "ocr_init_count": processor.ocr.initialization_count(), "model_init_count": processor.detector.ner_detector.initialization_count(),
        "document_processor_init_count": processor.initialization_count(), "ocr_memory_cache_entries": processor.ocr.cache_size(),
        "model_source": processor.detector.model_source, "model_path": "checkpoints/best", "model_name": "hfl/chinese-macbert-base",
        "local_finetuned": processor.detector.model_source == "local_finetuned", "thresholds": processor.config["ner"].get("thresholds", {}),
        "database": "data/app.db", "database_available": database.available, "output_dir": "data/outputs",
    }
