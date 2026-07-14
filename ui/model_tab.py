from __future__ import annotations

from pathlib import Path

import gradio as gr

from ui.helpers import HEAVY_CONCURRENCY_ID


def build_model_tab(processor) -> None:
    train_config = gr.Code(value=Path("configs/train_config.yaml").read_text(encoding="utf-8"), language="yaml", label="训练配置")
    inference_config = gr.Code(value=Path("configs/inference_config.yaml").read_text(encoding="utf-8"), language="yaml", label="推理配置")
    runs = gr.JSON(value=lambda: sorted(path.name for path in Path("checkpoints/runs").glob("*") if path.is_dir()), label="已有 Runs")
    reload_button = gr.Button("重新加载最佳模型"); message = gr.Textbox(label="模型状态")
    reload_button.click(
        lambda: processor.detector.ner_detector.status_message if processor.detector.ner_detector.reload() else processor.detector.ner_detector.status_message,
        outputs=message,
        concurrency_id=HEAVY_CONCURRENCY_ID,
        concurrency_limit=1,
        trigger_mode="once",
    )
