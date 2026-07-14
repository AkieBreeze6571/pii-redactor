from __future__ import annotations

from functools import partial

import gradio as gr

from ui.helpers import ENTITY_HEADERS, HEAVY_CONCURRENCY_ID, begin_request, detect_text_ui_stream, failed_request, restore_button
from ui.image_tab import ALL_TYPES


def build_text_tab(detector) -> None:
    request_state = gr.State({})
    text = gr.Textbox(lines=8, label="待检测文本")
    with gr.Row(): enabled = gr.CheckboxGroup(ALL_TYPES, value=ALL_TYPES, label="检测类型"); threshold = gr.Slider(0, 1, 0.6, step=0.01, label="NER 阈值")
    run = gr.Button("检测", variant="primary")
    status = gr.Markdown("尚未检测。", elem_classes=["status-bar"])
    with gr.Row(): rules = gr.JSON(label="规则结果"); ner = gr.JSON(label="NER 结果"); context = gr.JSON(label="上下文结果")
    table = gr.Dataframe(headers=ENTITY_HEADERS, datatype=["bool", "str", "str", "str", "number", "str", "str", "str", "number", "str", "str", "str"], label="融合结果")
    highlighted = gr.HTML(label="高亮文本"); report = gr.File(label="下载 JSON")
    accepted = run.click(
        lambda: begin_request("检测"),
        outputs=[request_state, run, status],
        queue=False,
        trigger_mode="once",
        show_progress="hidden",
    )
    heavy = accepted.then(
        partial(detect_text_ui_stream, detector),
        [text, enabled, threshold, request_state],
        [rules, ner, context, table, highlighted, report, status],
        concurrency_id=HEAVY_CONCURRENCY_ID,
        concurrency_limit=1,
        trigger_mode="once",
        show_progress="minimal",
    )
    heavy.then(lambda: restore_button("检测"), outputs=run, queue=False, show_progress="hidden")
    heavy.failure(lambda: failed_request("检测"), outputs=[run, status], queue=False, show_progress="hidden")
