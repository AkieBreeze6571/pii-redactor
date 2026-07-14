from __future__ import annotations

import gradio as gr

from ui.helpers import HEAVY_CONCURRENCY_ID, begin_request, failed_request, restore_button
from ui.image_tab import ALL_TYPES


def build_batch_tab(batch_service) -> None:
    request_state = gr.State({})
    files = gr.File(file_count="multiple", file_types=["image"], type="filepath", label="批量图片（最多 20 张）")
    with gr.Row(): enabled = gr.CheckboxGroup(ALL_TYPES, value=ALL_TYPES, label="检测类型"); mode = gr.Radio(["black", "mosaic", "blur"], value="black", label="打码方式")
    run = gr.Button("开始批量处理", variant="primary")
    status = gr.Markdown("尚未处理。", elem_classes=["status-bar"])
    table = gr.Dataframe(headers=["filename", "status", "entity_count", "result_path", "error"], label="批量结果")
    summary = gr.Textbox(label="处理汇总"); zip_file = gr.File(label="结果 ZIP"); json_file = gr.File(label="总报告 JSON"); csv_file = gr.File(label="总报告 CSV")
    def execute(values, types, selected_mode, progress=gr.Progress()):
        yield (*([gr.skip()] * 5), "处理中…")
        result = batch_service.process(values or [], set(types or []), selected_mode, progress=lambda value, desc: progress(value, desc=desc))
        rows = [[item.get(key) for key in ("filename", "status", "entity_count", "result_path", "error")] for item in result["rows"]]
        yield rows, f"成功 {result['success']}，失败 {result['failed']}", result["zip_path"], result["json_path"], result["csv_path"], "批量处理完成。"
    accepted = run.click(
        lambda: begin_request("开始批量处理"),
        outputs=[request_state, run, status],
        queue=False,
        trigger_mode="once",
        show_progress="hidden",
    )
    heavy = accepted.then(
        execute,
        [files, enabled, mode],
        [table, summary, zip_file, json_file, csv_file, status],
        concurrency_id=HEAVY_CONCURRENCY_ID,
        concurrency_limit=1,
        trigger_mode="once",
        show_progress="minimal",
    )
    heavy.then(lambda: restore_button("开始批量处理"), outputs=run, queue=False, show_progress="hidden")
    heavy.failure(lambda: failed_request("开始批量处理"), outputs=[run, status], queue=False, show_progress="hidden")
