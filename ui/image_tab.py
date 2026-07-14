from __future__ import annotations

from functools import partial

import gradio as gr

from ui.helpers import (
    ENTITY_HEADERS,
    HEAVY_CONCURRENCY_ID,
    begin_request,
    clear_image_ui,
    failed_request,
    process_image_ui_stream,
    regenerate_ui_for_display,
    restore_button,
)


ALL_TYPES = ["person", "address", "phone", "id_number", "email", "bank_card", "passport", "license_plate", "organization", "company", "school", "hospital", "ip_address", "url", "qq_number", "wechat_id", "postal_code"]


def build_image_tab(processor, database) -> None:
    state = gr.State({})
    request_state = gr.State({})
    with gr.Row(elem_classes=["main-workspace"]):
        with gr.Column(scale=5, min_width=360, elem_classes=["panel-section"]):
            upload = gr.Image(type="filepath", label="文档图片", sources=["upload", "clipboard"], height=360, elem_classes=["stable-image"])
            with gr.Accordion("检测与脱敏参数", open=True):
                enabled = gr.CheckboxGroup(ALL_TYPES, value=ALL_TYPES, label="检测类型")
                mode = gr.Radio(["black", "mosaic", "blur"], value="black", label="打码方式")
                default_threshold = gr.Slider(0, 1, 0.6, step=0.01, label="NER 总阈值")
                person = gr.Slider(0, 1, 0.7, step=0.01, label="姓名阈值")
                address = gr.Slider(0, 1, 0.55, step=0.01, label="地址阈值")
                organization = gr.Slider(0, 1, 0.65, step=0.01, label="组织阈值")
                horizontal_padding = gr.Slider(0, 30, processor.mapper.horizontal_padding, step=1, label="横向安全边距")
                vertical_padding = gr.Slider(0, 20, processor.mapper.vertical_padding, step=1, label="纵向安全边距")
                safety_mode = gr.Radio(["严格模式", "平衡模式"], value="严格模式", label="映射安全模式")
                gr.Markdown("严格模式在局部定位不可靠时扩大或整行遮挡；平衡模式优先减少多余遮挡，但需人工复核局部覆盖。")
            with gr.Row(elem_classes=["secondary-actions"]):
                run = gr.Button("开始检测", variant="primary", scale=3)
                clear = gr.Button("清空", variant="secondary", scale=1)
        with gr.Column(scale=7, min_width=420, elem_classes=["panel-section"]):
            status = gr.Markdown("尚未处理。", elem_classes=["status-bar"])
            with gr.Row():
                preview = gr.Image(type="pil", label="实体框预览", height=320, elem_classes=["stable-image"])
                result = gr.Image(type="pil", label="脱敏结果", height=320, elem_classes=["stable-image"])
            mask = gr.Image(type="pil", label="透明 Mask", height=260, elem_classes=["stable-image"])
    table = gr.Dataframe(headers=ENTITY_HEADERS, datatype=["bool", "str", "str", "str", "number", "str", "str", "str", "number", "str", "str", "str"], interactive=True, label="实体结果")
    regenerate = gr.Button("重新生成脱敏图", variant="secondary")
    warnings = gr.Textbox(label="警告", lines=3); timing = gr.JSON(label="处理耗时")
    with gr.Row(): result_file = gr.File(label="下载结果图"); report_file = gr.File(label="下载 JSON 报告"); mask_file = gr.File(label="下载 Mask")
    outputs = [state, preview, result, mask, table, warnings, timing, result_file, report_file, mask_file]
    accepted = run.click(
        lambda: begin_request("开始检测"),
        outputs=[request_state, run, status],
        queue=False,
        trigger_mode="once",
        show_progress="hidden",
    )
    heavy = accepted.then(
        partial(process_image_ui_stream, processor, database),
        [upload, enabled, mode, default_threshold, person, address, organization, horizontal_padding, vertical_padding, safety_mode, request_state],
        [*outputs, status],
        concurrency_id=HEAVY_CONCURRENCY_ID,
        concurrency_limit=1,
        trigger_mode="once",
        show_progress="minimal",
    )
    heavy.then(lambda: restore_button("开始检测"), outputs=run, queue=False, show_progress="hidden")
    heavy.failure(lambda: failed_request("开始检测"), outputs=[run, status], queue=False, show_progress="hidden")
    clear.click(
        clear_image_ui,
        outputs=[state, upload, preview, result, mask, table, warnings, timing, result_file, report_file, mask_file, status],
        queue=False,
        show_progress="hidden",
    )
    regenerate.click(
        lambda s, t, m: regenerate_ui_for_display(processor, s, t, m),
        [state, table, mode],
        [state, result, mask, warnings, result_file, mask_file],
        concurrency_id=HEAVY_CONCURRENCY_ID,
        concurrency_limit=1,
        trigger_mode="once",
    )
