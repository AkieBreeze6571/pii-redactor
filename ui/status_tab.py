from __future__ import annotations

import gradio as gr

from ui.helpers import system_status


def build_status_tab(processor, database) -> None:
    status = gr.JSON(value=lambda: system_status(processor, database), label="系统状态")
    refresh = gr.Button("刷新状态"); refresh.click(lambda: system_status(processor, database), outputs=status, queue=False, show_progress="hidden")
