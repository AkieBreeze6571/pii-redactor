from __future__ import annotations

import gradio as gr

from ui.batch_tab import build_batch_tab
from ui.history_tab import build_history_tab
from ui.image_tab import build_image_tab
from ui.model_tab import build_model_tab
from ui.status_tab import build_status_tab
from ui.text_tab import build_text_tab


APP_CSS = """
.gradio-container { max-width: 1600px !important; }
.app-header { margin-bottom: 0.35rem !important; }
.main-workspace { gap: 1rem !important; align-items: flex-start !important; }
.panel-section { gap: 0.75rem !important; }
.status-bar {
  min-height: 2.75rem;
  padding: 0.65rem 0.85rem;
  border: 1px solid var(--border-color-primary);
  border-radius: var(--radius-lg);
  background: var(--background-fill-secondary);
}
.stable-image { min-height: 320px; }
.secondary-actions { align-items: center !important; }
@media (max-width: 900px) {
  .main-workspace { flex-direction: column !important; }
  .main-workspace > div { width: 100% !important; min-width: 0 !important; }
  .stable-image { min-height: 240px; }
}
"""


def create_app(processor, database, batch_service) -> gr.Blocks:
    with gr.Blocks(title="中文文档敏感信息脱敏", fill_width=True) as app:
        gr.Markdown("# 中文文档敏感信息脱敏", elem_classes=["app-header"])
        with gr.Tabs():
            with gr.Tab("图片脱敏"): build_image_tab(processor, database)
            with gr.Tab("文本检测"): build_text_tab(processor.detector)
            with gr.Tab("批量处理"): build_batch_tab(batch_service)
            with gr.Tab("历史记录"): build_history_tab(database)
            with gr.Tab("模型配置"): build_model_tab(processor)
            with gr.Tab("系统状态"): build_status_tab(processor, database)
    return app
