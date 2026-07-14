from __future__ import annotations

import gradio as gr


def build_history_tab(database) -> None:
    with gr.Row(): filename = gr.Textbox(label="文件名搜索"); entity_type = gr.Textbox(label="实体类型"); refresh = gr.Button("刷新")
    table = gr.Dataframe(headers=["id", "original_filename", "created_at", "redaction_mode", "model_source", "result_path"], label="历史记录")
    def query(name, kind):
        return [[row.get(key) for key in ("id", "original_filename", "created_at", "redaction_mode", "model_source", "result_path")] for row in database.query_documents(filename=name or "", entity_type=kind or "")]
    refresh.click(query, [filename, entity_type], table, queue=False, show_progress="hidden")
    with gr.Row():
        document_id = gr.Number(label="记录 ID", precision=0)
        delete_files = gr.Checkbox(label="同时删除输出文件", value=False)
        delete_button = gr.Button("删除记录", variant="stop")
    delete_status = gr.Textbox(label="操作结果", interactive=False)

    def delete(document_id_value, remove_files):
        if document_id_value is None:
            return "请输入记录 ID。"
        success = database.delete_document(int(document_id_value), bool(remove_files))
        return "记录已删除。" if success else f"删除失败：{database.last_error or '记录不存在'}"

    delete_button.click(delete, [document_id, delete_files], delete_status, queue=False, show_progress="hidden")
