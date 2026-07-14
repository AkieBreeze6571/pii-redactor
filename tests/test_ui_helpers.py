from pathlib import Path

import pandas as pd
from PIL import Image

from ui.helpers import apply_table_edits, entities_to_rows, failed_request, highlight_entities, make_display_image, regenerate_ui


def test_entity_table_edits_enabled_and_type() -> None:
    entities = [{"type": "person", "text": "张三", "start": 0, "end": 2, "confidence": 0.9, "source": "ner", "boxes": []}]
    rows = entities_to_rows(entities); rows[0][0] = False; rows[0][1] = "organization"
    updated = apply_table_edits(pd.DataFrame(rows), entities)
    assert updated[0]["enabled"] is False and updated[0]["type"] == "organization"
    assert len(rows[0]) == 12


def test_highlight_escapes_html() -> None:
    text = "<张三>"
    html = highlight_entities(text, [{"type": "person", "start": 1, "end": 3}])
    assert "&lt;" in html and "<mark" in html and "张三" in html


class FakeProcessor:
    def __init__(self): self.calls = 0
    def regenerate(self, original, entities, mode, filename):
        self.calls += 1; return {"result_path": "result.png", "mask_path": "mask.png"}


def test_regenerate_does_not_call_detection() -> None:
    processor = FakeProcessor(); state = {"original_path": "input.png", "entities": [{"type": "person", "enabled": True}]}
    updated, result, mask, message = regenerate_ui(processor, state, [[False, "person"]], "black")
    assert processor.calls == 1 and updated["entities"][0]["enabled"] is False
    assert result == "result.png" and "未重复 OCR" in message


def test_display_preview_is_bounded_without_changing_download(tmp_path: Path) -> None:
    source = tmp_path / "result.png"; Image.new("RGB", (3200, 1200), "white").save(source)
    original = source.read_bytes()
    preview = make_display_image(source, max_side=1440)
    assert preview is not None and max(preview.size) == 1440
    assert Image.open(source).size == (3200, 1200)
    assert source.read_bytes() == original


def test_failure_restores_button_state() -> None:
    button, message = failed_request("开始检测")
    assert button["interactive"] is True and button["value"] == "开始检测"
    assert "失败" in message
