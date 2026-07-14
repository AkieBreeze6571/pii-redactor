from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import gradio as gr
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app as app_module
from ui.app_ui import create_app
from ui.helpers import HEAVY_CONCURRENCY_ID, clear_image_ui


class FakeNer:
    predictor = None
    status_message = "ready"

    @staticmethod
    def reload(): return True

    @staticmethod
    def initialization_count(): return 1


class FakeOcr:
    _engine = None

    @staticmethod
    def initialization_count(): return 1

    @staticmethod
    def cache_size(): return 0


class FakeProcessor:
    def __init__(self, detector=None) -> None:
        self.mapper = SimpleNamespace(horizontal_padding=8, vertical_padding=5)
        self.detector = detector or SimpleNamespace(ner_detector=FakeNer(), model_source="local_finetuned")
        self.ocr = FakeOcr()
        self.redactor = SimpleNamespace(output_dir=None)
        self.config = {"ner": {"thresholds": {}}}

    @staticmethod
    def initialization_count(): return 1


class FakeDatabase:
    available = True
    last_error = ""

    def __init__(self, path=None) -> None: self.path = path
    def query_documents(self, **kwargs): return []
    def delete_document(self, *args): return True


class FakeBatch:
    def __init__(self, processor=None, output_dir=None) -> None: pass
    def process(self, *args, **kwargs):
        return {"rows": [], "success": 0, "failed": 0, "zip_path": None, "json_path": None, "csv_path": None}


def test_application_services_are_singleton_under_concurrent_startup(monkeypatch) -> None:
    created = 0

    class CountingProcessor(FakeProcessor):
        def __init__(self, detector=None) -> None:
            nonlocal created
            created += 1
            super().__init__(detector)

    monkeypatch.delenv("PII_MODEL_PATH", raising=False)
    monkeypatch.setattr(app_module, "_services", None)
    monkeypatch.setattr(app_module, "DocumentProcessor", CountingProcessor)
    monkeypatch.setattr(app_module, "DatabaseService", FakeDatabase)
    monkeypatch.setattr(app_module, "BatchService", FakeBatch)
    with ThreadPoolExecutor(max_workers=8) as executor:
        services = list(executor.map(lambda _: app_module.get_application_services(), range(8)))
    assert created == 1
    assert len({id(service) for service in services}) == 1
    assert len({id(service.processor) for service in services}) == 1


def _demo():
    demo = create_app(FakeProcessor(), FakeDatabase(), FakeBatch())
    return demo.queue(max_size=16, default_concurrency_limit=2)


def test_six_tabs_and_queue_policies_are_preserved() -> None:
    demo = _demo(); config = demo.get_config_file()
    tabs = [component["props"]["label"] for component in config["components"] if component.get("type") == "tabitem"]
    assert tabs == ["图片脱敏", "文本检测", "批量处理", "历史记录", "模型配置", "系统状态"]
    heavy = [block_fn for block_fn in demo.fns.values() if block_fn.concurrency_id == HEAVY_CONCURRENCY_ID]
    assert len(heavy) >= 5
    assert all(block_fn.concurrency_limit == 1 for block_fn in heavy)
    assert sum(inspect.isgeneratorfunction(block_fn.fn) for block_fn in heavy) >= 3
    clear_id = next(index for index, block_fn in demo.fns.items() if block_fn.fn is clear_image_ui)
    clear_dependency = next(item for item in config["dependencies"] if item["id"] == clear_id)
    assert clear_dependency["queue"] is False


def test_mounted_page_returns_http_200_and_closes_cleanly() -> None:
    mounted = gr.mount_gradio_app(FastAPI(), _demo(), path="/")
    with TestClient(mounted) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "中文文档敏感信息脱敏" in response.text
