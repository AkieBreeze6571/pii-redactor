"""Initialize services and launch the local Gradio application."""

from __future__ import annotations

import logging
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import torch

from services.batch_service import BatchService
from services.database_service import DatabaseService
from services.document_processor import DocumentProcessor
from services.hybrid_detector import HybridDetector
from services.ner_detector import NerDetector
from ui.app_ui import APP_CSS, create_app
from utils.logger import configure_logging


LOGGER = logging.getLogger(__name__)
QUEUE_MAX_SIZE = 16
DEFAULT_CONCURRENCY_LIMIT = 2
HEAVY_CONCURRENCY_LIMIT = 1
DEFAULT_MAX_THREADS = min(8, max(4, (os.cpu_count() or 4) // 2))


@dataclass(frozen=True)
class ApplicationServices:
    processor: DocumentProcessor
    database: DatabaseService
    batch: BatchService


_services: ApplicationServices | None = None
_services_lock = threading.RLock()


def get_application_services() -> ApplicationServices:
    """Create heavyweight application services once, including under concurrent startup."""

    global _services
    if _services is not None:
        return _services
    with _services_lock:
        if _services is None:
            data_dir = Path(os.getenv("PII_DATA_DIR", "data"))
            model_path = os.getenv("PII_MODEL_PATH")
            detector = HybridDetector(ner_detector=NerDetector(model_path=model_path)) if model_path else None
            processor = DocumentProcessor(detector=detector)
            processor.redactor.output_dir = data_dir / "outputs"
            database = DatabaseService(data_dir / "app.db")
            batch = BatchService(processor, data_dir / "outputs" / "batches")
            _services = ApplicationServices(processor, database, batch)
    return _services


def build_application():
    services = get_application_services()
    return create_app(services.processor, services.database, services.batch).queue(
        status_update_rate="auto",
        max_size=QUEUE_MAX_SIZE,
        default_concurrency_limit=DEFAULT_CONCURRENCY_LIMIT,
    )


def log_startup_diagnostics(startup_seconds: float) -> None:
    services = get_application_services()
    processor = services.processor
    expected = (Path.cwd() / ".venv" / "Scripts" / "python.exe").resolve()
    actual = Path(sys.executable).resolve()
    if os.path.normcase(str(actual)) != os.path.normcase(str(expected)):
        LOGGER.warning("unexpected_interpreter actual=%s expected=%s", actual, expected)
    cuda_available = torch.cuda.is_available()
    torch_device = f"cuda:{torch.cuda.current_device()}" if cuda_available else "cpu"
    predictor = processor.detector.ner_detector.predictor
    model_device = str(getattr(predictor, "device", "unavailable"))
    ocr_device = processor.ocr.device_name()
    LOGGER.info(
        "startup sys.executable=%s sys.prefix=%s python=%s working_directory=%s process_id=%d",
        sys.executable, sys.prefix, platform.python_version(), Path.cwd(), os.getpid(),
    )
    LOGGER.info(
        "startup torch_cuda_available=%s torch_device=%s model_device=%s ocr_backend=PaddleOCR "
        "ocr_device=%s gradio=%s startup_seconds=%.6f",
        cuda_available, torch_device, model_device, ocr_device, gr.__version__, startup_seconds,
    )
    LOGGER.info(
        "startup model_init_count=%d ocr_init_count=%d document_processor_init_count=%d "
        "queue_max_size=%d heavy_concurrency_limit=%d gradio_max_threads=%d",
        processor.detector.ner_detector.initialization_count(),
        processor.ocr.initialization_count(),
        processor.initialization_count(),
        QUEUE_MAX_SIZE,
        HEAVY_CONCURRENCY_LIMIT,
        int(os.getenv("PII_APP_MAX_THREADS", str(DEFAULT_MAX_THREADS))),
    )


def main() -> None:
    configure_logging("app.log")
    started = time.perf_counter()
    app = build_application()
    log_startup_diagnostics(time.perf_counter() - started)
    host = os.getenv("PII_APP_HOST", "127.0.0.1")
    port = int(os.getenv("PII_APP_PORT", "7860"))
    share = os.getenv("PII_APP_SHARE", "false").lower() in {"1", "true", "yes"}
    max_threads = int(os.getenv("PII_APP_MAX_THREADS", str(DEFAULT_MAX_THREADS)))
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        allowed_paths=[str(Path("data/outputs").resolve())],
        max_file_size="20mb",
        max_threads=max_threads,
        css=APP_CSS,
    )


if __name__ == "__main__": main()
