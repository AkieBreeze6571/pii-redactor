import hashlib
from pathlib import Path

from services.database_service import DatabaseService


def record(path: Path, entity_type: str = "phone") -> dict:
    return {
        "original_filename": path.name, "original_path": str(path), "file_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "preview_path": "preview.png", "result_path": "result.png", "mask_path": "mask.png", "report_path": "report.json",
        "ocr_text": "synthetic", "detected_entities": [{"type": entity_type, "text": "masked"}], "redaction_mode": "black",
        "model_name": "test", "model_source": "rule_only", "processing_time": {"total": 0.1}, "warnings": ["fixture"],
    }


def test_initialize_insert_duplicate_and_json(tmp_path: Path) -> None:
    source = tmp_path / "sample.png"; source.write_bytes(b"fake")
    database = DatabaseService(tmp_path / "app.db")
    assert database.available
    first = database.insert_document(record(source)); second = database.insert_document(record(source))
    assert first == second
    rows = database.query_documents()
    assert len(rows) == 1
    assert rows[0]["detected_entities"][0]["type"] == "phone"
    assert rows[0]["processing_time"]["total"] == 0.1
    assert rows[0]["warnings"] == ["fixture"]


def test_pagination_search_filter_and_sort(tmp_path: Path) -> None:
    database = DatabaseService(tmp_path / "app.db")
    for index, entity_type in enumerate(("phone", "email", "phone")):
        source = tmp_path / f"document_{index}.png"; source.write_bytes(str(index).encode()); database.insert_document(record(source, entity_type))
    assert len(database.query_documents(page=1, page_size=2)) == 2
    assert len(database.query_documents(page=2, page_size=2)) == 1
    assert database.query_documents(filename="document_1")[0]["original_filename"] == "document_1.png"
    assert len(database.query_documents(entity_type="phone")) == 2


def test_delete_does_not_delete_source_by_default(tmp_path: Path) -> None:
    source = tmp_path / "source.png"; source.write_bytes(b"source")
    database = DatabaseService(tmp_path / "app.db"); document_id = database.insert_document(record(source))
    assert database.delete_document(document_id)
    assert source.exists()
    assert database.query_documents() == []


def test_unwritable_database_degrades(tmp_path: Path) -> None:
    directory = tmp_path / "as_database"; directory.mkdir()
    database = DatabaseService(directory)
    assert not database.available
    assert database.query_documents() == []
    assert "数据库初始化失败" in database.last_error
