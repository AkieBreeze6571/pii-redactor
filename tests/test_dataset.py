import argparse
import hashlib
import json
from pathlib import Path

import yaml

from training.prepare_ner_data import (
    deduplicate,
    load_jsonl,
    load_label_map,
    run,
    split_records,
    template_fingerprint,
    text_fingerprint,
)


def test_reads_local_formal_and_chat() -> None:
    formal = load_jsonl(Path("data/raw/formal.jsonl"))
    chat = load_jsonl(Path("data/raw/chat.jsonl"))
    assert len(formal) == 5000
    assert len(chat) == 3000
    assert set(formal[0]) == {"id", "text", "lang", "entities"}
    assert set(chat[0]) == {"id", "text", "lang", "entities"}


def test_all_local_entity_spans_are_valid() -> None:
    for path in (Path("data/raw/formal.jsonl"), Path("data/raw/chat.jsonl")):
        for record in load_jsonl(path):
            assert all(
                record["text"][entity["start"]:entity["end"]] == entity["text"]
                for entity in record["entities"]
            )


def test_label_aliases() -> None:
    aliases, targets = load_label_map(Path("configs/label_map.json"))
    assert aliases["PER"] == "person"
    assert aliases["COMPANY"] == "organization"
    assert {"person", "address", "organization"} <= targets


def sample_record(identifier: str, text: str, entities: list[dict] | None = None) -> dict:
    entities = entities or []
    return {
        "id": identifier,
        "source": "test",
        "license": "test",
        "split": "",
        "text": text,
        "entities": entities,
        "pseudo_label": False,
        "text_sha256": text_fingerprint(text),
        "template_sha256": template_fingerprint(text, entities),
    }


def test_exact_text_dedup_prefers_more_complete_annotations() -> None:
    sparse = sample_record("a", "Alice", [])
    complete = sample_record("b", "Alice", [{"type": "person", "text": "Alice", "start": 0, "end": 5}])
    conflicts: list[dict] = []
    records, duplicate_count, conflict_count = deduplicate([sparse, complete], conflicts)
    assert records == [complete]
    assert duplicate_count == 1
    assert conflict_count == 1
    assert conflicts


def test_same_text_and_template_do_not_cross_splits() -> None:
    records = [sample_record(str(index), f"document {index}") for index in range(30)]
    duplicate = sample_record("duplicate", "document 1")
    records, _, _ = deduplicate(records + [duplicate], [])
    splits = split_records(records, {"train": 0.8, "validation": 0.1, "test_fixed": 0.1}, 42)
    fingerprints = [{item["text_sha256"] for item in splits[name]} for name in ("train", "validation", "test_fixed")]
    assert not (fingerprints[0] & fingerprints[1])
    assert not (fingerprints[0] & fingerprints[2])
    assert not (fingerprints[1] & fingerprints[2])


def write_fixture_jsonl(path: Path, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"id": f"{prefix}_{index}", "text": f"{prefix} document {index}", "entities": []}) + "\n" for index in range(20)),
        encoding="utf-8",
    )


def test_frozen_test_is_not_overwritten(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_fixture_jsonl(raw / "formal.jsonl", "formal")
    write_fixture_jsonl(raw / "chat.jsonl", "chat")
    config = yaml.safe_load(Path("configs/data_config.yaml").read_text(encoding="utf-8"))
    config["raw"]["search_root"] = str(raw)
    config["processed"]["output_dir"] = str(tmp_path / "processed")
    config["labels"]["map_path"] = str(Path("configs/label_map.json").resolve())
    config["reports"]["output_dir"] = str(tmp_path / "reports")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    args = argparse.Namespace(config=config_path, formal=None, chat=None, output_dir=None, force_resplit=False)

    run(args)
    test_path = tmp_path / "processed" / "test_fixed.jsonl"
    first_hash = hashlib.sha256(test_path.read_bytes()).hexdigest()
    second_summary = run(args)

    assert hashlib.sha256(test_path.read_bytes()).hexdigest() == first_hash
    assert second_summary["frozen_test_reused"] is True
