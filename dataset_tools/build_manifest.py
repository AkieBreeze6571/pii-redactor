"""Build normalized image-test and MultiPriv metadata-only manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build_image_test(source_path: Path) -> list[dict]:
    rows = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        rows.append({
            "id": raw["id"], "source": f"pii_bench_zh_{raw['source']}_generated",
            "license": "Apache-2.0", "annotation_type": "bounding_box",
            "pseudo_label": False, "image_path": raw["image"], "text": raw["text"],
            "entities": raw.get("entities", []), "boxes": [], "template": raw.get("template"),
            "split": "image_test", "generation_seed": 42,
        })
    return rows


def build_multipriv(root: Path) -> list[dict]:
    rows = []
    for image_path in sorted(root.glob("VLM/individual-level/**/*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        metadata_files = sorted(image_path.parent.glob("*.json"))
        rows.append({
            "id": f"multipriv_{image_path.parent.name}_{image_path.stem}",
            "source": "multipriv", "license": "CC-BY-NC-SA-4.0",
            "annotation_type": "metadata_only", "pseudo_label": False,
            "image_path": image_path.as_posix(), "text": "", "entities": [], "boxes": [],
            "metadata_path": metadata_files[0].as_posix() if metadata_files else None,
            "split": "external_test", "usage_restriction": "non-commercial",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-annotations", type=Path, default=Path("data/generated/annotations_test.jsonl"))
    parser.add_argument("--multipriv-root", type=Path, default=Path("data/raw/multipriv"))
    parser.add_argument("--image-output", type=Path, default=Path("data/annotations/image_test.jsonl"))
    parser.add_argument("--external-output", type=Path, default=Path("data/processed/external_test.jsonl"))
    args = parser.parse_args()
    image_rows = build_image_test(args.generated_annotations) if args.generated_annotations.exists() else []
    external_rows = build_multipriv(args.multipriv_root) if args.multipriv_root.exists() else []
    write_jsonl(args.image_output, image_rows); write_jsonl(args.external_output, external_rows)
    print(json.dumps({"image_test": len(image_rows), "external_test": len(external_rows), "multipriv_annotation_type": "metadata_only"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
