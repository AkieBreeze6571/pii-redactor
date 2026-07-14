"""Build normalized image-test and MultiPriv metadata-only manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build_image_test(source_path: Path, generator_path: Path = Path("generate_images.py")) -> list[dict]:
    rows = []
    generator_version = hashlib.sha256(generator_path.read_bytes()).hexdigest()[:16] if generator_path.exists() else "unknown"
    for index, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = json.loads(line)
        rows.append({
            "id": f"image_test_{index:06d}", "source": "pii_bench_zh_generated",
            "source_text_id": raw["id"], "source_partition": raw.get("source"),
            "license": "Apache-2.0", "annotation_type": "bounding_box", "pseudo_label": False,
            "split": "image_test_fixed", "template": raw.get("template"), "image_path": raw["image"],
            "width": raw.get("width"), "height": raw.get("height"), "text": raw["text"],
            "entities": raw.get("entities", []), "generation_seed": 42, "generator_version": generator_version,
        })
    return rows


def build_multipriv(root: Path) -> list[dict]:
    rows = []
    for image_path in sorted(root.glob("VLM/individual-level/**/*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        metadata_files = sorted(image_path.parent.glob("*.json"))
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError:
            width = height = None
        person_group = image_path.parent.name
        rows.append({
            "id": f"multipriv_{image_path.parent.name}_{image_path.stem}",
            "source": "multipriv", "license": "CC BY-NC-SA 4.0",
            "annotation_type": "metadata_only", "pseudo_label": False,
            "image_path": image_path.as_posix(), "text": "", "entities": [], "boxes": [],
            "metadata_path": metadata_files[0].as_posix() if metadata_files else None,
            "person_group": person_group, "width": width, "height": height,
            "split": "external_test", "usage_restriction": "non-commercial",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-annotations", type=Path, default=Path("data/generated/annotations_test.jsonl"))
    parser.add_argument("--multipriv-root", type=Path, default=Path("data/raw/multipriv"))
    parser.add_argument("--image-output", type=Path, default=Path("data/annotations/image_test_fixed.jsonl"))
    parser.add_argument("--multipriv-output", type=Path, default=Path("data/annotations/multipriv_manifest.jsonl"))
    parser.add_argument("--external-output", type=Path, default=Path("data/processed/external_test.jsonl"))
    args = parser.parse_args()
    image_rows = build_image_test(args.generated_annotations) if args.generated_annotations.exists() else []
    external_rows = build_multipriv(args.multipriv_root) if args.multipriv_root.exists() else []
    if args.image_output.exists():
        existing = [json.loads(line) for line in args.image_output.read_text(encoding="utf-8").splitlines() if line.strip()]
        if existing != image_rows:
            raise FileExistsError(f"固定图片测试集已存在且内容不同，拒绝覆盖：{args.image_output}")
    else:
        write_jsonl(args.image_output, image_rows)
    write_jsonl(args.multipriv_output, external_rows); write_jsonl(args.external_output, external_rows)
    print(json.dumps({"image_test": len(image_rows), "external_test": len(external_rows), "multipriv_annotation_type": "metadata_only"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
