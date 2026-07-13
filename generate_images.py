from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc")
]


def first_value(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def load_font(font_path: str | None, font_size: int) -> ImageFont.FreeTypeFont:
    candidates = []

    if font_path:
        candidates.append(Path(font_path))

    candidates.extend(FONT_CANDIDATES)

    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), font_size)

    raise FileNotFoundError(
        "没有找到中文字体，请通过 --font 指定字体文件路径"
    )


def load_jsonl(path: Path, source: str) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"跳过无法解析的行：{path}:{line_number}")
                continue

            row["_source"] = source
            rows.append(row)

    return rows


def normalize_entities(
    row: dict[str, Any],
    text: str
) -> list[dict[str, Any]]:
    raw_entities = first_value(
        row,
        ["entities", "labels", "annotations", "spans"]
    )

    if raw_entities is None:
        return []

    if isinstance(raw_entities, str):
        try:
            raw_entities = json.loads(raw_entities)
        except json.JSONDecodeError:
            return []

    entities = []

    for raw in raw_entities:
        start = first_value(
            raw,
            ["start", "start_offset", "begin", "offset_start"]
        )
        end = first_value(
            raw,
            ["end", "end_offset", "stop", "offset_end"]
        )
        entity_type = first_value(
            raw,
            ["type", "label", "entity_type", "category"]
        )
        entity_text = first_value(
            raw,
            ["text", "value", "entity", "span"]
        )

        if start is None or end is None or entity_type is None:
            continue

        try:
            start = int(start)
            end = int(end)
        except (TypeError, ValueError):
            continue

        if not 0 <= start < len(text):
            continue

        if end <= start or end > len(text):
            continue

        extracted = text[start:end]

        if entity_text is None:
            entity_text = extracted

        entity_text = str(entity_text)

        if extracted != entity_text:
            if end < len(text) and text[start:end + 1] == entity_text:
                end += 1
                extracted = text[start:end]
            else:
                continue

        entities.append({
            "type": str(entity_type),
            "text": extracted,
            "start": start,
            "end": end
        })

    return entities


def extract_sample(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    text = first_value(row, ["text", "content", "sentence", "document"])

    if text is None:
        return None

    text = str(text)

    if not text.strip():
        return None

    sample_id = first_value(row, ["id", "sample_id", "uid"])

    if sample_id is None:
        sample_id = f"sample_{index:06d}"

    entities = normalize_entities(row, text)

    return {
        "id": str(sample_id),
        "text": text,
        "entities": entities,
        "source": row["_source"]
    }


def choose_theme(source: str, rng: random.Random) -> dict[str, Any]:
    if source == "chat":
        themes = [
            {
                "name": "chat_light",
                "background": (239, 242, 247),
                "panel": (255, 255, 255),
                "text": (30, 30, 30),
                "border": (210, 215, 225)
            },
            {
                "name": "chat_green",
                "background": (231, 238, 232),
                "panel": (255, 255, 255),
                "text": (28, 28, 28),
                "border": (195, 205, 195)
            },
            {
                "name": "chat_dark",
                "background": (32, 34, 38),
                "panel": (50, 53, 59),
                "text": (238, 238, 238),
                "border": (75, 78, 84)
            }
        ]
    else:
        themes = [
            {
                "name": "document_white",
                "background": (235, 235, 235),
                "panel": (255, 255, 255),
                "text": (25, 25, 25),
                "border": (185, 185, 185)
            },
            {
                "name": "document_gray",
                "background": (224, 227, 230),
                "panel": (248, 248, 246),
                "text": (32, 32, 32),
                "border": (178, 180, 182)
            },
            {
                "name": "document_warm",
                "background": (232, 228, 220),
                "panel": (255, 252, 244),
                "text": (35, 32, 28),
                "border": (192, 184, 170)
            }
        ]

    return rng.choice(themes)


def calculate_layout(
    text: str,
    font: ImageFont.FreeTypeFont,
    content_width: int,
    start_x: int,
    start_y: int,
    line_height: int
) -> tuple[list[dict[str, int] | None], int]:
    temp_image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(temp_image)

    x = start_x
    y = start_y
    max_x = start_x + content_width
    positions = []

    for char in text:
        if char == "\n":
            positions.append(None)
            x = start_x
            y += line_height
            continue

        if char == "\t":
            advance = max(1, int(font.size * 2))
        else:
            advance = max(
                1,
                int(math.ceil(draw.textlength(char, font=font)))
            )

        if x + advance > max_x and x > start_x:
            x = start_x
            y += line_height

        positions.append({
            "x": x,
            "y": y,
            "width": advance,
            "height": line_height
        })

        x += advance

    total_height = y + line_height
    return positions, total_height


def build_char_boxes(
    text: str,
    positions: list[dict[str, int] | None]
) -> list[list[int] | None]:
    boxes = []

    for char, position in zip(text, positions):
        if position is None or char == "\n":
            boxes.append(None)
            continue

        x = position["x"]
        y = position["y"]
        width = position["width"]
        height = position["height"]

        boxes.append([
            x,
            y,
            x + width,
            y + height
        ])

    return boxes


def merge_entity_boxes(
    char_boxes: list[list[int] | None],
    start: int,
    end: int,
    image_width: int,
    image_height: int,
    padding: int = 3
) -> list[list[int]]:
    selected = []

    for index in range(start, end):
        if index >= len(char_boxes):
            break

        box = char_boxes[index]

        if box is not None:
            selected.append(box)

    if not selected:
        return []

    lines: dict[int, list[list[int]]] = {}

    for box in selected:
        y_key = box[1]
        lines.setdefault(y_key, []).append(box)

    merged = []

    for boxes in lines.values():
        x1 = min(box[0] for box in boxes) - padding
        y1 = min(box[1] for box in boxes) - padding
        x2 = max(box[2] for box in boxes) + padding
        y2 = max(box[3] for box in boxes) + padding

        merged.append([
            max(0, x1),
            max(0, y1),
            min(image_width, x2),
            min(image_height, y2)
        ])

    merged.sort(key=lambda item: (item[1], item[0]))
    return merged


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = value.strip("_")

    if not value:
        return "sample"

    return value[:80]


def render_sample(
    sample: dict[str, Any],
    output_path: Path,
    font_path: str | None,
    rng: random.Random
) -> dict[str, Any]:
    width = rng.randint(780, 1080)
    font_size = rng.randint(24, 34)
    line_gap = rng.randint(10, 18)
    line_height = font_size + line_gap

    outer_margin = rng.randint(24, 44)
    content_padding = rng.randint(30, 50)
    header_height = rng.randint(26, 48)

    theme = choose_theme(sample["source"], rng)
    font = load_font(font_path, font_size)

    panel_x1 = outer_margin
    panel_y1 = outer_margin
    panel_x2 = width - outer_margin

    text_x = panel_x1 + content_padding
    text_y = panel_y1 + header_height + content_padding
    content_width = panel_x2 - text_x - content_padding

    positions, text_bottom = calculate_layout(
        sample["text"],
        font,
        content_width,
        text_x,
        text_y,
        line_height
    )

    panel_y2 = text_bottom + content_padding
    image_height = panel_y2 + outer_margin

    image = Image.new(
        "RGB",
        (width, image_height),
        theme["background"]
    )
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        [panel_x1, panel_y1, panel_x2, panel_y2],
        radius=12,
        fill=theme["panel"],
        outline=theme["border"],
        width=2
    )

    draw.rectangle(
        [
            panel_x1,
            panel_y1,
            panel_x2,
            panel_y1 + header_height
        ],
        fill=theme["border"]
    )

    for char, position in zip(sample["text"], positions):
        if position is None or char in ("\n", "\t"):
            continue

        draw.text(
            (position["x"], position["y"]),
            char,
            font=font,
            fill=theme["text"],
            anchor="lt"
        )

    char_boxes = build_char_boxes(sample["text"], positions)

    converted_entities = []

    for entity in sample["entities"]:
        boxes = merge_entity_boxes(
            char_boxes,
            entity["start"],
            entity["end"],
            width,
            image_height
        )

        converted_entities.append({
            "type": entity["type"],
            "text": entity["text"],
            "start": entity["start"],
            "end": entity["end"],
            "boxes": boxes
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    return {
        "id": sample["id"],
        "image": output_path.as_posix(),
        "source": sample["source"],
        "template": theme["name"],
        "width": width,
        "height": image_height,
        "text": sample["text"],
        "entities": converted_entities
    }


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--formal",
        type=Path,
        default=Path("data/raw/formal.jsonl")
    )
    parser.add_argument(
        "--chat",
        type=Path,
        default=Path("data/raw/chat.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated")
    )
    parser.add_argument(
        "--font",
        type=str,
        default=None
    )
    parser.add_argument(
        "--formal-count",
        type=int,
        default=300
    )
    parser.add_argument(
        "--chat-count",
        type=int,
        default=200
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)

    raw_rows = []

    if args.formal.exists():
        formal_rows = load_jsonl(args.formal, "formal")
        rng.shuffle(formal_rows)
        raw_rows.extend(formal_rows[:args.formal_count])
    else:
        print(f"文件不存在：{args.formal}")

    if args.chat.exists():
        chat_rows = load_jsonl(args.chat, "chat")
        rng.shuffle(chat_rows)
        raw_rows.extend(chat_rows[:args.chat_count])
    else:
        print(f"文件不存在：{args.chat}")

    samples = []

    for index, row in enumerate(raw_rows):
        sample = extract_sample(row, index)

        if sample is not None:
            samples.append(sample)

    rng.shuffle(samples)

    train_rows = []
    test_rows = []
    invalid_count = 0

    for index, sample in enumerate(tqdm(samples, desc="生成图片")):
        split = (
            "test"
            if rng.random() < args.test_ratio
            else "train"
        )

        source = sample["source"]
        filename = (
            f"{source}_{index:06d}_"
            f"{safe_filename(sample['id'])}.png"
        )

        image_path = (
            args.output
            / "images"
            / split
            / filename
        )

        try:
            annotation = render_sample(
                sample,
                image_path,
                args.font,
                rng
            )
        except Exception as error:
            invalid_count += 1
            print(f"生成失败：{sample['id']}，原因：{error}")
            continue

        annotation["split"] = split

        if split == "train":
            train_rows.append(annotation)
        else:
            test_rows.append(annotation)

    save_jsonl(
        train_rows,
        args.output / "annotations_train.jsonl"
    )
    save_jsonl(
        test_rows,
        args.output / "annotations_test.jsonl"
    )

    print()
    print(f"训练图片：{len(train_rows)}")
    print(f"测试图片：{len(test_rows)}")
    print(f"失败样本：{invalid_count}")
    print(f"输出目录：{args.output.resolve()}")


if __name__ == "__main__":
    main()