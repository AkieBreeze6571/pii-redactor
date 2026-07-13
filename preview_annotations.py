import json
from pathlib import Path

from PIL import Image, ImageDraw


ANNOTATION_PATH = Path(
    "data/generated/annotations_test.jsonl"
)
OUTPUT_DIR = Path("data/generated/previews")
MAX_COUNT = 30


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with ANNOTATION_PATH.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index >= MAX_COUNT:
                break

            row = json.loads(line)
            image = Image.open(row["image"]).convert("RGB")
            draw = ImageDraw.Draw(image)

            for entity in row["entities"]:
                for box in entity["boxes"]:
                    draw.rectangle(
                        box,
                        outline=(255, 0, 0),
                        width=3
                    )

            output_path = OUTPUT_DIR / f"preview_{index:03d}.png"
            image.save(output_path)

    print(f"预览图已保存到：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()