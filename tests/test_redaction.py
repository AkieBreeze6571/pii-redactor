from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from services.redaction_service import RedactionService


def entity(boxes: list[list[list[float]]]) -> dict:
    return {"type": "phone", "enabled": True, "boxes": boxes}


@pytest.mark.parametrize("mode", ["black", "mosaic", "blur"])
def test_redaction_modes(tmp_path: Path, mode: str) -> None:
    array = np.tile(np.arange(120, dtype=np.uint8), (60, 1))
    image = Image.fromarray(np.dstack([array, np.flip(array, axis=1), array]))
    service = RedactionService(tmp_path)
    paths = service.redact(image, [entity([[[20, 10], [100, 10], [100, 50], [20, 50]]])], mode, "sample.png")
    result = Image.open(paths["result_path"])
    assert result.tobytes() != image.convert("RGB").tobytes()
    assert Path(paths["preview_path"]).exists()
    assert Path(paths["mask_path"]).exists()
    if mode == "black": assert result.getpixel((50, 30)) == (0, 0, 0)


def test_small_and_out_of_bounds_regions(tmp_path: Path) -> None:
    image = Image.new("RGB", (20, 20), "white")
    boxes = [[[-10, -10], [2, -10], [2, 2], [-10, 2]], [[19, 19], [30, 19], [30, 30], [19, 30]]]
    paths = RedactionService(tmp_path).redact(image, [entity(boxes)], "mosaic")
    assert Image.open(paths["result_path"]).size == image.size


def test_multiple_boxes_and_mask(tmp_path: Path) -> None:
    image = Image.new("RGB", (100, 50), "white")
    boxes = [[[5, 5], [20, 5], [20, 20], [5, 20]], [[60, 20], [90, 20], [90, 40], [60, 40]]]
    paths = RedactionService(tmp_path).redact(image, [entity(boxes)], "black")
    mask = Image.open(paths["mask_path"])
    assert mask.getpixel((10, 10)) == 255
    assert mask.getpixel((70, 30)) == 255
    assert mask.getpixel((40, 30)) == 0


def test_empty_entities_preserves_pixels_and_unique_names(tmp_path: Path) -> None:
    image = Image.new("RGB", (30, 30), "blue")
    service = RedactionService(tmp_path)
    first = service.redact(image, [], filename="bad:name.png")
    second = service.redact(image, [], filename="bad:name.png")
    assert first["result_path"] != second["result_path"]
    assert Image.open(first["result_path"]).tobytes() == image.tobytes()
