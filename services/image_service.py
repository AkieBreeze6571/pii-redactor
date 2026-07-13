"""Conservative image loading and preprocessing with coordinate scaling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError


@dataclass
class PreprocessedImage:
    image: Image.Image
    original_size: tuple[int, int]
    original_image: Image.Image | None = None
    scale_x: float = 1.0
    scale_y: float = 1.0
    warnings: list[str] = field(default_factory=list)


def load_image(value: str | Path | Image.Image | np.ndarray, array_color: str = "bgr") -> Image.Image:
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.copy()
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim == 2:
            return Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB")
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("不支持的 numpy 图片形状")
        array = array.astype(np.uint8)
        if array.shape[2] == 4:
            code = cv2.COLOR_BGRA2RGBA if array_color.lower() == "bgr" else None
            array = cv2.cvtColor(array, code) if code is not None else array
            return Image.fromarray(array, mode="RGBA")
        if array_color.lower() == "bgr":
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(array, mode="RGB")
    raise TypeError(f"不支持的图片输入类型：{type(value).__name__}")


def preprocess_image(
    value: str | Path | Image.Image | np.ndarray,
    max_side: int = 2400,
    grayscale: bool = False,
    sharpen: bool = False,
    binarize: bool = False,
    array_color: str = "bgr",
) -> PreprocessedImage:
    warnings: list[str] = []
    try:
        image = ImageOps.exif_transpose(load_image(value, array_color))
    except (OSError, ValueError, TypeError, UnidentifiedImageError) as exc:
        raise ValueError(f"图片无法读取：{exc}") from exc
    original_size = image.size
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image).convert("RGB")
    else:
        image = image.convert("RGB")
    original_image = image.copy()
    try:
        luminance = float(np.asarray(image.convert("L"), dtype=np.float32).mean())
        if luminance < 70:
            image = ImageOps.autocontrast(image, cutoff=1)
            image = ImageEnhance.Contrast(image).enhance(1.15)
        if grayscale:
            image = image.convert("L").convert("RGB")
        if sharpen:
            image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))
        if binarize:
            gray = np.asarray(image.convert("L"))
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image = Image.fromarray(binary).convert("RGB")
    except Exception:
        warnings.append("图片增强失败，已继续使用基础转换结果。")
    width, height = image.size
    ratio = min(1.0, max_side / max(width, height)) if max_side > 0 else 1.0
    if ratio < 1:
        image = image.resize((max(1, round(width * ratio)), max(1, round(height * ratio))), Image.Resampling.LANCZOS)
    return PreprocessedImage(
        image=image,
        original_size=original_size,
        original_image=original_image,
        scale_x=original_size[0] / image.width,
        scale_y=original_size[1] / image.height,
        warnings=warnings,
    )
