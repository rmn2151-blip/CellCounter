"""Image loading helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SUPPORTED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as BGR uint8, honouring EXIF rotation.

    iPhone photos are often HEIC, which OpenCV cannot decode; we fall back to
    pillow-heif when it is installed and otherwise explain how to proceed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img

    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            import pillow_heif  # type: ignore
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                f"{path.name} is a HEIC file. Install support with "
                "`pip install pillow-heif`, or convert it to JPEG first."
            ) from exc
        pillow_heif.register_heif_opener()
        rgb = np.array(Image.open(path).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    raise RuntimeError(f"Could not decode {path}")


def iter_images(paths: list[str]) -> list[Path]:
    """Expand a mix of files and directories into a sorted list of images."""
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in SUPPORTED))
        else:
            out.append(p)
    return out
