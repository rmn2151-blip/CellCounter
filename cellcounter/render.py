"""Overlay rendering, so counts can be checked by eye rather than trusted."""

from __future__ import annotations

import cv2
import numpy as np

from .detect import Result
from .params import Params


# One colour per category, and each colour means exactly one thing. BGR.
GREEN = (0, 255, 0)          # counted as a colony
ORANGE = (0, 140, 255)       # air bubble
MAGENTA = (255, 0, 255)      # rejected on shape
RED = (0, 0, 255)            # too large to be a colony
INK_TINT = (200, 120, 0)     # masked handwriting -- a region, not a dot
GLARE_TINT = (200, 200, 0)   # masked specular highlight

CATEGORY_COLORS = {
    "counted": GREEN,
    "bubbles": ORANGE,
    "shape_rejected": MAGENTA,
    "oversized": RED,
}

REJECT_COLORS = {
    "bubble": ORANGE,
    "bubble_neighbour": ORANGE,
    "too_large": RED,
    "too_small": (120, 120, 120),
    "low_circularity": MAGENTA,
    "low_solidity": MAGENTA,
}

LEGEND_ROWS = [
    ("counted", "counted"),
    ("bubbles", "bubbles"),
    ("shape_rejected", "odd shape"),
    ("oversized", "oversized"),
]


def _draw_legend(out: np.ndarray, result: Result, scale: float) -> None:
    """Colour key with live counts, so the overlay explains itself."""
    counts = result.categories
    pad = int(14 * scale)
    row_h = int(28 * scale)
    box_w = int(240 * scale)
    box_h = row_h * (len(LEGEND_ROWS) + 2) + pad

    x0, y0 = pad, int(80 * scale)
    panel = out[y0:y0 + box_h, x0:x0 + box_w]
    if panel.size:
        out[y0:y0 + box_h, x0:x0 + box_w] = cv2.addWeighted(
            panel, 0.35, np.zeros_like(panel), 0.65, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = y0 + row_h
    for key, label in LEGEND_ROWS:
        swatch = CATEGORY_COLORS[key]
        cv2.circle(out, (x0 + pad + int(6 * scale), y - int(5 * scale)),
                   int(7 * scale), swatch, max(int(2 * scale), 1))
        cv2.putText(out, f"{label}: {counts[key]}",
                    (x0 + pad + int(24 * scale), y), font, 0.55 * scale,
                    swatch, max(int(2 * scale), 1), cv2.LINE_AA)
        y += row_h

    cv2.line(out, (x0 + pad, y - int(17 * scale)), (x0 + box_w - pad, y - int(17 * scale)),
             (255, 255, 255), max(int(1 * scale), 1))
    cv2.putText(out, f"total: {counts['total']}", (x0 + pad, y + int(4 * scale)),
                font, 0.65 * scale, (255, 255, 255), max(int(2 * scale), 1), cv2.LINE_AA)


def annotate(bgr: np.ndarray, result: Result, params: Params, show_masks: bool = True,
             show_rejects: bool = True, show_legend: bool = True) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    unit = max(1, int(round(min(h, w) / 900.0)))

    if show_masks and result.debug:
        tint = out.copy()
        ink = result.debug.get("ink")
        glare = result.debug.get("glare")
        if ink is not None:
            tint[ink > 0] = INK_TINT
        if glare is not None:
            tint[glare > 0] = GLARE_TINT
        out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)

    plate = result.plate
    cv2.circle(out, (int(plate.cx), int(plate.cy)), int(plate.radius), (120, 120, 120), unit)
    cv2.circle(out, (int(plate.cx), int(plate.cy)),
               int(plate.radius * params.roi_fraction), (0, 200, 255), unit)

    if show_rejects:
        for r in result.rejections:
            if r.reason == "too_small":
                continue  # noise blobs, far too numerous to be informative
            colour = REJECT_COLORS.get(r.reason, (0, 0, 255))
            radius = max(int(round(r.diameter_px)), 3 * unit)
            cv2.circle(out, (int(round(r.x)), int(round(r.y))), radius + 2 * unit,
                       colour, unit)

    for d in result.detections:
        radius = max(int(round(d.diameter_px)), 3 * unit)
        cv2.circle(out, (int(round(d.x)), int(round(d.y))), radius + 2 * unit,
                   GREEN, unit)

    scale = min(h, w) / 900.0
    text = f"count: {result.count}"
    cv2.putText(out, text, (int(20 * scale), int(60 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6 * scale, (0, 0, 0), int(6 * scale))
    cv2.putText(out, text, (int(20 * scale), int(60 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6 * scale, GREEN, int(2 * scale))

    if show_legend:
        _draw_legend(out, result, scale)
    return out


def debug_panel(result: Result) -> np.ndarray:
    """2x2 montage of the intermediate images, for diagnosing a bad count."""
    if not result.debug:
        raise ValueError("Run analyze(..., debug=True) first")

    def prep(img, title):
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1])))
        cv2.putText(img, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(img, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        return img

    d = result.debug
    tiles = [
        prep(d["gray"], "L channel"),
        prep(d["resp"], "local contrast"),
        prep(d["binary"], "thresholded"),
        prep(np.maximum(d["ink"], d["glare"]), "masked artefacts"),
    ]
    height = min(t.shape[0] for t in tiles)
    tiles = [t[:height] for t in tiles]
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])


def crop_sheet(bgr: np.ndarray, result: Result, size: int = 64, cols: int = 12,
               limit: int = 144) -> np.ndarray | None:
    """Grid of zoomed crops around each detection for quick visual QC."""
    if not result.detections:
        return None

    half = size // 2
    padded = cv2.copyMakeBorder(bgr, half, half, half, half, cv2.BORDER_REPLICATE)
    crops = []
    for d in result.detections[:limit]:
        cx, cy = int(round(d.x)) + half, int(round(d.y)) + half
        crop = padded[cy - half:cy + half, cx - half:cx + half].copy()
        if crop.shape[:2] != (size, size):
            continue
        crop = cv2.resize(crop, (size * 2, size * 2), interpolation=cv2.INTER_NEAREST)
        cv2.circle(crop, (size, size), max(int(d.diameter_px), 3) + 4, (0, 255, 0), 1)
        crops.append(cv2.copyMakeBorder(crop, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40)))

    if not crops:
        return None
    rows = []
    for i in range(0, len(crops), cols):
        row = crops[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(crops[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)
