"""Locate the petri dish in a photograph.

The dish is by far the largest bright object in these images, so an Otsu split
against the dark bench top plus a largest-component search is more reliable
than a Hough circle transform (which tends to latch onto the rim highlights).
Hough is kept as a fallback for images where the background is not dark.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Plate:
    cx: float
    cy: float
    radius: float

    def px_per_mm(self, plate_diameter_mm: float) -> float:
        return (2.0 * self.radius) / plate_diameter_mm

    def mask(self, shape: tuple[int, int], fraction: float = 1.0) -> np.ndarray:
        m = np.zeros(shape[:2], dtype=np.uint8)
        cv2.circle(m, (int(round(self.cx)), int(round(self.cy))),
                   int(round(self.radius * fraction)), 255, -1)
        return m


def _largest_component(binary: np.ndarray) -> np.ndarray | None:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return (labels == best).astype(np.uint8) * 255


def detect_plate(bgr: np.ndarray, work_width: int = 900) -> Plate:
    """Return the dish circle in full-resolution pixel coordinates."""
    h, w = bgr.shape[:2]
    scale = min(1.0, work_width / float(w))
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 2.0)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close gaps from the dark marker writing, then fill the interior so the
    # component is a solid disc rather than a ring of rim highlights.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    comp = _largest_component(binary)
    if comp is not None:
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        filled = np.zeros_like(comp)
        cv2.drawContours(filled, [contour], -1, 255, -1)

        area = float(filled.sum()) / 255.0
        (cx, cy), r_enclosing = cv2.minEnclosingCircle(contour)
        r_equivalent = float(np.sqrt(area / np.pi))

        # A spur or a stray reflection joined to the dish would inflate the
        # enclosing circle; trust the area-equivalent radius when they disagree.
        radius = r_enclosing if r_enclosing <= 1.25 * r_equivalent else r_equivalent
        if radius > 0.20 * gray.shape[1]:
            m = cv2.moments(filled, binaryImage=True)
            if r_enclosing > 1.25 * r_equivalent and m["m00"] > 0:
                cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            return Plate(cx / scale, cy / scale, radius / scale)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=gray.shape[0],
        param1=120, param2=60,
        minRadius=int(0.25 * gray.shape[1]), maxRadius=int(0.60 * gray.shape[1]),
    )
    if circles is not None:
        cx, cy, r = circles[0][0]
        return Plate(float(cx) / scale, float(cy) / scale, float(r) / scale)

    # Last resort: assume the dish fills the frame.
    return Plate(w / 2.0, h / 2.0, 0.48 * min(h, w))
