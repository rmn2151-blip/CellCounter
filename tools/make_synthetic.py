"""Generate synthetic plate photographs with a known ground-truth dot count.

These reproduce the hazards visible in real phone photos of a dish: a strong
illumination gradient, specular glare, black marker handwriting that reaches
well inside the plate circle, air bubbles that look like dark centres inside
bright rings, agar mottling and sensor noise.

    python tools/make_synthetic.py --out tests/data --n 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _agar_field(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency mottling, as agar is never perfectly uniform."""
    coarse = rng.normal(0.0, 1.0, size=(h // 40 + 2, w // 40 + 2)).astype(np.float32)
    return cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)


def make_plate(seed: int = 0, size: tuple[int, int] = (2000, 1500),
               n_dots: int = 120) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    h, w = size
    cx, cy = w / 2 + rng.uniform(-30, 30), h / 2 + rng.uniform(-40, 40)
    radius = min(h, w) * 0.47

    # Dark bench top with a little texture.
    img = rng.normal(28, 6, size=(h, w)).astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Agar disc with an illumination gradient across the frame.
    gradient = 1.0 - 0.18 * ((yy / h) + 0.5 * (xx / w))
    agar = (205.0 * gradient) + 3.0 * _agar_field(h, w, rng)
    inside = dist <= radius * 0.955
    img[inside] = agar[inside]

    # Bright plastic rim, and the darker meniscus just inside it.
    rim = (dist > radius * 0.955) & (dist <= radius * 1.02)
    img[rim] = 236 + rng.normal(0, 8, size=img.shape)[rim]
    meniscus = (dist > radius * 0.90) & (dist <= radius * 0.955)
    img[meniscus] -= 14.0

    truth = {"plate": [float(cx), float(cy), float(radius)], "dots": [],
             "bubbles": [], "ink": []}

    forbidden = np.zeros((h, w), dtype=np.uint8)

    # --- marker handwriting, angled and well inside the plate circle --------
    label_img = np.zeros((h, w), dtype=np.uint8)
    angle = rng.uniform(0, 2 * np.pi)
    for i, text in enumerate(["SG5/1SG5", "2A  GL-iv/19", "LB"]):
        ox = cx + 0.62 * radius * np.cos(angle) + i * 0.10 * radius * np.cos(angle)
        oy = cy + 0.62 * radius * np.sin(angle) + i * 0.10 * radius * np.sin(angle)
        cv2.putText(label_img, text, (int(ox - 180), int(oy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 6, cv2.LINE_AA)
    ink = label_img > 0
    img[ink] = rng.normal(45, 10, size=img.shape)[ink]
    truth["ink"] = [int(ink.sum())]
    forbidden |= cv2.dilate(ink.astype(np.uint8), np.ones((25, 25), np.uint8))

    # --- specular glare -----------------------------------------------------
    for _ in range(3):
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.55, 0.85) * radius
        gx, gy = int(cx + r * np.cos(a)), int(cy + r * np.sin(a))
        axes = (int(rng.uniform(60, 150)), int(rng.uniform(25, 70)))
        blob = np.zeros((h, w), np.float32)
        cv2.ellipse(blob, (gx, gy), axes, rng.uniform(0, 180), 0, 360, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), 25)
        img += blob * rng.uniform(45, 75)
        forbidden |= (blob > 0.05).astype(np.uint8)

    # --- air bubbles: dark centre, bright refracting rim --------------------
    for _ in range(6):
        a, r = rng.uniform(0, 2 * np.pi), rng.uniform(0.1, 0.85) * radius
        bx, by = int(cx + r * np.cos(a)), int(cy + r * np.sin(a))
        br = int(rng.uniform(5, 11))
        if forbidden[by, bx]:
            continue
        cv2.circle(img, (bx, by), br + 3, 255.0, 3, cv2.LINE_AA)
        cv2.circle(img, (bx, by), br, 165.0, -1, cv2.LINE_AA)
        truth["bubbles"].append([bx, by, br])
        forbidden |= cv2.circle(np.zeros((h, w), np.uint8), (bx, by), br + 14, 1, -1)

    # --- the dots we actually want counted ----------------------------------
    placed = 0
    attempts = 0
    while placed < n_dots and attempts < n_dots * 200:
        attempts += 1
        a = rng.uniform(0, 2 * np.pi)
        r = radius * 0.88 * np.sqrt(rng.uniform(0, 1))
        dx, dy = cx + r * np.cos(a), cy + r * np.sin(a)
        ix, iy = int(dx), int(dy)
        if not (0 <= ix < w and 0 <= iy < h) or forbidden[iy, ix]:
            continue

        # Dot sizes are physical, so a higher-resolution render of the same
        # plate has bigger dots in pixels rather than smaller colonies.
        dot_radius = rng.uniform(1.5, 3.2) * (radius / 705.0)
        amplitude = rng.uniform(18, 45)

        pad = int(dot_radius * 4) + 3
        y0, y1 = max(iy - pad, 0), min(iy + pad + 1, h)
        x0, x1 = max(ix - pad, 0), min(ix + pad + 1, w)
        ly, lx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        profile = np.exp(-((lx - dx) ** 2 + (ly - dy) ** 2) / (2 * (dot_radius * 0.6) ** 2))
        img[y0:y1, x0:x1] -= amplitude * profile

        truth["dots"].append([float(dx), float(dy), float(dot_radius), float(amplitude)])
        forbidden |= cv2.circle(np.zeros((h, w), np.uint8), (ix, iy), int(dot_radius) + 8, 1, -1)
        placed += 1

    # Optics and sensor.
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    img += rng.normal(0, 2.0, size=img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
    bgr[:, :, 0] *= 0.98  # faint colour cast, like the greenish agar in the real photos
    bgr[:, :, 2] *= 0.96
    return np.clip(bgr, 0, 255).astype(np.uint8), truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/data")
    ap.add_argument("--n", type=int, default=3, help="how many plates to generate")
    ap.add_argument("--dots", type=int, default=120)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        img, truth = make_plate(seed=i, n_dots=args.dots)
        cv2.imwrite(str(out / f"synthetic_{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / f"synthetic_{i}.json").write_text(json.dumps(truth))
        print(f"synthetic_{i}.jpg  dots={len(truth['dots'])}  bubbles={len(truth['bubbles'])}")


if __name__ == "__main__":
    main()
