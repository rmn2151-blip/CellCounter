"""Colony / dark-speck detection for petri dish photographs.

The pipeline, in order:

1.  Find the dish and build a circular region of interest just inside the rim.
2.  Estimate the smooth illumination background and subtract it, which removes
    the lighting gradient and leaves a local-contrast image where every dot has
    the same amplitude regardless of where it sits on the plate.
3.  Mask marker handwriting and blown-out specular highlights.
4.  Threshold the contrast image with hysteresis: a blob must peak above
    median + k_seed*sigma to be kept, and is then grown out to a lower level so
    its measured size is right.  Sigma is a robust (MAD) estimate taken from the
    plate itself, so the threshold adapts to each image's noise level.
5.  Measure every surviving blob and gate it on physical size and shape.
6.  Reject air bubbles, which read as a dark centre inside a bright ring.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.measure import label, regionprops

from .params import Params
from .plate import Plate, detect_plate


@dataclass
class Detection:
    x: float
    y: float
    area_px: float
    diameter_px: float
    diameter_mm: float
    contrast: float
    circularity: float
    solidity: float
    halo: float


#: Which overlay colour each rejection reason is drawn in. "noise" and
#: "oversized" are deliberately kept out of the headline total: noise blobs
#: number in the hundreds and oversized ones are glare edges and mottling,
#: so neither is a plausible colony.
CATEGORY_OF_REASON = {
    "bubble": "bubbles",
    "bubble_neighbour": "bubbles",
    "low_circularity": "shape_rejected",
    "low_solidity": "shape_rejected",
    "too_large": "oversized",
    "too_small": "noise",
}

#: Categories summed into the reported total, in overlay-colour order.
TOTAL_CATEGORIES = ("counted", "bubbles", "shape_rejected")


@dataclass
class Rejection:
    x: float
    y: float
    reason: str
    diameter_px: float
    contrast: float
    halo: float


@dataclass
class Result:
    count: int
    detections: list[Detection]
    plate: Plate
    rejected: Counter
    stats: dict
    rejections: list[Rejection] = field(default_factory=list)
    debug: dict = field(default_factory=dict)

    @property
    def categories(self) -> dict[str, int]:
        """Counts per overlay colour, plus the total of the three that could
        plausibly be colonies."""
        out = {"counted": self.count, "bubbles": 0, "shape_rejected": 0,
               "oversized": 0, "noise": 0}
        for reason, n in self.rejected.items():
            out[CATEGORY_OF_REASON.get(reason, "noise")] += n
        out["total"] = sum(out[c] for c in TOTAL_CATEGORIES)
        return out


def _odd(n: int) -> int:
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def _backgrounds(gray: np.ndarray, kernel_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Bright and dark illumination envelopes.

    Morphology with a kernel spanning several colony diameters is what we
    actually want, but a large elliptical structuring element at full
    resolution is slow.  Downsampling first is equivalent here because the
    background is smooth by construction, and the area-averaging step already
    dissolves the dots we are trying to exclude from the estimate.
    """
    h, w = gray.shape
    k_small = 15
    scale = float(np.clip(k_small / max(kernel_px, 1.0), 0.02, 0.5))
    dsize = (max(int(w * scale), 32), max(int(h * scale), 32))

    small = cv2.resize(gray, dsize, interpolation=cv2.INTER_AREA)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small, k_small))
    bright_env = cv2.morphologyEx(small, cv2.MORPH_CLOSE, se)   # dark dots removed
    dark_env = cv2.morphologyEx(small, cv2.MORPH_OPEN, se)      # bright dots removed

    up = lambda m: cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC)
    return up(bright_env), up(dark_env)


def _dilate(mask: np.ndarray, radius_px: float) -> np.ndarray:
    r = max(_odd(radius_px), 3)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))


def _keep_large(mask: np.ndarray, area_limit: float, length_limit: float | None = None) -> np.ndarray:
    """Drop connected regions small enough to plausibly be a colony."""
    out = np.zeros_like(mask)
    if not mask.any():
        return out
    for region in regionprops(label(mask, connectivity=2)):
        if region.area >= area_limit or (
            length_limit is not None and region.axis_major_length >= length_limit
        ):
            out[region.coords[:, 0], region.coords[:, 1]] = 1
    return out


def _ink_mask(gray: np.ndarray, roi: np.ndarray, p: Params, px_per_mm: float,
              max_diameter_px: float, plate_median: float) -> np.ndarray:
    """Marker handwriting and scratches.

    Handwriting is either far bigger than any colony, or long and thin.
    Anything small and compact is left alone: it might be a genuinely dark
    colony rather than ink.
    """
    ink_level = max(plate_median - p.ink_delta, 1.0)
    raw = ((gray < ink_level) & (roi > 0)).astype(np.uint8)
    max_colony_area = np.pi / 4.0 * max_diameter_px ** 2
    ink = _keep_large(raw, p.ink_area_factor * max_colony_area,
                      p.ink_length_factor * max_diameter_px)
    if not ink.any():
        return ink

    ink = _dilate(ink, p.ink_dilate_mm * px_per_mm)
    # If "handwriting" covers most of the dish, the plate is simply dark or was
    # mis-detected. Masking it all would leave nothing to measure, so trust the
    # image over the heuristic and mask nothing.
    if ink.sum() > 0.5 * max(int((roi > 0).sum()), 1):
        return np.zeros_like(ink)
    return ink


def _glare_mask(gray: np.ndarray, bright_resp: np.ndarray, roi: np.ndarray, p: Params,
                px_per_mm: float, max_diameter_px: float) -> np.ndarray:
    """Specular highlights.

    Only broad bright regions are masked; the area test deliberately leaves
    small bright features -- notably bubble rims -- out of the mask, because
    those carry the signal the halo test needs.
    """
    raw = (gray >= p.glare_value) & (roi > 0)
    if p.glare_response is not None:
        raw |= (bright_resp >= p.glare_response) & (roi > 0)
    max_colony_area = np.pi / 4.0 * max_diameter_px ** 2
    glare = _keep_large(raw.astype(np.uint8), p.glare_area_factor * max_colony_area)
    return _dilate(glare, p.glare_dilate_mm * px_per_mm) if glare.any() else glare


def _hysteresis(resp: np.ndarray, valid: np.ndarray, seed: float, grow: float) -> np.ndarray:
    """Keep every region that reaches `seed`, traced out down to `grow`.

    Image noise produces plenty of small regions that creep over a single
    threshold but never form a real peak; requiring a seed removes them, while
    growing down to the lower level keeps the measured size honest.
    """
    grown = ((resp >= grow) & valid).astype(np.uint8)
    if not grown.any():
        return grown

    n, labels = cv2.connectedComponents(grown, connectivity=8)
    if n <= 1:
        return grown

    seeded = np.unique(labels[(resp >= seed) & valid])
    seeded = seeded[seeded > 0]
    if seeded.size == 0:
        return np.zeros_like(grown)

    keep = np.zeros(n, dtype=np.uint8)
    keep[seeded] = 1
    return keep[labels]


def _split_touching(binary: np.ndarray, max_colony_area: float) -> np.ndarray:
    """Watershed-split blobs that look like two or more merged colonies."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    labels = label(binary, connectivity=2)
    out = binary.copy()
    for region in regionprops(labels):
        if region.area < 1.8 * max_colony_area:
            continue
        sub = (labels[region.slice] == region.label)
        dist = ndi.distance_transform_edt(sub)
        peaks = peak_local_max(
            dist, min_distance=max(int(np.sqrt(max_colony_area / np.pi)), 2), labels=sub
        )
        if len(peaks) < 2:
            continue
        markers = np.zeros(sub.shape, dtype=np.int32)
        for i, (py, px_) in enumerate(peaks, start=1):
            markers[py, px_] = i
        ws = watershed(-dist, markers, mask=sub)
        borders = np.zeros(sub.shape, dtype=bool)
        borders[:-1, :] |= (ws[:-1, :] != ws[1:, :]) & (ws[:-1, :] > 0) & (ws[1:, :] > 0)
        borders[:, :-1] |= (ws[:, :-1] != ws[:, 1:]) & (ws[:, :-1] > 0) & (ws[:, 1:] > 0)
        out[region.slice][borders] = 0
    return out


def _halo_score(bright_resp: np.ndarray, blob_mask: np.ndarray, sl, radius_px: float) -> float:
    """Mean bright-side contrast in a ring just outside the blob.

    An air bubble refracts light into a bright rim around its dark centre;
    a colony sits on undisturbed agar and scores near zero.
    """
    inner = 1
    outer = inner + max(int(round(0.5 * radius_px)), 3)

    pad = outer + 2
    y0 = max(sl[0].start - pad, 0)
    y1 = min(sl[0].stop + pad, bright_resp.shape[0])
    x0 = max(sl[1].start - pad, 0)
    x1 = min(sl[1].stop + pad, bright_resp.shape[1])

    local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    local[sl[0].start - y0: sl[0].stop - y0, sl[1].start - x0: sl[1].stop - x0] = blob_mask

    se_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(2 * inner + 1),) * 2)
    se_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(2 * outer + 1),) * 2)
    ring = cv2.dilate(local, se_out).astype(bool) & ~cv2.dilate(local, se_in).astype(bool)
    if not ring.any():
        return 0.0
    return float(bright_resp[y0:y1, x0:x1][ring].mean())


def analyze(bgr: np.ndarray, params: Params | None = None, debug: bool = False) -> Result:
    p = (params or Params()).scaled()

    plate = detect_plate(bgr)
    roi = plate.mask(bgr.shape, p.roi_fraction)

    px_per_mm = plate.px_per_mm(p.plate_diameter_mm)
    min_diameter_px = p.min_diameter_mm * px_per_mm
    max_diameter_px = p.max_diameter_mm * px_per_mm
    max_colony_area = np.pi / 4.0 * max_diameter_px ** 2

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = lab[:, :, 0]

    plate_median = float(np.median(gray[roi > 0])) if roi.any() else 128.0
    ink = _ink_mask(gray, roi, p, px_per_mm, max_diameter_px, plate_median)

    # Flatten the area outside the dish so the dark bench top does not drag the
    # background estimate down near the rim.
    flat = gray.copy()
    flat[roi == 0] = np.uint8(plate_median)

    bright_env, dark_env = _backgrounds(flat, 2.5 * max_diameter_px)
    dark_resp = cv2.subtract(bright_env, flat)   # how much darker than local background
    bright_resp = cv2.subtract(flat, dark_env)   # how much brighter

    glare = _glare_mask(gray, bright_resp, roi, p, px_per_mm, max_diameter_px)

    resp = dark_resp if p.polarity == "dark" else bright_resp
    # Mild matched filtering: suppresses single-pixel sensor noise while
    # leaving real dots intact. Tied to the smallest dot we are willing to
    # count, so that a higher-resolution photo of the same plate is smoothed
    # proportionally rather than under-filtered.
    blur_sigma = float(np.clip(min_diameter_px / 3.0, 0.6, 2.0))
    resp = cv2.GaussianBlur(resp.astype(np.float32), (0, 0), blur_sigma)

    valid = (roi > 0) & (ink == 0) & (glare == 0)
    if not valid.any():
        roi_px = max(int((roi > 0).sum()), 1)
        raise RuntimeError(
            "Nothing left to measure inside the plate: "
            f"{100 * int(glare.sum()) / roi_px:.0f}% masked as glare and "
            f"{100 * int(ink.sum()) / roi_px:.0f}% as marker ink. "
            "The photo is probably overexposed, or the dish was not found "
            "correctly - check the overlay written with --debug."
        )

    vals = resp[valid]
    med = float(np.median(vals))
    sigma = max(1.4826 * float(np.median(np.abs(vals - med))), 0.5)

    seed_level = max(med + p.k_seed * sigma, p.min_contrast)
    # The grow level is held to a fraction of the seed rather than allowed to
    # follow it all the way down. Otherwise raising sensitivity sinks it into
    # the noise, blobs bleed into their neighbours, and the merged result is
    # thrown out as oversized -- which makes the count fall as sensitivity rises.
    grow_level = float(np.clip(max(med + p.k_grow * sigma, 0.5 * p.min_contrast),
                               0.6 * seed_level, seed_level))
    binary = _hysteresis(resp, valid, seed_level, grow_level)
    if p.split_touching and binary.any():
        binary = _split_touching(binary, max_colony_area)

    detections: list[Detection] = []
    rejected: Counter = Counter()
    rejections: list[Rejection] = []

    labels = label(binary, connectivity=2)
    for region in regionprops(labels, intensity_image=resp):
        area = float(region.area)
        equiv_diameter = 2.0 * np.sqrt(area / np.pi)
        cy, cx = region.centroid

        def drop(reason: str, halo: float = 0.0) -> None:
            rejected[reason] += 1
            rejections.append(Rejection(float(cx), float(cy), reason,
                                        float(equiv_diameter),
                                        float(region.intensity_mean), halo))

        if area < p.min_area_px or equiv_diameter < min_diameter_px:
            drop("too_small")
            continue
        if equiv_diameter > max_diameter_px:
            drop("too_large")
            continue

        perimeter = float(region.perimeter) or 1.0
        circularity = float(np.clip(4.0 * np.pi * area / (perimeter ** 2), 0.0, 1.0))
        solidity = float(region.solidity)

        if area >= p.circularity_min_area_px:
            if circularity < p.min_circularity:
                drop("low_circularity")
                continue
            if solidity < p.min_solidity:
                drop("low_solidity")
                continue

        halo = 0.0
        if p.reject_bubbles and p.polarity == "dark":
            blob = (labels[region.slice] == region.label).astype(np.uint8)
            halo = _halo_score(bright_resp, blob, region.slice, equiv_diameter / 2.0)
            if halo > p.bubble_halo:
                drop("bubble", halo)
                continue

        detections.append(
            Detection(
                x=float(cx), y=float(cy), area_px=area,
                diameter_px=float(equiv_diameter),
                diameter_mm=float(equiv_diameter / px_per_mm),
                contrast=float(region.intensity_mean),
                circularity=circularity, solidity=solidity, halo=halo,
            )
        )

    # A bubble's bright rim is broken into several dark arcs; the strongest arc
    # trips the halo test but a weaker one can survive as a "dot". Once any
    # part of a bubble is identified, clear its whole neighbourhood.
    bubbles = [r for r in rejections if r.reason == "bubble"]
    if bubbles and detections:
        kept = []
        for d in detections:
            near = any(
                np.hypot(d.x - b.x, d.y - b.y) <= 1.5 * b.diameter_px + 3.0
                for b in bubbles
            )
            if near:
                rejected["bubble_neighbour"] += 1
                rejections.append(Rejection(d.x, d.y, "bubble_neighbour",
                                            d.diameter_px, d.contrast, d.halo))
            else:
                kept.append(d)
        detections = kept

    roi_area_cm2 = np.pi * (plate.radius * p.roi_fraction / px_per_mm / 10.0) ** 2
    stats = {
        "seed_level": round(seed_level, 2),
        "grow_level": round(grow_level, 2),
        "noise_sigma": round(sigma, 3),
        "plate_median": round(plate_median, 1),
        "px_per_mm": round(px_per_mm, 3),
        "plate_radius_px": round(plate.radius, 1),
        "analysed_area_cm2": round(roi_area_cm2, 2),
        "density_per_cm2": round(len(detections) / roi_area_cm2, 2) if roi_area_cm2 else 0.0,
        "ink_masked_px": int(ink.sum()),
        "glare_masked_px": int(glare.sum()),
    }

    result = Result(len(detections), detections, plate, rejected, stats, rejections)
    if debug:
        result.debug = {
            "gray": gray, "resp": resp, "binary": binary * 255,
            "ink": ink * 255, "glare": glare * 255, "roi": roi,
        }
    return result


def analyze_path(path, params: Params | None = None, debug: bool = False) -> Result:
    from .io_utils import load_image

    return analyze(load_image(path), params, debug)
