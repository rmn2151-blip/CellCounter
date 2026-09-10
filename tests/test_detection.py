"""Tests against synthetic plates with a known dot count.

The images are generated on the fly so the repository does not have to carry
binary fixtures.  They reproduce the hazards of a real phone photo: an
illumination gradient, specular glare, marker handwriting inside the plate
circle, air bubbles, agar mottling and sensor noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellcounter import Params, analyze  # noqa: E402
from cellcounter.plate import detect_plate  # noqa: E402
from cellcounter.render import annotate, crop_sheet, debug_panel  # noqa: E402
from tools.make_synthetic import make_plate  # noqa: E402


@pytest.fixture(scope="module")
def plates():
    return [make_plate(seed=i, n_dots=120) for i in range(3)]


def match(detections, truth_dots, tolerance=8.0):
    truth = np.array([[d[0], d[1]] for d in truth_dots])
    used = np.zeros(len(truth), bool)
    tp = 0
    for det in detections:
        dist = np.hypot(truth[:, 0] - det.x, truth[:, 1] - det.y)
        dist[used] = np.inf
        j = int(np.argmin(dist))
        if dist[j] <= tolerance:
            used[j] = True
            tp += 1
    return tp, len(detections) - tp, len(truth) - tp


def test_plate_detection_is_accurate(plates):
    for img, truth in plates:
        plate = detect_plate(img)
        cx, cy, radius = truth["plate"]
        assert abs(plate.radius - radius) / radius < 0.05
        assert np.hypot(plate.cx - cx, plate.cy - cy) < 0.05 * radius


def test_counts_are_close_to_truth(plates):
    for img, truth in plates:
        result = analyze(img, Params())
        error = abs(result.count - len(truth["dots"])) / len(truth["dots"])
        assert error < 0.20, f"count {result.count} vs truth {len(truth['dots'])}"


def test_precision_and_recall(plates):
    totals = np.zeros(3, dtype=int)
    for img, truth in plates:
        result = analyze(img, Params())
        totals += np.array(match(result.detections, truth["dots"]))
    tp, fp, fn = totals
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    assert precision > 0.95, f"precision {precision:.3f}"
    assert recall > 0.80, f"recall {recall:.3f}"


def test_marker_writing_is_never_counted(plates):
    """The handwriting is black and sits well inside the plate circle, so it
    is the single most dangerous false positive source."""
    for img, _ in plates:
        result = analyze(img, Params(), debug=True)
        ink = result.debug["ink"] > 0
        on_ink = [d for d in result.detections if ink[int(d.y), int(d.x)]]
        assert not on_ink, f"{len(on_ink)} detections landed on marker ink"


def test_bubbles_are_mostly_rejected(plates):
    for img, truth in plates:
        result = analyze(img, Params())
        bubbles = np.array([[b[0], b[1]] for b in truth["bubbles"]])
        if not len(bubbles):
            continue
        counted = sum(
            1 for d in result.detections
            if np.min(np.hypot(bubbles[:, 0] - d.x, bubbles[:, 1] - d.y)) <= 14
        )
        assert counted <= 1, f"{counted} of {len(bubbles)} bubbles counted as dots"


def test_blank_plate_counts_zero():
    img, _ = make_plate(seed=11, n_dots=0)
    assert analyze(img, Params()).count == 0


def test_sensitivity_increases_count_over_the_useful_range(plates):
    """Below about 1.2 the count rises with sensitivity, as expected.

    It is deliberately not tested above that: once the grow threshold nears the
    noise floor, agar mottling merges into large regions that swallow real dots
    and are then discarded as oversized, so the count can fall again. That is a
    property of the image, not a bug -- see `cellcounter sweep`.
    """
    img, _ = plates[0]
    counts = [analyze(img, Params(sensitivity=s)).count for s in (0.6, 0.8, 1.0)]
    assert counts == sorted(counts), counts


def test_size_filter_respects_physical_units(plates):
    """Sizes are specified in mm, so a stricter window must count fewer dots."""
    img, _ = plates[0]
    wide = analyze(img, Params()).count
    narrow = analyze(img, Params(min_diameter_mm=0.30)).count
    assert narrow < wide


def test_resolution_invariance():
    """The same plate photographed larger should give a similar count."""
    small, truth = make_plate(seed=7, size=(2000, 1500), n_dots=200)
    large, _ = make_plate(seed=7, size=(4032, 3024), n_dots=200)
    a, b = analyze(small, Params()).count, analyze(large, Params()).count
    assert abs(a - b) / max(a, b) < 0.15, f"{a} vs {b}"


@pytest.mark.parametrize("image", [
    np.zeros((500, 500, 3), np.uint8),
    np.random.default_rng(0).integers(0, 255, (600, 800, 3), dtype=np.uint8).astype(np.uint8),
])
def test_degenerate_images_do_not_crash(image):
    try:
        assert analyze(image, Params()).count >= 0
    except RuntimeError as exc:
        assert "plate" in str(exc).lower() or "measure" in str(exc).lower()


def test_overexposed_image_explains_itself():
    with pytest.raises(RuntimeError, match="glare"):
        analyze(np.full((500, 500, 3), 255, np.uint8), Params())


def test_categories_are_consistent(plates):
    """The reported total must equal the three counted colours, and every
    rejection must land in exactly one category."""
    for img, _ in plates:
        result = analyze(img, Params())
        c = result.categories
        assert c["counted"] == result.count == len(result.detections)
        assert c["total"] == c["counted"] + c["bubbles"] + c["shape_rejected"]

        categorised = c["bubbles"] + c["shape_rejected"] + c["oversized"] + c["noise"]
        assert categorised == sum(result.rejected.values()) == len(result.rejections)


def test_bubbles_category_matches_rejection_reasons(plates):
    img, _ = plates[0]
    result = analyze(img, Params())
    expected = result.rejected.get("bubble", 0) + result.rejected.get("bubble_neighbour", 0)
    assert result.categories["bubbles"] == expected
    assert result.categories["bubbles"] > 0, "synthetic plates contain bubbles"


def test_rendering_runs(plates):
    img, _ = plates[0]
    result = analyze(img, Params(), debug=True)
    assert annotate(img, result, Params(), show_rejects=True).shape == img.shape
    assert debug_panel(result).ndim == 3
    assert crop_sheet(img, result) is not None
