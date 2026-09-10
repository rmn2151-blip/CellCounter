"""Score the detector against synthetic ground truth.

    python tools/evaluate.py tests/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cellcounter import Params, analyze
from cellcounter.io_utils import load_image


def match(detections, truth_dots, tolerance: float) -> tuple[int, int, int]:
    """Greedy nearest-neighbour matching. Returns (tp, fp, fn)."""
    truth = np.array([[d[0], d[1]] for d in truth_dots], dtype=np.float64)
    used = np.zeros(len(truth), dtype=bool)
    tp = 0
    for det in detections:
        if len(truth) == 0:
            break
        d = np.hypot(truth[:, 0] - det.x, truth[:, 1] - det.y)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tolerance:
            used[j] = True
            tp += 1
    return tp, len(detections) - tp, len(truth) - tp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", default="tests/data", nargs="?")
    ap.add_argument("--sensitivity", type=float, default=1.0)
    ap.add_argument("--tolerance", type=float, default=8.0, help="match radius in px")
    args = ap.parse_args()

    params = Params(sensitivity=args.sensitivity)
    totals = np.zeros(3, dtype=int)

    for img_path in sorted(Path(args.data_dir).glob("synthetic_*.jpg")):
        truth = json.loads(img_path.with_suffix(".json").read_text())
        result = analyze(load_image(img_path), params)
        tp, fp, fn = match(result.detections, truth["dots"], args.tolerance)
        totals += np.array([tp, fp, fn])

        plate_err = abs(result.plate.radius - truth["plate"][2]) / truth["plate"][2]
        print(f"{img_path.name}: count={result.count:>4} truth={len(truth['dots']):>4}  "
              f"tp={tp} fp={fp} fn={fn}  plate_radius_err={plate_err:.1%}  "
              f"seed={result.stats['seed_level']}  rejected={dict(result.rejected)}")

    tp, fp, fn = totals
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"\nTOTAL  precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")


if __name__ == "__main__":
    main()
