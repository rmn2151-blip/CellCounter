"""Command line interface.

    python -m cellcounter count images/ --out-dir results --debug
    python -m cellcounter sweep images/plate1.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

from .detect import CATEGORY_OF_REASON, analyze
from .io_utils import iter_images, load_image
from .params import Params
from .render import annotate, crop_sheet, debug_panel

CSV_COLOUR = {"counted": "green", "bubbles": "orange",
              "shape_rejected": "magenta", "oversized": "red"}


def _add_param_args(ap: argparse.ArgumentParser) -> None:
    d = Params()
    ap.add_argument("--plate-diameter-mm", type=float, default=d.plate_diameter_mm,
                    help="physical dish diameter, used to convert sizes (default: %(default)s)")
    ap.add_argument("--roi-fraction", type=float, default=d.roi_fraction,
                    help="fraction of the plate radius analysed (default: %(default)s)")
    ap.add_argument("--sensitivity", type=float, default=d.sensitivity,
                    help=">1 finds fainter dots, <1 is stricter (default: %(default)s)")
    ap.add_argument("--min-diameter-mm", type=float, default=d.min_diameter_mm)
    ap.add_argument("--max-diameter-mm", type=float, default=d.max_diameter_mm)
    ap.add_argument("--min-area-px", type=int, default=d.min_area_px)
    ap.add_argument("--min-circularity", type=float, default=d.min_circularity)
    ap.add_argument("--polarity", choices=["dark", "bright"], default=d.polarity,
                    help="count dark dots on light agar, or the reverse")
    ap.add_argument("--keep-bubbles", action="store_true",
                    help="do not reject dark spots ringed by a bright halo")
    ap.add_argument("--split-touching", action="store_true",
                    help="watershed-split merged colonies")


def _params_from_args(a: argparse.Namespace) -> Params:
    return Params(
        plate_diameter_mm=a.plate_diameter_mm,
        roi_fraction=a.roi_fraction,
        sensitivity=a.sensitivity,
        min_diameter_mm=a.min_diameter_mm,
        max_diameter_mm=a.max_diameter_mm,
        min_area_px=a.min_area_px,
        min_circularity=a.min_circularity,
        polarity=a.polarity,
        reject_bubbles=not a.keep_bubbles,
        split_touching=a.split_touching,
    )


def cmd_count(a: argparse.Namespace) -> int:
    params = _params_from_args(a)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = iter_images(a.images)
    if not paths:
        print("No images found.", file=sys.stderr)
        return 1

    summary = []
    for path in paths:
        try:
            bgr = load_image(path)
        except Exception as exc:
            print(f"{path.name}: SKIPPED ({exc})", file=sys.stderr)
            continue

        result = analyze(bgr, params, debug=True)
        stem = path.stem

        cv2.imwrite(str(out_dir / f"{stem}_annotated.jpg"),
                    annotate(bgr, result, params, show_rejects=not a.hide_rejects),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        px_per_mm = result.stats["px_per_mm"]
        with open(out_dir / f"{stem}_detections.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["category", "overlay_colour", "reason", "x_px", "y_px",
                             "area_px", "diameter_px", "diameter_mm", "contrast",
                             "circularity", "solidity", "halo"])
            for d in result.detections:
                writer.writerow(["counted", "green", "", f"{d.x:.1f}", f"{d.y:.1f}",
                                 f"{d.area_px:.0f}", f"{d.diameter_px:.2f}",
                                 f"{d.diameter_mm:.3f}", f"{d.contrast:.2f}",
                                 f"{d.circularity:.3f}", f"{d.solidity:.3f}",
                                 f"{d.halo:.2f}"])
            for r in result.rejections:
                category = CATEGORY_OF_REASON.get(r.reason, "noise")
                if category == "noise":
                    continue  # hundreds of sub-threshold specks, not useful here
                writer.writerow([category, CSV_COLOUR[category], r.reason,
                                 f"{r.x:.1f}", f"{r.y:.1f}", "",
                                 f"{r.diameter_px:.2f}",
                                 f"{r.diameter_px / px_per_mm:.3f}",
                                 f"{r.contrast:.2f}", "", "", f"{r.halo:.2f}"])

        if a.debug:
            cv2.imwrite(str(out_dir / f"{stem}_debug.jpg"), debug_panel(result))
            sheet = crop_sheet(bgr, result)
            if sheet is not None:
                cv2.imwrite(str(out_dir / f"{stem}_crops.jpg"), sheet)

        rejected = dict(result.rejected)
        c = result.categories
        print(f"{path.name}: {c['counted']} counted (green) | "
              f"{c['bubbles']} bubbles (orange) | "
              f"{c['shape_rejected']} odd shape (magenta) | "
              f"total {c['total']}")
        print(f"{'':>{len(path.name)}}  also {c['oversized']} oversized (red), "
              f"{c['noise']} sub-threshold specks, not in the total.  "
              f"{result.stats['density_per_cm2']}/cm^2, seed={result.stats['seed_level']}")
        summary.append({"image": path.name, "count": result.count,
                        "categories": c, "stats": result.stats,
                        "rejected": rejected})

    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote overlays and CSVs to {out_dir}/")
    return 0


def cmd_sweep(a: argparse.Namespace) -> int:
    """Show how the count responds to sensitivity.

    A trustworthy setting sits on a plateau: if the count changes sharply with
    a small nudge, the threshold is sitting in the noise and the number should
    not be believed.
    """
    params = _params_from_args(a)
    bgr = load_image(a.image)
    print(f"{'sensitivity':>12}  {'count':>6}  {'seed level':>10}")
    for s in [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0]:
        params.sensitivity = s
        result = analyze(bgr, params)
        note = "  <- noise-dominated, do not trust" if s >= 1.5 else ""
        print(f"{s:>12.2f}  {result.count:>6}  {result.stats['seed_level']:>10.2f}{note}")
    print("\nPick a value in the middle of a flat stretch. High sensitivities can\n"
          "report a *lower* count, because merged noise regions get discarded as\n"
          "oversized -- a sign the threshold has gone past anything meaningful.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cellcounter",
                                 description="Count dark colonies on petri dish photos")
    sub = ap.add_subparsers(dest="command", required=True)

    c = sub.add_parser("count", help="count dots in one or more images")
    c.add_argument("images", nargs="+", help="image files or directories")
    c.add_argument("--out-dir", default="results")
    c.add_argument("--debug", action="store_true", help="also write diagnostic panels")
    c.add_argument("--hide-rejects", action="store_true",
                   help="draw only the counted dots, omitting the colour-coded rejects")
    _add_param_args(c)
    c.set_defaults(func=cmd_count)

    s = sub.add_parser("sweep", help="report counts across a range of sensitivities")
    s.add_argument("image")
    _add_param_args(s)
    s.set_defaults(func=cmd_sweep)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
