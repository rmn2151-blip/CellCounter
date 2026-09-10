# CellCounter

Counts small dark colonies ("black dots") on photographs of petri dishes, and
reports enough about what it rejected that you can check its work.

Built for phone photos taken on a bench rather than scanner images, so it
expects an uneven illumination gradient, specular glare, marker handwriting on
the dish, and air bubbles in the agar.

## Install

Requires Python 3.9+ and four packages:

```bash
pip install -r requirements.txt
```

If you use Anaconda, you most likely already have `numpy`, `scipy` and
`scikit-image`, and only need OpenCV:

```bash
pip install opencv-python-headless
# or: conda install -c conda-forge opencv
```

Check everything imports before going further:

```bash
python -c "import numpy, scipy, skimage, cv2; print('READY')"
```

iPhone `.HEIC` photos need `pip install pillow-heif`, or export them as JPEG
first (Preview → File → Export → JPEG).

## Use

```bash
# count every image in a folder, writing overlays and CSVs to results/
python -m cellcounter count images/ --out-dir results

# see how the count responds to the detection threshold
python -m cellcounter sweep images/plate1.jpg

# add the intermediate images and the zoomed crop sheet
python -m cellcounter count images/ --debug
```

Try it without any photos of your own — this generates a plate with a known
number of dots and counts it:

```bash
python tools/make_synthetic.py --out demo --n 1
python -m cellcounter count demo/ --out-dir demo_results
```

It plants 120 dots and should report roughly `107 counted | 11 bubbles |
17 odd shape | total 135`.

## What the colours mean

Every candidate is drawn in one colour, and the overlay carries a legend with
live counts:

| colour | category | in the total |
| --- | --- | --- |
| **green** | counted as a colony | yes |
| **orange** | air bubble — dark centre inside a bright refracting rim | yes |
| **magenta** | rejected on shape — too irregular for a colony | yes |
| **red** | oversized — glare edges and merged mottling, never a colony | no |
| blue tint | masked marker handwriting (a region, not a dot) | no |
| cyan tint | masked specular highlight | no |

The console prints the same breakdown:

```
plate1.jpg: 107 counted (green) | 11 bubbles (orange) | 17 odd shape (magenta) | total 135
            also 14 oversized (red), 0 sub-threshold specks, not in the total.
```

**Only the green number is a colony count.** The total tallies everything
marked and is a diagnostic rather than a result: a large orange count means a
bubbly plate, and a large magenta count usually means the sensitivity is too
high. Sub-threshold noise specks are excluded throughout — there are typically
hundreds and they are not candidates in any meaningful sense.

Every marked candidate appears in the CSV with its `category`,
`overlay_colour` and `reason`, so you can filter or re-tally in a spreadsheet.
`--hide-rejects` draws only the green circles.

## Outputs

| file | contents |
| --- | --- |
| `*_annotated.jpg` | the photo with every candidate circled, plus the legend |
| `*_detections.csv` | one row per marked candidate: category, colour, reason, position, area, diameter in px and mm, contrast, shape |
| `*_crops.jpg` | a grid of zoomed crops, for checking detections at a glance |
| `*_debug.jpg` | the intermediate images, with `--debug` |
| `summary.json` | counts, densities and rejection tallies for the batch |

From Python:

```python
from cellcounter import analyze_path, Params

result = analyze_path("images/plate1.jpg", Params(sensitivity=1.2))
print(result.count, result.categories, result.stats["density_per_cm2"])
```

## How it works

1. **Find the dish.** It is the largest bright object against the dark bench,
   so an Otsu split plus a largest-component search is steadier than a Hough
   circle transform, which tends to lock onto rim highlights. Analysis is
   restricted to the inner 90% of the radius, dropping the rim, the meniscus
   and most edge glare.
2. **Flatten the lighting.** A smooth background is estimated with morphology
   on a downsampled copy and subtracted, leaving a local-contrast image in
   which a dot has the same amplitude wherever it sits on the plate. Without
   this, any single global threshold either misses dots on the dim side of the
   plate or floods the bright side.
3. **Mask the artefacts.** Marker handwriting reaches well inside the plate
   circle and is black, which makes it the most dangerous false-positive
   source. Ink-dark regions are masked only when far larger than any colony,
   or long and thin — so a genuinely dark colony survives. Blown-out
   highlights are masked the same way.
4. **Threshold with hysteresis.** A blob must *peak* above `median + 5σ` to be
   kept, then is grown down to a lower level so its measured size is honest.
   σ is a robust (MAD) estimate taken from the plate itself, so the threshold
   adapts to each photo's noise.
5. **Gate on physical size and shape.** Limits are given in millimetres and
   converted using the measured plate diameter, so the same settings work at
   any camera distance or resolution.
6. **Reject bubbles.** An air bubble refracts light into a bright rim around a
   dark centre. Measuring the brightness of a ring just outside each candidate
   separates them cleanly.

Two design decisions came from measurement rather than first principles:

- **Hysteresis instead of a single threshold.** Noise blobs drift over a low
  threshold but never form a real peak. Requiring a seed peak took precision
  from 0.51 to 0.97 at essentially unchanged recall.
- **Extra masking of non-clipping glare is off by default.** It consistently
  cost precision, because widening the mask shrinks the sample the noise
  estimate is drawn from, while the size filter already discards glare-edge
  blobs.

## Tuning

Start with `sweep`, which prints the count across a range of thresholds:

```
 sensitivity   count  seed level
        0.70      94       12.41
        0.85     105       10.53
        1.00     107        9.21
        1.20     109        7.96
        1.50     111        6.72
        2.00     164        5.47
```

A count you can trust sits in the middle of a **flat stretch** — here 105–111
across a wide range. The jump to 164 is the threshold entering the noise floor.
Pick a sensitivity in the plateau and use it for the whole batch so plates stay
comparable.

Counts can also *fall* as sensitivity rises past about 1.5. That is not a bug:
once the lower threshold nears the noise floor, agar mottling merges into large
regions that swallow real dots, and the merged blob is discarded as oversized.
A falling count means you have gone too far.

Other options:

- `--plate-diameter-mm` (default 90) — set this to your actual dish, since
  every size limit derives from it.
- `--min-diameter-mm` / `--max-diameter-mm` — the size window for a colony.
- `--sensitivity` — the master threshold knob.
- `--keep-bubbles` — if your colonies genuinely have bright halos.
- `--polarity bright` — for pale colonies on dark medium.
- `--split-touching` — watershed-splits merged colonies. Off by default: on the
  test plates it cost precision without recovering any dots, but it is the
  right switch for a crowded plate.

## Accuracy, and what has been tested

On synthetic plates reproducing the hazards above — 360 dots across three
images — the defaults give **precision 1.000, recall 0.867**: no false
positives, and about 13% of dots missed, almost all the faintest and smallest.
Marker writing is never counted. Reproduce with:

```bash
python tools/make_synthetic.py --out tests/data --n 3
python tools/evaluate.py tests/data
pytest tests/
```

**Not yet validated against real photographs.** The parameters were chosen on
simulated images, and simulated noise is better behaved than a real camera
sensor. Before trusting a number from a real plate:

1. Run `count --debug` and look at the overlay legend.
2. Open `*_crops.jpg` — a wrong detection is obvious in seconds.
3. Hand-count one plate and compare. If the count is consistently low, raise
   `--sensitivity`; if it finds things that are not there, lower it.

The count is a measurement, not a fact. Report the settings alongside it, and
keep them fixed across plates you intend to compare.

## Getting better photos

The cheapest accuracy improvements are at the camera:

- **Diffuse the light.** Glare is the one artefact that destroys information
  rather than merely obscuring it — no pixel value survives under a blown-out
  highlight. A sheet of paper between lamp and dish helps enormously.
- **Shoot against a dark, matte background**, which is what makes dish
  detection reliable.
- **Photograph straight down**, centred. A tilted plate becomes an ellipse and
  the mm calibration is then wrong across the frame.
- **Write on the rim, not across the base**, so handwriting never competes
  with the colonies.
- **Fill the frame with the dish** at full camera resolution. Recall falls off
  sharply once dots shrink below about three pixels.
