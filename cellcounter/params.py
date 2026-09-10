"""Tunable parameters for the colony counting pipeline.

Sizes are given in millimetres wherever possible.  They are converted to pixels
at run time using the detected plate diameter, so the same settings work for
photos taken at different distances or with different cameras.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class Params:
    # --- geometry -----------------------------------------------------------
    plate_diameter_mm: float = 90.0
    """Physical inner diameter of the dish. Standard plates are 90 or 100 mm."""

    roi_fraction: float = 0.90
    """Fraction of the detected plate radius that is analysed. Trims the rim,
    the meniscus ring and most of the edge glare."""

    # --- what counts as a dot ----------------------------------------------
    polarity: str = "dark"
    """'dark' counts dark features on bright agar, 'bright' does the opposite."""

    min_diameter_mm: float = 0.15
    max_diameter_mm: float = 2.00
    min_area_px: int = 4
    """Absolute floor in pixels; kills single-pixel sensor noise regardless of
    how the mm limits work out."""

    # --- detection threshold ------------------------------------------------
    sensitivity: float = 1.0
    """Master knob. >1 detects fainter dots, <1 is more conservative."""

    k_seed: float = 5.0
    """A blob is only kept if its peak reaches median + k_seed * robust_sigma
    of the local-contrast image. Noise blobs hover just above the background
    without ever producing a real peak, so this is what separates them.

    On synthetic plates precision falls off a cliff below about 4.5; 5.0 leaves
    headroom because real photographs have heavier-tailed noise than the
    simulation. Use --sensitivity rather than editing this."""

    k_grow: float = 3.0
    """Once seeded, the blob is grown out to median + k_grow * sigma, which is
    what makes the measured size and shape reflect the whole dot rather than
    just its core."""

    min_contrast: float = 6.0
    """Absolute floor on the seed threshold, in grey levels. Stops the
    statistical threshold from collapsing on a very clean, low-noise plate."""

    # --- shape gates --------------------------------------------------------
    min_circularity: float = 0.55
    circularity_min_area_px: int = 12
    """Circularity is meaningless for blobs of a handful of pixels, so it is
    only enforced above this area."""

    min_solidity: float = 0.70

    # --- artefact rejection -------------------------------------------------
    ink_delta: float = 60.0
    """A pixel darker than (plate median - ink_delta) is candidate marker ink."""

    ink_area_factor: float = 2.0
    """Ink-coloured regions bigger than this many maximum-colony areas are
    treated as handwriting rather than as colonies."""

    ink_length_factor: float = 3.0
    """...as are ink-coloured regions longer than this many maximum colony
    diameters, which catches thin marker strokes and scratches."""

    ink_dilate_mm: float = 1.0
    """Marker strokes are masked out with this much margin. Generous, because
    a thick out-of-focus stroke has a soft edge that reads as dark for some
    distance beyond the stroke itself."""

    glare_value: int = 250
    """Pixels at or above this are blown-out specular highlights."""

    glare_response: float | None = None
    """Optionally also mask highlights that never clip, at this many grey
    levels above their surroundings.

    Off by default: on the synthetic plates it consistently costs precision,
    because widening the mask shrinks the sample the noise estimate is drawn
    from, and the size filter already discards the large blobs that glare
    edges produce. Worth trying if a real photo has glare bad enough to
    survive that filter."""

    glare_area_factor: float = 4.0
    """Only bright regions bigger than this many maximum-colony areas are
    treated as glare, so that small bright features such as bubble rims stay
    out of the mask and are handled by the halo test instead."""

    glare_dilate_mm: float = 0.8

    reject_bubbles: bool = True
    bubble_halo: float = 14.0
    """Reject a dark blob if the ring around it is this many grey levels
    brighter than the local background -- the signature of an air bubble."""

    split_touching: bool = False
    """Watershed-split blobs that look like merged colonies."""

    def scaled(self) -> "Params":
        """Apply `sensitivity` to the threshold terms and return a copy."""
        s = max(self.sensitivity, 1e-3)
        out = Params(**{f.name: getattr(self, f.name) for f in fields(self)})
        out.k_seed = self.k_seed / s
        out.k_grow = self.k_grow / s
        out.min_contrast = self.min_contrast / s
        return out
