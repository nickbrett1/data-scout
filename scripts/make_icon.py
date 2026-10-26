"""Generate the app icon — one dark square, three bars, a baseline.

Run with ``python3 scripts/make_icon.py``. The PNGs it writes are committed, so the
image build needs no font or drawing library; this script exists so the mark can be
changed deliberately rather than being a binary nobody can reproduce.

Why these choices:

* **No text.** A home-screen icon is 60 pt; a wordmark at that size is mush.
* **Three bars on a baseline** is the project in one glyph — measurement, not a lens
  and not a document. It also survives being shrunk to 32 px, which a thin ring
  does not.
* **Dark background** because the report stylesheet is dark-first in the evening and
  a near-black tile with a warm accent sits quietly next to system apps.
* **Full-bleed square with rounded corners drawn in** rather than relying on the OS
  mask: iOS masks to a squircle, Android to its own shape, and a browser bookmark
  tile to whatever the browser decides.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ASSETS = Path(__file__).resolve().parent.parent / "src" / "data_scout" / "assets"
CANVAS = 1024
SIZES = {"apple-touch-icon.png": 180, "icon-512.png": 512, "favicon-32.png": 32}

TOP = (27, 29, 34)
BOTTOM = (14, 15, 18)
BAR = (232, 180, 100)
BAR_MUTED = (150, 156, 166)
BASELINE = (62, 67, 76)


def _background(size: int) -> Image.Image:
    """A vertical gradient, drawn once per row."""
    image = Image.new("RGB", (size, size), TOP)
    draw = ImageDraw.Draw(image)
    for y in range(size):
        blend = y / (size - 1)
        colour = tuple(round(a + (b - a) * blend) for a, b in zip(TOP, BOTTOM, strict=True))
        draw.line([(0, y), (size, y)], fill=colour)
    return image


def draw_mark(size: int) -> Image.Image:
    image = _background(size)
    draw = ImageDraw.Draw(image)
    s = size

    # The baseline: the thing the bars are measured against.
    baseline_y = 0.735 * s
    draw.rounded_rectangle(
        [(0.215 * s, baseline_y), (0.785 * s, baseline_y + 0.032 * s)],
        radius=0.016 * s,
        fill=BASELINE,
    )

    # Three bars, deliberately uneven: a distribution, not a logo's three equal lines.
    bar_width = 0.132 * s
    gap = 0.058 * s
    start_x = 0.215 * s
    heights = (0.30, 0.47, 0.19)
    colours = (BAR_MUTED, BAR, BAR_MUTED)
    for index, (height, colour) in enumerate(zip(heights, colours, strict=True)):
        x0 = start_x + index * (bar_width + gap)
        top = baseline_y - height * s
        draw.rounded_rectangle(
            [(x0, top), (x0 + bar_width, baseline_y)],
            radius=0.03 * s,
            fill=colour,
        )

    # A little headroom marker: the tallest bar's cap, so the middle reads as the peak.
    cap_x0 = start_x + (bar_width + gap) - 0.012 * s
    draw.rounded_rectangle(
        [(cap_x0, baseline_y - 0.47 * s - 0.055 * s), (cap_x0 + bar_width + 0.024 * s, baseline_y - 0.47 * s - 0.031 * s)],
        radius=0.012 * s,
        fill=BAR,
    )
    return image


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    master = draw_mark(CANVAS)
    for name, size in SIZES.items():
        target = ASSETS / name
        master.resize((size, size), Image.LANCZOS).save(target, format="PNG", optimize=True)
        print(f"wrote {target.relative_to(ASSETS.parents[3])} ({size}x{size}, {target.stat().st_size} B)")
    master.resize((CANVAS, CANVAS), Image.LANCZOS).save(ASSETS / "icon-1024.png", format="PNG", optimize=True)
    print(f"wrote icon-1024.png ({CANVAS}x{CANVAS})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
