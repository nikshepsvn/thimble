"""Render the Thimble card, sized for social preview (1600x900, 16:9).

    python scripts/make_card.py

Drawn as a tailor's pattern sheet — a pattern sheet already IS a spec document,
so the vocabulary maps without forcing: benchmarks are a size chart, the
guarantees are notions, the five decode decisions are notches on a seam.

Sized for the feed, not the desk. An X card renders around 600px wide, so type is
set roughly twice the size a document would use and the content is cut to what
survives at that scale. Figures are set in Avenir Next, not the display face:
Gill Sans draws "1" as a bare stroke, which turns 48.12M into 48.I2M.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
W, H = 1600, 900

TISSUE, GRID = "#F1EDE4", "#E981D0"
TISSUE, GRID = "#F1EDE4", "#E0D9C9"
INK, MADDER, PENCIL = "#24406E", "#A63A2E", "#6E6960"

DISPLAY = ["Gill Sans", "Futura", "DejaVu Sans"]          # wordmark only
NUM = ["Avenir Next", "Avenir", "Helvetica Neue", "DejaVu Sans"]  # unambiguous figures
BODY = ["Charter", "Iowan Old Style", "Georgia", "DejaVu Serif"]
MONO = ["Andale Mono", "Menlo", "DejaVu Sans Mono"]

track = lambda s, n=1: (" " * n).join(s)

NOTIONS = [("circle", "Always well-formed JSON"),
           ("triangle", "Keys taken from your schema"),
           ("square", "Undeclared tools can't be called")]

# three rows, not six: at feed size the known/unseen contrast is the whole story
KNOWN = [("Mobile Actions", 86.3), ("Seal-Tools, in-domain", 33.1)]
UNSEEN = [("Seal-Tools, unseen catalogs", 28.1)]

SEAM = ["refuse?", "which tool", "optional?", "value", "stop?"]


def card() -> None:
    fig = plt.figure(figsize=(16, 9), dpi=110)
    fig.patch.set_facecolor(TISSUE)
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")

    L, R, MID = 78, W - 78, 850

    for x in range(0, W + 1, 50):
        ax.plot([x, x], [0, H], color=GRID, lw=0.7, zorder=0)
    for y in range(0, H + 1, 50):
        ax.plot([0, W], [y, y], color=GRID, lw=0.7, zorder=0)
    ax.add_patch(Rectangle((32, 32), W - 64, H - 64, fill=False, edgecolor=INK,
                           lw=1.8, linestyle=(0, (10, 7)), zorder=2))

    def eyebrow(x, y, text, color=MADDER, ha="left"):
        ax.text(x, y, text.upper(), family=MONO, fontsize=17,
                color=color, va="center", ha=ha, zorder=4)

    # --- header ------------------------------------------------------------
    ax.text(L, 838, track("THIMBLE", 2), family=DISPLAY, fontsize=62,
            color=INK, va="top", zorder=4)
    for i, r in enumerate((28, 19, 11)):
        ax.add_patch(Circle((R - 34, 792), r, fill=False, edgecolor=INK,
                            lw=1.4, zorder=3, alpha=0.85 - i * 0.2))
    ax.text(L, 736, "A tool-calling layer — not a language model.",
            family=BODY, fontsize=28, color=PENCIL, va="top", style="italic",
            zorder=4)
    ax.plot([L, R], [676, 676], color=INK, lw=2, zorder=3)

    # --- left: notions -----------------------------------------------------
    eyebrow(L, 640, "notions")
    for i, (mk, label) in enumerate(NOTIONS):
        y = 578 - i * 62
        cx = L + 14
        if mk == "circle":
            ax.add_patch(Circle((cx, y), 13, fill=False, edgecolor=MADDER, lw=2.2, zorder=4))
        elif mk == "triangle":
            ax.add_patch(Polygon([(cx - 13, y - 11), (cx + 13, y - 11), (cx, y + 14)],
                                 fill=False, edgecolor=MADDER, lw=2.2, zorder=4))
        else:
            ax.add_patch(Rectangle((cx - 12, y - 12), 24, 24, fill=False,
                                   edgecolor=MADDER, lw=2.2, zorder=4))
        ax.text(L + 54, y, label, family=BODY, fontsize=27, color=INK,
                va="center", zorder=4)
    ax.text(L, 386, "No training required: these come from the grammar,",
            family=BODY, fontsize=23, color=PENCIL, va="center", style="italic", zorder=4)
    ax.text(L, 352, "not from the weights.",
            family=BODY, fontsize=23, color=PENCIL, va="center", style="italic", zorder=4)

    # --- right: size chart -------------------------------------------------
    eyebrow(MID, 640, "size chart")

    def row(y, name, val, strong):
        ax.text(MID, y, name, family=BODY, fontsize=27,
                color=INK if strong else PENCIL, va="center", zorder=4)
        ax.plot([MID + 400, R - 112], [y, y], color=GRID, lw=1.6,
                linestyle=(0, (2, 4)), zorder=3)
        ax.text(R, y, f"{val:.1f}", family=NUM, fontsize=34,
                color=INK if strong else PENCIL, ha="right", va="center",
                fontweight="bold" if strong else "normal", zorder=4)

    ax.text(MID, 592, "CATALOG IN THE PATTERN", family=MONO, fontsize=16,
            color=INK, va="center", zorder=4)
    for i, (name, val) in enumerate(KNOWN):
        row(534 - i * 58, name, val, True)
    ax.plot([MID, R], [438, 438], color=GRID, lw=1.6, linestyle=(0, (6, 5)), zorder=3)
    ax.text(MID, 408, "NEVER SEEN", family=MONO, fontsize=16,
            color=PENCIL, va="center", zorder=4)
    for i, (name, val) in enumerate(UNSEEN):
        row(350 - i * 58, name, val, False)

    # --- base: the five notches -------------------------------------------
    eyebrow(L, 268, "stitching order · the only five decisions")
    sy = 186
    ax.plot([L, R], [sy, sy], color=INK, lw=2, linestyle=(0, (8, 6)), zorder=3)
    seg = (R - L) / (len(SEAM) - 1)
    for i, step in enumerate(SEAM):
        x = L + i * seg
        ax.plot([x, x], [sy - 15, sy + 15], color=INK, lw=2.2, zorder=4)
        ax.add_patch(Circle((x, sy), 8, facecolor=TISSUE, edgecolor=INK,
                            lw=2.2, zorder=5))
        ha = "left" if i == 0 else ("right" if i == len(SEAM) - 1 else "center")
        ax.text(x, sy - 32, step, family=BODY, fontsize=25, color=INK,
                ha=ha, va="top", zorder=4)
    ax.text(L, 82, "48.12M parameters · 768-token context · MIT",
            family=MONO, fontsize=17, color=PENCIL, va="center", zorder=4)
    ax.text(R, 82, "huggingface.co/flashvenom/thimble", family=MONO,
            fontsize=17, color=INK, ha="right", va="center", zorder=4)

    out = ROOT / "assets" / "model-card.png"
    fig.savefig(out, facecolor=TISSUE)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}  ({W}x{H}, 16:9)")


if __name__ == "__main__":
    card()
