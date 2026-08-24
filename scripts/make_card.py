"""Render the Thimble spec card.

    python scripts/make_card.py

Drawn as a tailor's pattern sheet, because a pattern sheet already *is* a spec
document — finished measurements, a size chart, a notions legend, and a stitching
order. The model's own vocabulary maps onto it without forcing: benchmarks are
sizes, the guarantees are notions, and the five decode decisions are notches
along a seam.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
W, H = 1600, 1150

TISSUE, GRID = "#F1EDE4", "#E3DDD0"
INK, MADDER, PENCIL = "#24406E", "#A63A2E", "#77726A"

DISPLAY = ["Gill Sans", "Gill Sans MT", "Futura", "DejaVu Sans"]
BODY = ["Charter", "Iowan Old Style", "Georgia", "DejaVu Serif"]
MONO = ["Andale Mono", "Menlo", "DejaVu Sans Mono"]

track = lambda s, n=1: (" " * n).join(s)

MEASURES = [("48.12M", "parameters"), ("768", "token context"),
            ("5", "decisions per call"), ("100%", "well-formed JSON")]

# the size chart: benchmark rows, grouped the way the model actually behaves
SIZES = [("Mobile Actions", "961 rows", 86.3),
         ("  · rows needing 2+ calls", "", 73.5),
         ("DroidCall", "200 rows", 52.5),
         ("Seal-Tools, in-domain", "700 rows", 33.1)]
SIZES_OOD = [("Seal-Tools, out-of-domain", "654 rows", 28.1),
             ("BFCL v4, single-turn", "3,641 rows", 23.5)]

NOTIONS = [("well-formed JSON", "always — malformed output is unreachable"),
           ("argument keys", "taken from your schema, never invented"),
           ("undeclared tools", "cannot be called at all")]

SEAM = [("refuse", "or call"), ("which", "tool"), ("include", "this optional"),
        ("what", "value"), ("stop", "or continue")]


def card() -> None:
    fig = plt.figure(figsize=(16, 11.5), dpi=110)
    fig.patch.set_facecolor(TISSUE)
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")

    # every y below is an explicit pixel: matplotlib's implicit text height put
    # blocks on top of each other in the first pass, so nothing is inferred
    def rule(y, x0, x1, color=INK, lw=1, ls="solid"):
        ax.plot([x0, x1], [y, y], color=color, lw=lw, linestyle=ls, zorder=3)

    def eyebrow(x, y, text, color=MADDER):
        ax.text(x, y, track(text.upper()), family=MONO, fontsize=10.5,
                color=color, va="top", zorder=4)

    def lines(x, y, rows, step, **kw):
        for i, t in enumerate(rows):
            ax.text(x, y - i * step, t, va="top", zorder=4, **kw)

    L, R, MID = 86, W - 86, 792

    # --- pattern paper -----------------------------------------------------
    for x in range(0, W + 1, 40):
        ax.plot([x, x], [0, H], color=GRID, lw=0.6, zorder=0)
    for y in range(0, H + 1, 40):
        ax.plot([0, W], [y, y], color=GRID, lw=0.6, zorder=0)
    ax.add_patch(Rectangle((38, 38), W - 76, H - 76, fill=False, edgecolor=INK,
                           lw=1.4, linestyle=(0, (9, 6)), zorder=2))

    # --- header ------------------------------------------------------------
    ax.text(L, 1072, track("THIMBLE", 3), family=DISPLAY, fontsize=52,
            color=INK, va="top", zorder=4)
    ax.text(L, 978, "A tool-calling layer — not a language model.",
            family=BODY, fontsize=20, color=PENCIL, va="top", style="italic", zorder=4)
    ax.add_patch(Rectangle((R - 250, 1000), 250, 78, fill=False,
                           edgecolor=INK, lw=1.1, zorder=3))
    ax.text(R - 125, 1058, track("PATTERN"), family=MONO, fontsize=10,
            color=PENCIL, ha="center", va="center", zorder=4)
    ax.text(R - 125, 1026, track("No. V6"), family=DISPLAY, fontsize=22,
            color=INK, ha="center", va="center", zorder=4)
    for i, r in enumerate((26, 18, 10)):
        ax.add_patch(Circle((R - 300, 1038), r, fill=False, edgecolor=INK,
                            lw=1.1, zorder=3, alpha=0.9 - i * 0.22))
    rule(932, L, R, lw=1.4)

    # --- finished measurements --------------------------------------------

    eyebrow(L, 908, "finished measurements")
    cw = (R - L) / 4
    for i, (val, lab) in enumerate(MEASURES):
        x = L + i * cw
        if i:
            ax.plot([x - 20, x - 20], [794, 878], color=GRID, lw=1.2, zorder=3)
        ax.text(x, 878, val, family=DISPLAY, fontsize=36, color=INK, va="top", zorder=4)
        ax.text(x, 816, lab, family=BODY, fontsize=14.5, color=PENCIL, va="top", zorder=4)
    rule(776, L, R, color=GRID, lw=1.2)

    # --- left column: size chart ------------------------------------------
    eyebrow(L, 750, "size chart")
    lines(L, 722, ["Ordered strict exact match: every function name, the",
                   "call order, and every argument value must match."], 24,
          family=BODY, fontsize=13.5, color=PENCIL)

    def size_row(y, name, n, val, strong):
        ax.text(L, y, name, family=BODY, fontsize=15,
                color=INK if strong else PENCIL, va="center", zorder=4)
        ax.text(MID - 168, y, n, family=MONO, fontsize=11, color=PENCIL,
                ha="right", va="center", zorder=4)
        ax.plot([MID - 156, MID - 76], [y, y], color=GRID, lw=1.2,
                linestyle=(0, (1, 3)), zorder=3)
        ax.text(MID - 62, y, f"{val:.1f}", family=MONO, fontsize=16,
                color=INK if strong else PENCIL, ha="right", va="center", zorder=4)

    ax.text(L, 640, track("CATALOG IN THE PATTERN"), family=MONO, fontsize=9.5,
            color=INK, va="center", zorder=4)
    for i, (name, n, val) in enumerate(SIZES):
        size_row(604 - i * 36, name, n, val, True)
    rule(470, L, MID - 62, color=GRID, lw=1.2, ls=(0, (5, 4)))
    ax.text(L, 444, track("CATALOG NEVER SEEN"), family=MONO, fontsize=9.5,
            color=PENCIL, va="center", zorder=4)
    for i, (name, n, val) in enumerate(SIZES_OOD):
        size_row(408 - i * 36, name, n, val, False)
    ax.text(L, 336, "Fit depends on the catalog, not on the size.",
            family=BODY, fontsize=13.5, color=PENCIL, va="center",
            style="italic", zorder=4)

    # --- right column: notions --------------------------------------------
    eyebrow(MID, 750, "notions · included with every catalog")
    lines(MID, 722, ["These hold with no training on your tools at all —",
                     "they come from the grammar, not from the weights."], 24,
          family=BODY, fontsize=13.5, color=PENCIL)

    for i, ((title, detail), mk) in enumerate(zip(NOTIONS, ("circle", "triangle", "square"))):
        y = 648 - i * 72
        cx = MID + 12
        if mk == "circle":
            ax.add_patch(Circle((cx, y), 9, fill=False, edgecolor=MADDER, lw=1.6, zorder=4))
        elif mk == "triangle":
            ax.add_patch(Polygon([(cx - 9, y - 8), (cx + 9, y - 8), (cx, y + 10)],
                                 fill=False, edgecolor=MADDER, lw=1.6, zorder=4))
        else:
            ax.add_patch(Rectangle((cx - 8, y - 8), 16, 16, fill=False,
                                   edgecolor=MADDER, lw=1.6, zorder=4))
        ax.text(MID + 44, y + 12, title, family=DISPLAY, fontsize=17, color=INK,
                va="center", zorder=4)
        ax.text(MID + 44, y - 12, detail, family=BODY, fontsize=13.5,
                color=PENCIL, va="center", zorder=4)

    ax.add_patch(Rectangle((MID, 306), R - MID, 140, facecolor="#EAE5D9",
                           edgecolor=GRID, lw=1.2, zorder=3))
    ax.text(MID + 24, 420, track("SAMPLE STITCH"), family=MONO, fontsize=9.5,
            color=PENCIL, va="center", zorder=4)
    ax.text(MID + 24, 392, '$ demo.py "text Sam that dinner is on"',
            family=MONO, fontsize=12, color=PENCIL, va="center", zorder=4)
    for i, ln in enumerate(['[{"name": "sendMessage",',
                            '   "arguments": {"body": "dinner is on",',
                            '                 "contact": "Sam"}}]']):
        ax.text(MID + 24, 360 - i * 24, ln, family=MONO, fontsize=12,
                color=INK, va="center", zorder=4)

    # --- stitching order ---------------------------------------------------
    eyebrow(L, 292, "stitching order · the only points the model is consulted")
    ax.add_patch(FancyArrow(L, 244, R - L, 0, width=0.6, head_width=9,
                            head_length=16, length_includes_head=True,
                            color=PENCIL, zorder=4))
    ax.text((L + R) / 2, 262, track("ONE CALL"), family=MONO, fontsize=9.5,
            color=PENCIL, ha="center", va="bottom", zorder=4)
    sy = 190
    rule(sy, L, R, color=INK, lw=1.3, ls=(0, (7, 5)))
    seg = (R - L) / (len(SEAM) - 1)
    for i, (top, bot) in enumerate(SEAM):
        x = L + i * seg
        ax.plot([x, x], [sy - 11, sy + 11], color=INK, lw=1.6, zorder=4)
        ax.add_patch(Circle((x, sy), 5.5, facecolor=TISSUE, edgecolor=INK,
                            lw=1.6, zorder=5))
        ha = "left" if i == 0 else ("right" if i == len(SEAM) - 1 else "center")
        dx = 0 if ha == "center" else (-4 if ha == "right" else 4)
        ax.text(x + dx, 162, top, family=BODY, fontsize=14, color=INK,
                ha=ha, va="top", zorder=4)
        ax.text(x + dx, 138, bot, family=BODY, fontsize=14, color=INK,
                ha=ha, va="top", zorder=4)

    ax.text(L, 78, "Everything between the notches — braces, quotes, commas, every argument key — is emitted by the grammar, unasked.",
            family=BODY, fontsize=13.5, color=PENCIL, va="center",
            style="italic", zorder=4)
    ax.text(R, 78, "huggingface.co/flashvenom/thimble", family=MONO,
            fontsize=11.5, color=INK, ha="right", va="center", zorder=4)

    out = ROOT / "assets" / "model-card.png"
    fig.savefig(out, facecolor=TISSUE)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    card()
