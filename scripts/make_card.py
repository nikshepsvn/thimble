"""Render the Thimble spec card — a single image that explains the model.

    python scripts/make_card.py

Laid out as a precision spec plate rather than a marketing banner: the subject is
an instrument with a contract, so the card leads with what is guaranteed, then
what is measured, then the mechanism that produces both.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
W, H = 1600, 1000

INK, PANEL, RULE = "#0E1116", "#151A21", "#232B36"
FG, MUTED, DIM = "#E8EDF3", "#8894A5", "#5C6675"
ACCENT, JADE = "#E23E58", "#3FB68B"

DISPLAY = ["Futura", "Avenir Next", "Helvetica Neue", "DejaVu Sans"]
BODY = ["Avenir Next", "Avenir", "Helvetica Neue", "DejaVu Sans"]
MONO = ["JetBrains Mono", "Menlo", "Monaco", "DejaVu Sans Mono"]

# matplotlib has no letter-spacing; spacing the characters is the standard trick
track = lambda s, n=1: (" " * n).join(s)

STATS = [("48.12M", "PARAMETERS"), ("768", "TOKEN CONTEXT"),
         ("100%", "WELL-FORMED JSON"), ("MIT", "LICENSE")]

CONTRACT = ["Output is always well-formed JSON",
            "Argument keys come from your schema",
            "Undeclared tools cannot be called"]

BARS = [("Mobile Actions", 86.3, True), ("… 2+ call rows", 73.5, True),
        ("DroidCall", 52.5, True), ("Seal-Tools in-domain", 33.1, True),
        ("Seal-Tools out-of-domain", 28.1, False), ("BFCL v4 single-turn", 23.5, False)]

STEPS = ["refuse or call", "which tool", "include optional", "what value", "stop or continue"]


def card() -> None:
    fig = plt.figure(figsize=(16, 10), dpi=110)
    fig.patch.set_facecolor(INK)
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")

    def rule(y, x0=70, x1=W - 70, color=RULE, lw=1.1):
        ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=2)

    # every multi-line block is positioned line by line: matplotlib's linespacing
    # made blocks taller than the layout budgeted and they collided with what
    # followed, so nothing here relies on implicit text height
    def lines(x, y, rows, step, **kw):
        for i, t in enumerate(rows):
            ax.text(x, y - i * step, t, va="top", **kw)

    # ---- header -----------------------------------------------------------
    ax.plot([70, 70], [H - 148, H - 66], color=ACCENT, lw=5, solid_capstyle="butt")
    ax.text(92, H - 68, "THIMBLE", family=DISPLAY, fontsize=56, color=FG, va="top")
    ax.text(92, H - 170, "A tool-calling layer, not a language model.",
            family=BODY, fontsize=19, color=MUTED, va="top")
    lines(W - 70, H - 88, [track("48M PARAMETER"), track("FUNCTION CALLER")], 24,
          family=MONO, fontsize=11.5, color=DIM, ha="right")
    rule(H - 218)

    # ---- stat strip -------------------------------------------------------
    cell = (W - 140) / 4
    for i, (val, lab) in enumerate(STATS):
        x = 70 + i * cell
        if i:
            ax.plot([x, x], [H - 230, H - 322], color=RULE, lw=1.1)
        ax.text(x + 28, H - 244, val, family=DISPLAY, fontsize=36, color=FG, va="top")
        ax.text(x + 28, H - 302, track(lab), family=MONO, fontsize=10.5, color=DIM, va="top")
    rule(H - 348)

    # ---- left: the contract ----------------------------------------------
    ax.text(70, H - 390, track("THE CONTRACT"), family=MONO, fontsize=12,
            color=ACCENT, va="top")
    lines(70, H - 424, ["Holds on any catalog, with no training.",
                        "These come from the grammar, not the weights."], 26,
          family=BODY, fontsize=14.5, color=MUTED)
    for i, line in enumerate(CONTRACT):
        y = H - 512 - i * 46
        ax.text(72, y, "✓", family=BODY, fontsize=17, color=JADE, va="center")
        ax.text(104, y, line, family=BODY, fontsize=16, color=FG, va="center")
    # ---- left, lower: it in use ------------------------------------------
    # panel is pinned in absolute y so it always clears the base rule at 196
    ax.text(70, H - 660, track("IN PRACTICE"), family=MONO, fontsize=12,
            color=ACCENT, va="top")
    ax.add_patch(FancyBboxPatch((70, 216), 590, 108,
                                boxstyle="round,pad=0,rounding_size=6",
                                facecolor=PANEL, edgecolor=RULE, lw=1.1, zorder=1))
    ax.text(92, 300, '$ demo.py "text Sam that dinner is on"',
            family=MONO, fontsize=12.5, color=MUTED, va="center", zorder=2)
    for i, ln in enumerate(['[{"name": "sendMessage",',
                            '   "arguments": {"body": "dinner is on",',
                            '                 "contact": "Sam"}}]']):
        ax.text(92, 272 - i * 24, ln, family=MONO, fontsize=12.5, color=FG,
                va="center", zorder=2)

    # ---- right: measured accuracy ----------------------------------------
    bx, bw = 720, W - 790
    ax.text(bx, H - 390, track("MEASURED"), family=MONO, fontsize=12,
            color=ACCENT, va="top")
    lines(bx, H - 424, ["Ordered strict exact match — the names, the order,",
                        "and every argument value must match."], 26,
          family=BODY, fontsize=14.5, color=MUTED)
    ys = [H - 512, H - 550, H - 588, H - 626, H - 706, H - 744]
    for i, ((name, val, known), y) in enumerate(zip(BARS, ys)):
        if i == 4:  # zone break, with the caption in its own gap below the rule
            ax.plot([bx, bx + bw], [H - 664, H - 664], color=DIM, lw=1,
                    linestyle=(0, (4, 3)))
            ax.text(bx + bw, H - 674, track("CATALOG NEVER SEEN"), family=MONO,
                    fontsize=9.5, color=DIM, ha="right", va="top")
        ax.text(bx, y, name, family=BODY, fontsize=13.5,
                color=FG if known else MUTED, va="center")
        x0, full = bx + 260, bw - 330
        ax.add_patch(Rectangle((x0, y - 7), full, 14, facecolor=RULE, zorder=2))
        ax.add_patch(Rectangle((x0, y - 7), full * val / 100, 14,
                               facecolor=ACCENT if known else DIM, zorder=3))
        ax.text(bx + bw, y, f"{val:.1f}", family=MONO, fontsize=13.5,
                color=FG if known else MUTED, ha="right", va="center")
    ax.text(bx, 224, "Catalog familiarity is the variable — adapting it to yours takes hours, not weeks.",
            family=BODY, fontsize=13.5, color=DIM, va="center")

    # ---- base: the five decisions ----------------------------------------
    rule(196)
    ax.text(70, 170, track("THE MODEL IS ASKED FIVE QUESTIONS"), family=MONO,
            fontsize=12, color=ACCENT, va="top")
    y, seg = 96, (W - 140) / len(STEPS)
    for i, step in enumerate(STEPS):
        x = 70 + i * seg
        ax.add_patch(FancyBboxPatch((x, y - 26), seg - 30, 54,
                                    boxstyle="round,pad=0,rounding_size=6",
                                    facecolor=PANEL, edgecolor=RULE, lw=1.1, zorder=1))
        ax.text(x + 20, y + 10, f"0{i+1}", family=MONO, fontsize=10.5,
                color=ACCENT, va="center", zorder=2)
        ax.text(x + 20, y - 12, step, family=BODY, fontsize=14, color=FG,
                va="center", zorder=2)
        if i < len(STEPS) - 1:
            ax.text(x + seg - 15, y, "→", family=BODY, fontsize=15, color=DIM,
                    ha="center", va="center")
    ax.text(70, 38, "Everything else — braces, quotes, commas, every argument key — is emitted by the grammar without consulting the model.",
            family=BODY, fontsize=13, color=DIM, va="center")
    ax.text(W - 70, 38, "huggingface.co/flashvenom/thimble", family=MONO,
            fontsize=12, color=MUTED, ha="right", va="center")

    out = ROOT / "assets" / "model-card.png"
    fig.savefig(out, facecolor=INK)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    card()
