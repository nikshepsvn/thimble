"""Regenerate the README figures. Run after any change to the headline numbers.

    python scripts/make_charts.py

Emits light and dark variants of each figure so the README can serve whichever
theme the reader is using (GitHub honours <picture> + prefers-color-scheme).
Numbers are hard-coded here deliberately: these are published results, and a
figure that silently follows a moving file is a figure you cannot cite.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"

# suite, ours, needle-2 — grouped by whether the catalog was in training
KNOWN = [("Mobile Actions", 86.3, 63.7),
         ("Mobile Actions\n2+ call rows", 73.5, 48.4),
         ("DroidCall", 52.5, 17.0),
         ("Seal-Tools\nin-domain", 33.1, 32.6)]
UNSEEN = [("Seal-Tools\nout-of-domain", 28.1, 28.7),
          ("BFCL v4\nsingle-turn", 23.5, 42.6)]

THEMES = {
    "light": dict(bg="#ffffff", fg="#14171f", muted="#5b6472", grid="#e6e9ee",
                  ours="#c8324c", ref="#9aa4b2", band="#eef0f4"),
    "dark":  dict(bg="#0d1117", fg="#e6edf3", muted="#9198a1", grid="#21262d",
                  ours="#f0506e", ref="#6e7681", band="#161b22"),
}


def hero(theme: str) -> None:
    c = THEMES[theme]
    rows = KNOWN + UNSEEN
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=170)
    fig.patch.set_facecolor(c["bg"]); ax.set_facecolor(c["bg"])

    # shade the "never seen" band so the split reads before any label does
    ax.axhspan(-0.6, len(UNSEEN) - 0.5, color=c["band"], zorder=0)

    ys = range(len(rows))
    labels = [r[0] for r in rows][::-1]
    ours = [r[1] for r in rows][::-1]
    ref = [r[2] for r in rows][::-1]
    h = 0.34
    ax.barh([y + h / 2 for y in ys], ours, height=h, color=c["ours"], zorder=3)
    ax.barh([y - h / 2 for y in ys], ref, height=h, color=c["ref"], zorder=3)

    for y, (o, r) in enumerate(zip(ours, ref)):
        ax.text(o + 1.2, y + h / 2, f"{o:.1f}", va="center", ha="left",
                color=c["fg"], fontsize=10.5, fontweight="bold", zorder=4)
        ax.text(r + 1.2, y - h / 2, f"{r:.1f}", va="center", ha="left",
                color=c["muted"], fontsize=10, zorder=4)

    ax.set_yticks(list(ys)); ax.set_yticklabels(labels, fontsize=10.5, color=c["fg"])
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(rows) - 0.02)  # headroom for the zone caption
    ax.set_xlabel("% of rows exactly correct", fontsize=10, color=c["muted"], labelpad=8)
    ax.tick_params(axis="x", colors=c["muted"], labelsize=9.5)
    ax.xaxis.grid(True, color=c["grid"], linewidth=1, zorder=1)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)

    # zone divider; each caption sits at the TOP of the zone it names
    split = len(UNSEEN) - 0.5
    ax.axhline(split, color=c["muted"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=5)
    ax.text(99, len(rows) - 0.12, "CATALOG REPRESENTED IN TRAINING", ha="right",
            va="top", fontsize=9.5, color=c["muted"], fontweight="bold", zorder=6)
    ax.text(99, split - 0.16, "CATALOG NEVER SEEN", ha="right", va="top",
            fontsize=9.5, color=c["muted"], fontweight="bold", zorder=6)

    ax.set_title("Accuracy is a function of catalog familiarity",
                 fontsize=16, fontweight="bold", color=c["fg"], loc="left", pad=34)
    ax.text(0, 1.045, "ordered strict exact match · Needle 2 (45M, 153B tokens) shown for scale",
            transform=ax.transAxes, fontsize=10, color=c["muted"], ha="left", va="bottom")
    ax.legend(handles=[Patch(color=c["ours"], label="Thimble v6  ·  48M, ~1B tokens"),
                       Patch(color=c["ref"], label="Needle 2  ·  45M, 153B tokens")],
              loc="lower right", frameon=False, fontsize=9.5,
              labelcolor=c["fg"], bbox_to_anchor=(1.0, 0.03))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / f"results-{theme}.png", facecolor=c["bg"])
    plt.close(fig)


def control(theme: str) -> None:
    """The controlled comparison: one suite, one model, catalogs swapped.

    Two series only, so the bars are labelled directly — a legend here would
    either float over the bars or cost a third of the figure.
    """
    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8.4, 3.6), dpi=170)
    fig.patch.set_facecolor(c["bg"]); ax.set_facecolor(c["bg"])
    pairs = [("row accuracy", 33.1, 28.1), ("tool-name\nsequence", 88.0, 79.0)]
    for i, (_, a, b) in enumerate(pairs):
        y = len(pairs) - 1 - i
        ax.barh(y + 0.19, a, height=0.34, color=c["ours"], zorder=3)
        ax.barh(y - 0.19, b, height=0.34, color=c["ref"], zorder=3)
        ax.text(a + 1.5, y + 0.19, f"{a:.1f}", va="center", color=c["fg"],
                fontsize=11, fontweight="bold", zorder=4)
        ax.text(b + 1.5, y - 0.19, f"{b:.1f}", va="center", color=c["muted"],
                fontsize=10.5, zorder=4)
        if i == 0:  # label the series once, on the top group
            ax.text(a + 8, y + 0.19, "in-domain catalogs", va="center",
                    color=c["ours"], fontsize=9.5, fontweight="bold", zorder=4)
            ax.text(b + 8, y - 0.19, "unseen catalogs", va="center",
                    color=c["muted"], fontsize=9.5, zorder=4)

    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([p[0] for p in pairs][::-1], fontsize=11, color=c["fg"])
    ax.set_xlim(0, 100); ax.set_ylim(-0.55, len(pairs) - 0.45)
    ax.tick_params(axis="x", colors=c["muted"], labelsize=9.5)
    ax.xaxis.grid(True, color=c["grid"], linewidth=1); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Same model, same metric, only the catalogs change",
                 fontsize=13.5, fontweight="bold", color=c["fg"], loc="left", pad=30)
    ax.text(0, 1.05, "Seal-Tools in-domain (700 rows) vs out-of-domain (654 rows), %",
            transform=ax.transAxes, fontsize=9.5, color=c["muted"], ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / f"catalog-control-{theme}.png", facecolor=c["bg"])
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for t in THEMES:
        hero(t); control(t)
        print(f"wrote assets/results-{t}.png, assets/catalog-control-{t}.png")
