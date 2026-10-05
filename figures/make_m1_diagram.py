import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
# -*- coding: utf-8 -*-
"""Figure 1 of the paper: the workload-state network for capa = 3."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

plt.rcParams["font.family"] = ["Times New Roman", "Malgun Gothic"]
plt.rcParams["mathtext.fontset"] = "stix"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "figures")
os.makedirs(OUT, exist_ok=True)

CAPA = 3
C_ASSIGN = "#2a78d6"
C_PICK = "#008300"
C_DROP = "#52514e"
C_ABS = "#e34948"
INK = "#0b0b0b"
NODE_EDGE = "#0b0b0b"
NODE_FILL = "#fcfcfb"
NODE_FULL = "#e8e6df"

DX, DY = 2.05, 1.75
RAD = 0.40


def pos(i, j):
    return i * DX, -j * DY


def arrow(ax, s, e, color, patchA, patchB, curve=0.0, lw=2.4, ls="-", z=3):
    a = FancyArrowPatch(s, e, arrowstyle="-|>", mutation_scale=22,
                        connectionstyle=f"arc3,rad={curve}", lw=lw,
                        color=color, linestyle=ls, zorder=z,
                        patchA=patchA, patchB=patchB, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    states = [(i, j) for i in range(CAPA + 1) for j in range(CAPA + 1)
              if i + j <= CAPA]
    circles = {}
    for (i, j) in states:
        x, y = pos(i, j)
        full = (i + j == CAPA)
        circles[(i, j)] = Circle((x, y), RAD,
                                 facecolor=NODE_FULL if full else NODE_FILL,
                                 edgecolor=NODE_EDGE, lw=1.5, zorder=5)

    for (i, j) in states:
        if i + j < CAPA:
            arrow(ax, pos(i, j), pos(i, j + 1), C_ASSIGN,
                  circles[(i, j)], circles[(i, j + 1)], curve=0.0, lw=2.6)
        if j >= 1:
            arrow(ax, pos(i, j), pos(i + 1, j - 1), C_PICK,
                  circles[(i, j)], circles[(i + 1, j - 1)], curve=0.0, lw=2.6)
        if i >= 1:
            arrow(ax, pos(i, j), pos(i - 1, j), C_DROP,
                  circles[(i, j)], circles[(i - 1, j)], curve=0.0, lw=2.6)
        if i + j < CAPA:
            arrow(ax, pos(i, j), pos(i + 1, j), C_ABS,
                  circles[(i, j)], circles[(i + 1, j)], curve=0.30,
                  lw=2.6, ls=(0, (2, 2)))

    for (i, j) in states:
        x, y = pos(i, j)
        ax.add_patch(circles[(i, j)])
        ax.text(x, y, f"({i},{j})", ha="center", va="center",
                fontsize=11.5, zorder=6, color=INK)

    handles = [
        plt.Line2D([], [], marker="o", mfc=NODE_FILL, mec=NODE_EDGE, mew=1.2,
                   ms=14, linestyle="None", label="available state ($i{+}j < c$)"),
        plt.Line2D([], [], marker="o", mfc=NODE_FULL, mec=NODE_EDGE, mew=1.2,
                   ms=14, linestyle="None", label="saturated state ($i{+}j = c$)"),
        plt.Line2D([], [], color=C_ASSIGN, lw=2.6,
                   label="assignment $a_{ij}$: $(i,j)\\to(i,j{+}1)$"),
        plt.Line2D([], [], color=C_PICK, lw=2.6,
                   label="pick-up $p_{ij}$: $(i,j)\\to(i{+}1,j{-}1)$"),
        plt.Line2D([], [], color=C_DROP, lw=2.6,
                   label="drop-off $d_{ij}$: $(i,j)\\to(i{-}1,j)$"),
        plt.Line2D([], [], color=C_ABS, lw=2.6, ls=(0, (2, 2)),
                   label="absorption $\\Phi_{ij}$: $(i,j)\\to(i{+}1,j)$"),
    ]
    anchor_x, anchor_y = CAPA * DX + 0.9, -CAPA * DY / 2.0
    ax.legend(handles=handles, loc="center left",
             bbox_to_anchor=(anchor_x, anchor_y), bbox_transform=ax.transData,
             ncol=1, fontsize=12, frameon=False, handlelength=2.2,
             labelspacing=0.9)

    ax.set_xlim(-1.3, CAPA * DX + 6.3)
    ax.set_ylim(-CAPA * DY - 0.6, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    out = os.path.join(OUT, "M1.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    exp = os.environ.get("FIG_EXPORT_DIR")
    if exp:
        os.makedirs(exp, exist_ok=True)
        for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
            fig.savefig(os.path.join(exp, f"M1.{ext}"), bbox_inches="tight", facecolor="white", **kw)
    print("saved", out)


if __name__ == "__main__":
    main()
