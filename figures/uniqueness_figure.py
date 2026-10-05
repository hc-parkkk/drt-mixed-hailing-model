import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
"""Fig. 4: number of solutions over the load factor and the e-hail share (Section 4.3).

Panels for 2, 3, 5, and 10 seats. Shading: 40 vehicles and an AV share of 0.4 (the AV share changes the
regions little). Lines: boundary of the two-solution region with 20 and 100 vehicles. Circle: case-study
reference. Input: outputs/tables/verif_uniqueness_detail.csv (scripts/verification_grid.py).
Output: outputs/figures/V1_en.(png|pdf)
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis_figures import CAT, INK, INK2, MUTED

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "figures"
ETAS = np.round(np.arange(0.1, 1.501, 0.1), 2)
PS = np.round(np.arange(0.0, 1.001, 0.1), 2)


def grid(d, capa, m, r=0.4):
    s = d[(d.capa == capa) & (d.m == m) & (d.r.round(1) == r)]
    z = np.zeros((len(PS), len(ETAS)))
    for row in s.itertuples():
        i, j = int(round(row.p * 10)), int(round(row.eta * 10)) - 1
        z[i, j] = 2 if row.cls == "ineff" else (1 if row.cls == "two" else 0)
    return z


def main():
    d = pd.read_csv(ROOT / "outputs" / "tables" / "verif_uniqueness_detail.csv")
    ineff = (d.n_distinct == 1) & (d.top_nE_frac < 1e-3)
    d["cls"] = np.where(d.n_distinct >= 2, "two", np.where(ineff, "ineff", "one"))
    cmap = ListedColormap(["#f3f2ed", CAT["yellow"], INK2])
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.5), sharey=True)
    ext = [0.05, 1.55, -0.05, 1.05]
    for ax, capa, tag in zip(axes, (2, 3, 5, 10), "abcd"):
        ax.imshow(grid(d, capa, 40.0), origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=2,
                  aspect="auto", interpolation="nearest")
        for m, ls in ((20.0, (0, (4, 2))), (100.0, (0, (1, 1.5)))):
            z = (grid(d, capa, m) >= 1).astype(float)
            ax.contour(ETAS, PS, z, levels=[0.5], colors=[INK], linewidths=0.9, linestyles=[ls])
        ax.plot(0.6, 0.6, marker="o", ms=4.5, mfc="white", mec=INK, mew=0.9, zorder=5)
        ax.set_xlim(0.05, 1.55); ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0.1, 0.5, 1.0, 1.5])
        ax.set_title(f"({tag}) {capa} seats", fontsize=10, color=INK, pad=3)
        ax.set_xlabel("Load factor $\\eta$", fontsize=10)
        ax.tick_params(labelsize=9, color=MUTED)
        for sp in ax.spines.values():
            sp.set_color(MUTED); sp.set_linewidth(0.6)
    axes[0].set_ylabel("E-hail share $\\theta_E$", fontsize=10)
    handles = [Patch(fc="#f3f2ed", ec=MUTED, lw=0.5, label="One solution"),
               Patch(fc=CAT["yellow"], label="Two solutions"),
               Patch(fc=INK2, label="Inefficient solution only"),
               Line2D([], [], color=INK, lw=0.9, ls=(0, (4, 2)), label="Two-solution boundary, 20 veh"),
               Line2D([], [], color=INK, lw=0.9, ls=(0, (1, 1.5)), label="Two-solution boundary, 100 veh"),
               Line2D([], [], marker="o", ms=4.5, mfc="white", mec=INK, ls="", label="Case-study reference")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout(w_pad=0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext_ in ("png", "pdf"):
        fig.savefig(OUT / f"V1_en.{ext_}", dpi=600, bbox_inches="tight")
    exp = os.environ.get("FIG_EXPORT_DIR")
    if exp:
        os.makedirs(exp, exist_ok=True)
        fig.savefig(os.path.join(exp, "V1_en.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(exp, "V1_en.png"), dpi=600, bbox_inches="tight")
    print("saved", OUT / "V1_en.png")


if __name__ == "__main__":
    main()
