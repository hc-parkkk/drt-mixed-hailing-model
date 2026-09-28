import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
# -*- coding: utf-8 -*-
"""Figures 2-10 of the paper (run with --en for English labels). Usage: python analysis_figures.py D1 --en"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as _mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drtmodel.model import HybridDRTModel, HybridDRTParams
from drtmodel.optimize import evaluate

plt.rcParams["font.family"] = ["Times New Roman", "Malgun Gothic"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

LANG = "ko"


def T(ko, en):
    """T (see the paper for the definitions)."""
    return en if LANG == "en" else ko


def set_lang(lang):
    """set_lang (see the paper for the definitions)."""
    global LANG
    LANG = lang

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "figures")
os.makedirs(OUT, exist_ok=True)

_R_DONGTAN = 33.04
_L_DONGTAN = 3.6213
BASE = HybridDRTParams(
    m=500.0, r=0.0, p=0.6,
    lam=0.6 * 33.0 * 100.0 / (_R_DONGTAN * _L_DONGTAN),
    R=_R_DONGTAN, v=33.0,
    capa=3, kappa=0.63, Delta1=0.586, Delta2=0.586,
    wt_max=60.0 / 60.0, de_max=1.0, te_max=60.0 / 60.0,
    availability_rule="a", flexible_dropoff=True, balking=True,
)

CAT = {
    "blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100",
    "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
    "magenta": "#e87ba4", "orange": "#eb6834",
}
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

CAPA_ROWS = [1, 5, 10]
CAPA_COLORS = [CAT["red"], CAT["aqua"], CAT["blue"]]


def _panel_tag(ax, letter, corner=None, linespacing=None, labelpad=None):
    """_panel_tag (see the paper for the definitions)."""
    ax._panel_letter = f"({letter})"
    ax._panel_linespacing = linespacing
    ax._panel_labelpad = labelpad


def _colorbar_title(cb, text, fontsize=8, ticklabelsize=7):
    """_colorbar_title (see the paper for the definitions)."""
    cb.ax.tick_params(labelsize=ticklabelsize)
    cb.ax.set_title(text, fontsize=fontsize, color=INK, loc="center", pad=6)


def _recenter_colorbar_title(fig, cb):
    """_recenter_colorbar_title (see the paper for the definitions)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_disp = cb.ax.get_tightbbox(renderer)
    inv = cb.ax.transAxes.inverted()
    x0_af = inv.transform((bbox_disp.x0, 0))[0]
    x1_af = inv.transform((bbox_disp.x1, 0))[0]
    cb.ax.title.set_x((x0_af + x1_af) / 2.0)


def _apply_panel_tags(fig):
    """_apply_panel_tags (see the paper for the definitions)."""
    for ax in fig.axes:
        letter = getattr(ax, "_panel_letter", None)
        if letter is None:
            continue
        xl = ax.get_xlabel()
        linespacing = getattr(ax, "_panel_linespacing", None) or 1.5
        labelpad = getattr(ax, "_panel_labelpad", None)
        labelpad = 3 if labelpad is None else labelpad
        ax.set_xlabel(f"{xl}\n{letter}" if xl else letter,
                      linespacing=linespacing, labelpad=labelpad)
        ax._panel_letter = None


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _save(fig, fig_id, df):
    suffix = "_en" if LANG == "en" else ""
    png = os.path.join(OUT, f"{fig_id}{suffix}.png")
    csv = os.path.join(OUT, f"{fig_id}{suffix}.csv")
    _apply_panel_tags(fig)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    exp = os.environ.get("FIG_EXPORT_DIR")
    if exp:
        os.makedirs(exp, exist_ok=True)
        for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
            fig.savefig(os.path.join(exp, f"{fig_id}{suffix}.{ext}"), bbox_inches="tight",
                        pad_inches=0.15, facecolor="white", **kw)
    plt.close(fig)
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    print(f"saved {fig_id}{suffix}: {png}  ({len(df)} rows)")


def _trim_precliff_crossover(df, group_cols, x_col, y_col, ceiling):
    """_trim_precliff_crossover (see the paper for the definitions)."""
    return df[df[y_col] <= ceiling].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_A3():
    r_panels = [0.0, 0.5]
    p_grid = np.linspace(0.02, 0.98, 40)
    m = 150.0
    COL = {"E": CAT["red"], "S": CAT["green"], "T": CAT["blue"]}

    rows = []
    for capa in CAPA_ROWS:
        for r in r_panels:
            for p in p_grid:
                res = evaluate(BASE, p=float(p), capa=capa, r=r, m=m)
                T_w_E = res["T_w_E"] or 0.0
                T_w_S = res["T_w_S"] or 0.0
                T_w = p * T_w_E + (1 - p) * T_w_S
                rows.append(dict(capa=capa, r=r, p=p, converged=res["converged"],
                                 T_w_E_min=T_w_E * 60, T_w_S_min=T_w_S * 60, T_w_min=T_w * 60))
    df = pd.DataFrame(rows)
    df_ok = df[df.converged]
    y_top = df_ok[["T_w_E_min", "T_w_S_min", "T_w_min"]].to_numpy().max() * 1.05

    fig, axes = plt.subplots(len(CAPA_ROWS), len(r_panels),
                             figsize=(9.6, 4.0 * len(CAPA_ROWS)), sharex=True, sharey=True)
    for i, capa in enumerate(CAPA_ROWS):
        for j, r in enumerate(r_panels):
            ax = axes[i, j]
            sub = df_ok[(df_ok.capa == capa) & (df_ok.r == r)].sort_values("p")
            ax.plot(sub.p, sub.T_w_E_min, color=COL["E"], lw=2.0, label="e-hail 대기시간")
            ax.plot(sub.p, sub.T_w_S_min, color=COL["S"], lw=2.0, label="street-hail 대기시간")
            ax.plot(sub.p, sub.T_w_min, color=COL["T"], lw=2.0, label="전체 대기시간")
            ax.set_title(f"정원={capa}{'  (T-ITS 원형)' if capa==1 else ''},  AT비율={r:g}",
                        fontsize=11, color=INK)
            if i == len(CAPA_ROWS) - 1:
                ax.set_xlabel("e-hailing 비율")
            if j == 0:
                ax.set_ylabel("대기시간 [분]")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, y_top)
            _style_axes(ax)

    handles = [plt.Line2D([], [], color=COL["E"], lw=2.2, label="e-hail 대기시간"),
              plt.Line2D([], [], color=COL["S"], lw=2.2, label="street-hail 대기시간"),
              plt.Line2D([], [], color=COL["T"], lw=2.2, label="전체 대기시간 ([T-ITS] 결합식)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
              frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"A3 · [T-ITS Fig 9 구조] 호출유형별 대기시간 vs e-hailing 비율 — 대전 파라미터, 차량대수={m:.0f}",
                fontsize=13, y=1.01, color=INK)
    fig.tight_layout()
    return df, fig


P_LEVELS_6 = [0.999, 0.8, 0.6, 0.4, 0.2, 0.001]
P_LABELS_6 = ["1", "0.8", "0.6", "0.4", "0.2", "0"]
GRAYS_6 = ["#0b0b0b", "#3a3a38", "#5c5c59", "#82827d", "#aaa9a2", "#cfcec6"]


def _combined_Tw(res, p):
    T_w_E = res["T_w_E"] or 0.0
    T_w_S = res["T_w_S"] or 0.0
    return p * T_w_E + (1 - p) * T_w_S


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_A1():
    m_panels = [100, 200]
    p_grid = np.linspace(0.02, 0.98, 40)
    COL = {"E": CAT["red"], "S": CAT["green"], "T": CAT["blue"]}

    rows = []
    for capa in CAPA_ROWS:
        for m in m_panels:
            for p in p_grid:
                res = evaluate(BASE, p=float(p), capa=capa, r=0.0, m=float(m))
                rows.append(dict(capa=capa, m=m, p=p, converged=res["converged"],
                                 T_w_E_min=(res["T_w_E"] or 0.0) * 60,
                                 T_w_S_min=(res["T_w_S"] or 0.0) * 60,
                                 T_w_min=_combined_Tw(res, p) * 60))
    df = pd.DataFrame(rows)
    df_ok = df[df.converged]
    y_top = df_ok[["T_w_E_min", "T_w_S_min", "T_w_min"]].to_numpy().max() * 1.05

    fig, axes = plt.subplots(len(CAPA_ROWS), len(m_panels),
                             figsize=(9.6, 4.0 * len(CAPA_ROWS)), sharex=True, sharey=True)
    for i, capa in enumerate(CAPA_ROWS):
        for j, m in enumerate(m_panels):
            ax = axes[i, j]
            sub = df_ok[(df_ok.capa == capa) & (df_ok.m == m)].sort_values("p")
            ax.plot(sub.p, sub.T_w_E_min, color=COL["E"], lw=2.0)
            ax.plot(sub.p, sub.T_w_S_min, color=COL["S"], lw=2.0)
            ax.plot(sub.p, sub.T_w_min, color=COL["T"], lw=2.0)
            ax.set_title(f"정원={capa}{'  (T-ITS 원형)' if capa==1 else ''},  차량대수={m}",
                        fontsize=11, color=INK)
            if i == len(CAPA_ROWS) - 1:
                ax.set_xlabel("e-hailing 비율")
            if j == 0:
                ax.set_ylabel("대기시간 [분]")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, y_top)
            _style_axes(ax)

    handles = [plt.Line2D([], [], color=COL["E"], lw=2.2, label="e-hail 대기시간"),
              plt.Line2D([], [], color=COL["S"], lw=2.2, label="street-hail 대기시간"),
              plt.Line2D([], [], color=COL["T"], lw=2.2, label="전체 대기시간 ([T-ITS] 결합식)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
              frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("A1 · [T-ITS Fig 6 구조] 호출유형별 대기시간 vs e-hailing 비율 (AT비율=0) — 대전 파라미터",
                fontsize=13, y=1.01, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_A2():
    m_grid = np.linspace(60, 400, 35)

    rows = []
    for capa in CAPA_ROWS:
        for p in P_LEVELS_6:
            for m in m_grid:
                res = evaluate(BASE, p=float(p), capa=capa, r=0.0, m=float(m))
                if not res["converged"]:
                    continue
                Tw = _combined_Tw(res, p)
                f_taxi = 1.0 + BASE.v * Tw / BASE.l
                rows.append(dict(capa=capa, p=p, m=m, f_taxi=f_taxi))
    df = pd.DataFrame(rows)
    df = _trim_precliff_crossover(df, ["capa", "p"], "m", "f_taxi", ceiling=4.0)

    m_ref = np.linspace(20, 400, 100)
    f_public = 1.0 + 4 * np.sqrt(5) / (BASE.kappa * np.sqrt(m_ref))
    m_auto = BASE.lam * BASE.R * BASE.l / BASE.v

    fig, axes = plt.subplots(1, len(CAPA_ROWS), figsize=(6.2 * len(CAPA_ROWS), 5.2))
    for ax, capa in zip(axes, CAPA_ROWS):
        for p, lab, color in zip(P_LEVELS_6, P_LABELS_6, GRAYS_6):
            sub = df[(df.capa == capa) & (df.p == p)].sort_values("m")
            ax.plot(sub.m, sub.f_taxi, color=color, lw=1.9, label=f"e비율={lab}")
        ax.plot(m_ref, f_public, color=CAT["blue"], lw=2.0)
        ax.text(360, np.interp(360, m_ref, f_public), "대중교통",
                color=CAT["blue"], fontsize=9, va="bottom", ha="right")
        ax.plot([m_auto], [1.0], "o", color=CAT["red"], ms=7, zorder=5)
        ax.text(m_auto, 1.05, "자가용", color=CAT["red"], fontsize=9, ha="center")
        ax.set_title(f"정원={capa}{'  (T-ITS 원형)' if capa==1 else ''}",
                    fontsize=11.5, color=INK)
        ax.set_xlabel("차량 대수")
        ax.set_ylabel("사용자 통행시간")
        ax.set_xlim(0, 400)
        ax.set_ylim(1.0, 4.0)
        _style_axes(ax)
    axes[-1].legend(loc="upper right", fontsize=8.5, frameon=False, title="e-hailing 비율")
    fig.suptitle("A2 · [T-ITS Fig 7 구조] 사용자 통행시간 vs 차량 대수 (AT비율=0) — 대전 파라미터, 식 (19)",
                fontsize=13, y=1.02, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_A4():
    r_panels = [0.0, 0.5]
    m_grid = np.linspace(60, 400, 35)

    rows = []
    for capa in CAPA_ROWS:
        for r in r_panels:
            for p in P_LEVELS_6:
                for m in m_grid:
                    res = evaluate(BASE, p=float(p), capa=capa, r=r, m=float(m))
                    if not res["converged"]:
                        continue
                    Tw = _combined_Tw(res, p)
                    f_taxi = 1.0 + BASE.v * Tw / BASE.l
                    rows.append(dict(capa=capa, r=r, p=p, m=m, f_taxi=f_taxi))
    df = pd.DataFrame(rows)
    df = _trim_precliff_crossover(df, ["capa", "r", "p"], "m", "f_taxi", ceiling=4.0)

    fig, axes = plt.subplots(len(CAPA_ROWS), len(r_panels),
                             figsize=(6.2 * len(r_panels), 4.6 * len(CAPA_ROWS)), sharex=True)
    for i, capa in enumerate(CAPA_ROWS):
        for j, r in enumerate(r_panels):
            ax = axes[i, j]
            for p, lab, color in zip(P_LEVELS_6, P_LABELS_6, GRAYS_6):
                sub = df[(df.capa == capa) & (df.r == r) & (df.p == p)].sort_values("m")
                ax.plot(sub.m, sub.f_taxi, color=color, lw=1.9, label=f"e비율={lab}")
            ax.set_title(f"정원={capa}{'  (T-ITS 원형)' if capa==1 else ''},  AT비율={r:g}",
                        fontsize=11, color=INK)
            if i == len(CAPA_ROWS) - 1:
                ax.set_xlabel("차량 대수")
            if j == 0:
                ax.set_ylabel("사용자 통행시간")
            ax.set_xlim(0, 400)
            ax.set_ylim(1.0, 4.0)
            _style_axes(ax)
    axes[0, -1].legend(loc="upper right", fontsize=8.5, frameon=False, title="e-hailing 비율")
    fig.suptitle("A4 · [T-ITS Fig 10 구조] 사용자 통행시간 vs 차량 대수, AT비율별 비교 — 대전 파라미터",
                fontsize=13, y=1.01, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_A5():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from drtmodel.optimize import find_m_c

    p_grid = np.linspace(0.05, 0.95, 10)
    r_grid = np.linspace(0.0, 0.9, 10)
    P, Rm = np.meshgrid(p_grid, r_grid)

    rows = []
    Mc_by_capa, Zc_by_capa = {}, {}
    for capa in CAPA_ROWS:
        Mc = np.zeros_like(P)
        Zc = np.zeros_like(P)
        for a in range(P.shape[0]):
            for b in range(P.shape[1]):
                p, r = float(P[a, b]), float(Rm[a, b])
                base_i = replace_base(p=p, r=r, capa=capa)
                mc = find_m_c(base_i, sigmaS_target=0.5, rtol=5e-3,
                              m_hi=max(4.0 * base_i.lam * base_i.R * base_i.l / base_i.v, 900.0))
                res = evaluate(base_i, m=mc)
                Mc[a, b] = mc
                Zc[a, b] = res["z_beta"]
                rows.append(dict(capa=capa, p=p, r=r, m_c=mc, z_beta=res["z_beta"]))
        Mc_by_capa[capa], Zc_by_capa[capa] = Mc, Zc
    df = pd.DataFrame(rows)

    mc_lo, mc_hi = df.m_c.min(), df.m_c.max()
    z_lo, z_hi = df.z_beta.min(), df.z_beta.max()

    fig = plt.figure(figsize=(13.5, 6.0 * len(CAPA_ROWS)))
    for ridx, capa in enumerate(CAPA_ROWS):
        for cidx, (Z, zlab, title, vlo, vhi) in enumerate((
                (Mc_by_capa[capa], "차량 대수", "(a) 최소 차량 대수", mc_lo, mc_hi),
                (Zc_by_capa[capa], "사회적 비용", "(b) 사회적 비용", z_lo, z_hi))):
            ax = fig.add_subplot(len(CAPA_ROWS), 2, ridx * 2 + cidx + 1, projection="3d")
            surf = ax.plot_surface(P, Rm, Z, cmap="viridis", edgecolor="k",
                                   linewidth=0.25, antialiased=True, vmin=vlo, vmax=vhi)
            ax.set_xlabel("e-hailing 비율")
            ax.set_ylabel("AT 비율")
            ax.set_zlabel(zlab)
            ax.set_zlim(vlo, vhi)
            ax.set_title(f"정원={capa}{'  (T-ITS 원형)' if capa==1 else ''} — {title}",
                        fontsize=10.5, color=INK)
            fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    fig.suptitle("A5 · [T-ITS Fig 11 구조] e-hailing 비율×AT 비율 서피스 — 대전 파라미터 (서비스 제약: street-hail 서비스율 50% 이상)",
                fontsize=13, y=1.0, color=INK)
    fig.tight_layout()
    return df, fig


def replace_base(**kw):
    from dataclasses import replace
    return replace(BASE, **kw)


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_A6():
    p_panels = [0.5, 0.75, 1.0]
    r_lines = [0.0, 0.5, 1.0]
    r_colors = [CAT["violet"], CAT["aqua"], CAT["yellow"]]
    m_grid = np.linspace(60, 400, 35)

    rows = []
    for capa in CAPA_ROWS:
        for p in p_panels:
            for r in r_lines:
                for m in m_grid:
                    res = evaluate(BASE, p=float(p), capa=capa, r=r, m=float(m))
                    if not res["converged"]:
                        continue
                    unmet = res["unmet_total"] * 2.0
                    rows.append(dict(capa=capa, p=p, r=r, m=m, unmet=unmet))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(len(CAPA_ROWS), len(p_panels),
                             figsize=(5.6 * len(p_panels), 4.4 * len(CAPA_ROWS)), sharex=True, sharey=True)
    for i, capa in enumerate(CAPA_ROWS):
        for j, p in enumerate(p_panels):
            ax = axes[i, j]
            for r, color in zip(r_lines, r_colors):
                sub = df[(df.capa == capa) & (df.p == p) & (df.r == r)].sort_values("m")
                ax.plot(sub.m, sub.unmet, "o-", color=color, ms=3, lw=1.5, label=f"AT비율={r:g}")
            ax.set_title(f"정원={capa}{'  (원형)' if capa==1 else ''},  e비율={p:g}",
                        fontsize=10.5, color=INK)
            if i == len(CAPA_ROWS) - 1:
                ax.set_xlabel("차량 대수")
            if j == 0:
                ax.set_ylabel("미충족 수요 [건]")
            _style_axes(ax)
    axes[0, -1].legend(loc="upper right", fontsize=8.5, frameon=False, title="AT 비율")
    fig.suptitle("A6 · [T-ITS Fig 12 구조] 미충족 수요 (e-hail+street-hail, 식 21) — 대전 파라미터",
                fontsize=13, y=1.02, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_B1():
    from drtmodel.optimize import find_m_c
    capa_grid = list(range(1, 11))
    combos = [(0.6, 0.2, "AT 적음 (r=0.2)"), (0.6, 0.8, "AT 많음 (r=0.8)")]
    combo_colors = [CAT["blue"], CAT["red"]]
    m_fixed = 150.0

    rows = []
    for p, r, label in combos:
        for capa in capa_grid:
            res = evaluate(BASE, p=p, r=r, capa=capa, m=m_fixed)
            base_i = replace_base(p=p, r=r, capa=capa)
            mc = find_m_c(base_i, sigmaS_target=0.5, rtol=5e-3, m_hi=900.0)
            rows.append(dict(capa=capa, label=label,
                             T_w_E_min=(res["T_w_E"] or 0.0) * 60,
                             T_w_S_min=(res["T_w_S_avg"] or 0.0) * 60,
                             m_c=mc, z_beta=res["z_beta"]))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.6))
    panels = [("T_w_E_min", f"① e-hail 대기시간 (차량대수={m_fixed:.0f} 고정)", "e-hail 대기시간 [분]"),
             ("T_w_S_min", f"② street-hail 체감 대기시간 (차량대수={m_fixed:.0f} 고정)", "street-hail 체감 대기시간 [분]"),
             ("m_c", "③ 최소 차량 대수 (street-hail 서비스율 50% 이상)", "최소 차량 대수 [대]"),
             ("z_beta", f"④ 사회적 비용 (차량대수={m_fixed:.0f} 고정)", "사회적 비용")]
    for ax, (ycol, title, ylab) in zip(axes, panels):
        for (p, r, label), color in zip(combos, combo_colors):
            sub = df[df.label == label].sort_values("capa")
            ax.plot(sub.capa, sub[ycol], "o-", color=color, lw=2.0, ms=6, label=label)
        ax.set_xlabel("정원")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_xticks(capa_grid)
        ax.axvline(1, color=MUTED, lw=1.0, ls=":", zorder=0)
        _style_axes(ax)
    axes[0].legend(fontsize=9, frameon=False, loc="upper right")
    fig.suptitle("B1 · 정원=1(흡수 없음) → 정원 증대(하이브리드 흡수)의 순효과 — 대전 파라미터, e비율=0.6",
                fontsize=13, y=1.03, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_B2():
    de_max_grid = np.linspace(0.25, 2.0, 15)
    wt_max_levels = [2.0, 5.0, 10.0]
    wt_colors = [CAT["blue"], CAT["aqua"], CAT["yellow"]]
    m_fixed, capa = 150.0, 3

    rows = []
    for wt in wt_max_levels:
        for de in de_max_grid:
            base_i = replace_base(capa=capa, wt_max=wt / 60.0, de_max=float(de), m=m_fixed)
            mdl = HybridDRTModel(base_i)
            res = mdl.solve()
            delta_bar = mdl.detour_inflation(res)
            z = mdl.social_cost(res, delta_bar=delta_bar)
            demand_s = (1 - base_i.p) * base_i.lam * base_i.R
            rows.append(dict(wt_max=wt, de_max=de, sigma_S=res["sigma"],
                             unmet_S=res["unmet"], unmet_S_pct=res["unmet"] / demand_s * 100,
                             delta_bar=delta_bar, z_beta=z))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.4))
    panels = [("sigma_S", "① street-hail 서비스율", "street-hail 서비스율"),
             ("unmet_S_pct", "② 미충족 비율", "미충족 비율 [%]"),
             ("delta_bar", "③ 차내시간 팽창 계수 (식 24)", "차내시간 팽창 계수"),
             ("z_beta", "④ 사회적 비용", "사회적 비용")]
    for ax, (ycol, title, ylab) in zip(axes, panels):
        for wt, color in zip(wt_max_levels, wt_colors):
            sub = df[df.wt_max == wt].sort_values("de_max")
            ax.plot(sub.de_max, sub[ycol], color=color, lw=2.0, label=f"최대대기시간={wt:g}분")
        ax.set_xlabel("허용 우회거리 [km]")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.8, color=INK)
        _style_axes(ax)
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle(f"B2 · street-hail 운영 설계 민감도 (허용우회거리가 지배 변수라는 H5 검증) — 정원={capa}, 차량대수={m_fixed:.0f}",
                fontsize=13, y=1.04, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_B3():
    scenarios = [("저수요", 0.5), ("기준", 2.0), ("고수요", 4.0)]
    capa, m, p, r = 3, 150.0, 0.6, 0.4

    rows = []
    grids = {}
    for label, lam in scenarios:
        base_i = replace_base(capa=capa, m=m, p=p, r=r, lam=lam)
        res = HybridDRTModel(base_i).solve()
        for fleet_key, fleet_name in (("n_AT", "AT"), ("n_HT", "HT")):
            grid = np.full((capa + 1, capa + 1), np.nan)
            for (i, j), val in res[fleet_key].items():
                grid[i, j] = val
                rows.append(dict(scenario=label, lam=lam, fleet=fleet_name, i=i, j=j, n=val))
            grids[(label, fleet_name)] = grid
    df = pd.DataFrame(rows)
    vmax_all = df.n.max()

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4))
    for col, (label, lam) in enumerate(scenarios):
        for row_i, (fleet_key, fleet_name) in enumerate((("n_AT", "AT"), ("n_HT", "HT"))):
            ax = axes[row_i, col]
            grid = grids[(label, fleet_name)]
            im = ax.imshow(grid, origin="lower", cmap="Blues", aspect="equal",
                           vmin=0, vmax=vmax_all)
            for i in range(capa + 1):
                for j in range(capa + 1):
                    if not np.isnan(grid[i, j]):
                        txt_color = INK if grid[i, j] < vmax_all * 0.6 else "white"
                        ax.text(j, i, f"{grid[i,j]:.1f}", ha="center", va="center",
                               fontsize=8.5, color=txt_color)
            ax.set_xticks(range(capa + 1)); ax.set_yticks(range(capa + 1))
            ax.set_xlabel("배정 대기 인원"); ax.set_ylabel("탑승 인원")
            ax.set_title(f"{fleet_name} — {label} (수요밀도={lam:g})", fontsize=10.5, color=INK)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(colors=INK2)
    fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, label="차량 대수 (6패널 공통 스케일)")
    fig.suptitle(f"B3 · 상태분포 히트맵 — 정원={capa}, 차량대수={m:.0f}, e비율={p}, AT비율={r}",
                fontsize=13, y=1.01, color=INK)
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_B4():
    r_grid = np.linspace(0.02, 0.95, 30)
    p = 0.6
    rows = []
    for capa in CAPA_ROWS:
        for r in r_grid:
            res = evaluate(BASE, p=p, r=float(r), capa=capa, m=150.0)
            rows.append(dict(capa=capa, r=r, c=res["c"]))
    df = pd.DataFrame(rows)

    m, lam, R, v, l = 150.0, BASE.lam, BASE.R, BASE.v, BASE.l
    c_ref = [r_ / (1 - (1 - p) * lam * R * l / v / m) for r_ in r_grid]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(r_grid, r_grid, color=MUTED, lw=1.2, ls=":", label="동적가용비율=AT비율 (참조선)")
    ax.plot(r_grid, c_ref, color=INK2, lw=1.6, ls=(0, (5, 2)), label="[T-ITS 식9] 폐형식")
    for capa, color in zip(CAPA_ROWS, CAPA_COLORS):
        sub = df[df.capa == capa].sort_values("r")
        ax.plot(sub.r, sub.c, color=color, lw=2.2,
               label=f"정원={capa}{' (T-ITS 원형)' if capa==1 else ''}")
    ax.set_xlabel("AT(자율주행) 비율")
    ax.set_ylabel("동적 AT 가용 비율 (식 22)")
    ax.set_title("B4 · 동적 AT 가용 비율 vs AT 비율 — 정원=1에서 [T-ITS 식9]와 겹치는지 확인", fontsize=11.5, color=INK)
    ax.legend(fontsize=9, frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_B5():
    p_grid = np.linspace(0.05, 0.95, 25)
    de_levels = [0.5, 1.0, 2.0]
    de_colors = [CAT["blue"], CAT["aqua"], CAT["yellow"]]
    capa, m, r = 3, 150.0, 0.3

    rows = []
    for de in de_levels:
        for p in p_grid:
            base_i = replace_base(capa=capa, m=m, r=r, p=float(p), de_max=de)
            mdl = HybridDRTModel(base_i)
            res = mdl.solve()
            delta_bar = mdl.detour_inflation(res)
            ride_time = (base_i.l / base_i.v) * (1 + delta_bar) * 60
            rows.append(dict(de_max=de, p=p, delta_bar=delta_bar, ride_time_min=ride_time))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    for ax, ycol, title, ylab in (
        (axes[0], "delta_bar", "① 차내시간 팽창 계수 (식 24)", "차내시간 팽창 계수"),
        (axes[1], "ride_time_min", "② 유효 차내시간", "유효 차내시간 [분]"),
    ):
        for de, color in zip(de_levels, de_colors):
            sub = df[df.de_max == de].sort_values("p")
            ax.plot(sub.p, sub[ycol], color=color, lw=2.0, label=f"허용우회거리={de:g}km")
        ax.set_xlabel("e-hailing 비율")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, color=INK)
        _style_axes(ax)
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle(f"B5 · 합승 흡수의 우회 비용 — 정원={capa}, 차량대수={m:.0f}, AT비율={r}",
                fontsize=13, y=1.03, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_B6():
    capa_grid = list(range(1, 11))
    lam_mults = [1.0, 4.0, 8.0]
    lam_labels = ["1x (대전 기준)", "4x (고수요)", "8x (초고수요)"]
    lam_colors = [CAT["blue"], CAT["aqua"], CAT["red"]]
    m, p, r = 500.0, 0.6, 0.4

    rows = []
    for mult, label in zip(lam_mults, lam_labels):
        for capa in capa_grid:
            base_i = replace_base(capa=capa, m=m, p=p, r=r, lam=BASE.lam * mult)
            res = HybridDRTModel(base_i).solve()
            idle = res["n_AT"][(0, 0)] + res["n_HT"][(0, 0)]
            total = sum(res["n_AT"].values()) + sum(res["n_HT"].values())
            rows.append(dict(mult=mult, label=label, capa=capa,
                             idle_pct=idle / total * 100.0,
                             sigma_S=res["sigma"] or 0.0,
                             T_w_E_min=(res["T_w_E"] or 0.0) * 60,
                             unmet_total=res["unmet_E"] + res["unmet"]))
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(11.5, 10.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.42, wspace=0.28,
                          top=0.93, bottom=0.05, left=0.09, right=0.97)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :])

    panels = [(ax1, "idle_pct", "① 완전 유휴 차량 비율", "유휴 비율 [%]"),
             (ax2, "sigma_S", "② street-hail 서비스율", "street-hail 서비스율"),
             (ax3, "T_w_E_min", "③ e-hail 대기시간", "e-hail 대기시간 [분]"),
             (ax4, "unmet_total", "④ 미충족 수요", "미충족 수요 [건/시]")]
    for ax, ycol, title, ylab in panels:
        for label, color in zip(lam_labels, lam_colors):
            sub = df[df.label == label].sort_values("capa")
            ax.plot(sub.capa, sub[ycol], "o-", color=color, lw=2.0, ms=5, label=label)
        ax.set_xlabel("정원")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_xticks(capa_grid)
        _style_axes(ax)
    ax1.legend(fontsize=9, frameon=False, title="수요 밀도")
    fig.suptitle(f"B6 · 정원 효과는 수요-의존적 — 기준 수요에서 포화가 빠른 이유 (차량대수={m:.0f}, e비율={p}, AT비율={r})",
                fontsize=13, y=0.975, color=INK)
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_B7():
    m_panels = [100, 200]
    p_grid = np.linspace(0.02, 0.98, 40)
    lam_mult = 2.0
    COL = {"E": CAT["red"], "S": CAT["green"], "T": CAT["blue"]}
    CEIL = 70.0

    rows = []
    for capa in CAPA_ROWS:
        for m in m_panels:
            for p in p_grid:
                res = evaluate(BASE, p=float(p), capa=capa, r=0.0, m=float(m),
                               lam=BASE.lam * lam_mult)
                if not res["converged"]:
                    continue
                T_w_E = (res["T_w_E"] or 0.0) * 60
                T_w_S = (res["T_w_S"] or 0.0) * 60
                T_w = _combined_Tw(res, p) * 60
                if max(T_w_E, T_w_S, T_w) > CEIL:
                    continue
                rows.append(dict(capa=capa, m=m, p=p,
                                 T_w_E_min=T_w_E, T_w_S_min=T_w_S, T_w_min=T_w))
    df = pd.DataFrame(rows)
    y_top = df[["T_w_E_min", "T_w_S_min", "T_w_min"]].to_numpy().max() * 1.05

    fig, axes = plt.subplots(len(CAPA_ROWS), len(m_panels),
                             figsize=(9.6, 4.0 * len(CAPA_ROWS)), sharex=True, sharey=True)
    for i, capa in enumerate(CAPA_ROWS):
        for j, m in enumerate(m_panels):
            ax = axes[i, j]
            sub = df[(df.capa == capa) & (df.m == m)].sort_values("p")
            ax.plot(sub.p, sub.T_w_E_min, color=COL["E"], lw=2.0)
            ax.plot(sub.p, sub.T_w_S_min, color=COL["S"], lw=2.0)
            ax.plot(sub.p, sub.T_w_min, color=COL["T"], lw=2.0)
            ax.set_title(f"정원={capa}{'  (T-ITS 원형)' if capa==1 else ''},  차량대수={m}",
                        fontsize=11, color=INK)
            if i == len(CAPA_ROWS) - 1:
                ax.set_xlabel("e-hailing 비율")
            if j == 0:
                ax.set_ylabel("대기시간 [분]")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, y_top)
            _style_axes(ax)

    handles = [plt.Line2D([], [], color=COL["E"], lw=2.2, label="e-hail 대기시간"),
              plt.Line2D([], [], color=COL["S"], lw=2.2, label="street-hail 대기시간"),
              plt.Line2D([], [], color=COL["T"], lw=2.2, label="전체 대기시간 ([T-ITS] 결합식)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
              frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"B7 · A1을 수요 {lam_mult:g}배 조건에서 재현 — 정원 민감도가 커지는 것을 확인 (AT비율=0)",
                fontsize=13, y=1.01, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_B8():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    m_fixed = 100.0
    lam_mult = 1.3
    capa_grid = list(range(1, 11))
    p_grid_surface = np.linspace(0.02, 0.98, 20)
    p_grid_line = np.linspace(0.02, 0.98, 25)
    line_capas = [1, 2, 3, 5, 10]
    line_colors = ["#8b0000", "#e34948", "#eda100", "#1baf7a", "#2a78d6"]

    rows = []
    for capa in capa_grid:
        for p in p_grid_surface:
            res = evaluate(BASE, p=float(p), capa=capa, r=0.0, m=m_fixed, lam=BASE.lam * lam_mult)
            rows.append(dict(capa=capa, p=p, T_w_min=_combined_Tw(res, p) * 60,
                             converged=res["converged"]))
    df = pd.DataFrame(rows)

    line_rows = []
    for capa in line_capas:
        for p in p_grid_line:
            res = evaluate(BASE, p=float(p), capa=capa, r=0.0, m=m_fixed, lam=BASE.lam * lam_mult)
            line_rows.append(dict(capa=capa, p=p, T_w_min=_combined_Tw(res, p) * 60))
    line_df = pd.DataFrame(line_rows)

    fig = plt.figure(figsize=(14.5, 6.0))

    ax1 = fig.add_subplot(1, 2, 1)
    for capa, color in zip(line_capas, line_colors):
        sub = line_df[line_df.capa == capa].sort_values("p")
        ax1.plot(sub.p, sub.T_w_min, color=color, lw=2.2, label=f"정원={capa}")
    ax1.set_xlabel("e-hailing 비율")
    ax1.set_ylabel("전체 대기시간 [분]")
    ax1.set_title("① 정원별 곡선 — e-hailing 비율이 늘수록, 정원이 클수록 모두 감소", fontsize=11, color=INK)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=9, frameon=False, title="정원")
    _style_axes(ax1)

    P, Cp = np.meshgrid(p_grid_surface, capa_grid)
    Z = df.pivot(index="capa", columns="p", values="T_w_min").reindex(index=capa_grid).to_numpy()
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf = ax2.plot_surface(P, Cp, Z, cmap="viridis", edgecolor="k", linewidth=0.2, antialiased=True)
    ax2.set_xlabel("e-hailing 비율")
    ax2.set_ylabel("정원")
    ax2.set_zlabel("전체 대기시간 [분]")
    ax2.set_title("② 정원×e-hailing 비율 서피스 — 두 방향 모두 증가할수록 대기시간 하강", fontsize=11, color=INK)
    fig.colorbar(surf, ax=ax2, shrink=0.6, pad=0.12)

    fig.suptitle(f"B8 · 전체 대기시간의 동시 감소 — 차량대수={m_fixed:.0f}, 수요={lam_mult:g}배, AT비율=0",
                fontsize=13, y=1.02, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
RHO_AT_M150 = BASE.lam * BASE.R * BASE.l / (BASE.v * 150.0)
RHO_DAEJEON = RHO_AT_M150

M_V2 = 100.0

D2_AZIM = -144

D2_CMAP = _mcolors.LinearSegmentedColormap.from_list(
    "D2_occ", ["#4a000e", "#7a0016", "#a80c1e", "#cd3122", "#e2695f",
               "#f2a58f", "#fad3c4", "#fff7f4"])
D2_GAMMA = 2.2
D2_LEVELS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
D2_LINE = "#c9184a"


def _lam_for(rho, m=M_V2):
    """_lam_for (see the paper for the definitions)."""
    return rho * BASE.v * m / (BASE.R * BASE.l)


def _collapsed(res):
    """_collapsed (see the paper for the definitions)."""
    return (not res["converged"]) or res["n_E"] < 1.0 or (res["sigma_E"] or 0.0) < 0.9


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_D1():
    """fig_D1 (see the paper for the definitions)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    m_fixed = 40.0
    capa_colors = {2: CAT["aqua"], 3: CAT["blue"]}
    forms = [(T("합승택시", "shared-taxi"), "b", False, (0, (4, 2)), (2,),
              lambda capa: CAT["red"]),
             (T("DRT", "DRT"), "a", True, "-", (2, 3),
              lambda capa: capa_colors[capa])]
    ref_label = T("참조", "reference")

    rho_grid = np.linspace(0.15, 1.42, 191)
    rows = []
    for rho in rho_grid:
        lam = _lam_for(rho, m_fixed)
        res = evaluate(BASE, p=0.999, r=0.0, capa=1, m=m_fixed, lam=lam)
        bad = _collapsed(res)
        rows.append(dict(panel="wait", rho=rho, form=ref_label, capa=1,
                         val=np.nan if bad else res["T_w_E"] * 60))
        for fname, rule, flex, _, capas, _ in forms:
            for capa in capas:
                res = evaluate(BASE, p=0.999, r=0.0, capa=capa, m=m_fixed, lam=lam,
                               availability_rule=rule, flexible_dropoff=flex)
                bad = _collapsed(res)
                rows.append(dict(panel="wait", rho=rho, form=fname, capa=capa,
                                 val=np.nan if bad else res["T_w_E"] * 60))

    def rho_limit(capa, rule, flex):
        lo, hi = 0.2, 1.5
        def ok(rho):
            lam = _lam_for(rho, m_fixed)
            res = evaluate(BASE, p=0.999, r=0.0, capa=capa, m=m_fixed, lam=lam,
                           availability_rule=rule, flexible_dropoff=flex)
            return not _collapsed(res)
        if not ok(lo):
            return np.nan
        while hi - lo > 0.01:
            mid = 0.5 * (lo + hi)
            if ok(mid):
                lo = mid
            else:
                hi = mid
        return lo
    limit_configs = [(T("택시, 정원=1", "taxi, capa = 1"), 1, "a", True, MUTED),
                      (T("합승택시, 정원=2", "shared-taxi, capa = 2"), 2, "b", False, CAT["red"]),
                      (T("DRT, 정원=2", "DRT, capa = 2"), 2, "a", True, CAT["aqua"]),
                      (T("DRT, 정원=3", "DRT, capa = 3"), 3, "a", True, CAT["blue"])]
    limit_vals = {}
    for label, capa, rule, flex, _ in limit_configs:
        lim = rho_limit(capa, rule, flex)
        limit_vals[(capa, rule)] = lim
        rows.append(dict(panel="limit", rho=lim, form=label, capa=capa, val=lim))

    from drtmodel.optimize import find_m_c
    mc_cfgs = [(T("택시 (정원=1)", "taxi (capa = 1)"), 1, "a", True, "#6b6a64", ":"),
               (T("합승택시 (정원=2)", "shared-taxi (capa = 2)"),
                2, "b", False, CAT["red"], (0, (4, 2))),
               (T("DRT (정원=3)", "DRT (capa = 3)"), 3, "a", True, CAT["blue"], "-")]
    dem_grid = np.concatenate([np.array([5.0, 15.0, 30.0]), np.linspace(60.0, 510.0, 16)])
    for fname, capa_s, rule, flex, _, _ in mc_cfgs:
        for dem in dem_grid:
            base_i = replace_base(p=0.999, r=0.0, capa=capa_s, lam=dem / BASE.R,
                                  availability_rule=rule, flexible_dropoff=flex)
            try:
                mc = find_m_c(base_i, TwE_target=5.0 / 60.0, sigmaE_target=0.9,
                              rtol=5e-3, m_hi=900.0)
            except RuntimeError:
                mc = np.nan
            rows.append(dict(panel="mc", demand=dem, form=fname, capa=capa_s, val=mc))
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(8.6, 3.7))
    gs = fig.add_gridspec(1, 2, wspace=0.26, top=0.86, bottom=0.14, left=0.08, right=0.97)
    ax = fig.add_subplot(gs[0, 0])

    def _plot_with_collapse_marker(sub, color, lw, ls, label, lim, lim_dy):
        ax.plot(sub.rho, sub.val, color=color, lw=lw, ls=ls, label=label)
        valid = sub.dropna(subset=["val"])
        if not valid.empty and valid.rho.max() < rho_grid.max() - 1e-6:
            x_end, y_end = valid.rho.iloc[-1], valid.val.iloc[-1]
            ax.plot(x_end, y_end, marker="o", color=color, ms=5, zorder=5, lw=0)
            ax.text(x_end, y_end + lim_dy, f"η={lim:.2f}", fontsize=8.3,
                    color=color, ha="center", zorder=6,
                    va="bottom" if lim_dy > 0 else "top",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="none", alpha=0.85))

    sub = df[(df.panel == "wait") & (df.form == ref_label)].sort_values("rho")
    _plot_with_collapse_marker(sub, MUTED, 1.6, ":", T("택시, 정원=1", "taxi, capa = 1"),
                               limit_vals[(1, "a")], -0.18)
    for fname, rule, flex, ls, capas, color_fn in forms:
        for capa in capas:
            sub = df[(df.panel == "wait") & (df.form == fname) & (df.capa == capa)].sort_values("rho")
            lim_dy = -0.18 if (capa, rule) == (2, "b") else -0.18 if capa == 2 else 0.18
            _plot_with_collapse_marker(sub, color_fn(capa), 2.1, ls,
                                       T(f"{fname}, 정원={capa}", f"{fname}, capa = {capa}"),
                                       limit_vals[(capa, rule)], lim_dy)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.08)
    ax.axvline(1.0, color=INK2, lw=1.2, ls="--", zorder=1)
    ax.text(1.0, 0.05, T("1인 승차 한계", "single-occupancy limit"),
            fontsize=8.3, color=INK2, ha="center", va="bottom",
            transform=ax.get_xaxis_transform())
    ax.set_xlim(0, rho_grid.max())
    ax.set_xlabel(T("부하율 (η)", "load factor (η)"))
    ax.set_ylabel(T("e-hail 대기시간 [분]", "e-hail waiting time [min]"))
    _panel_tag(ax, "a", corner="upper right")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    _style_axes(ax)
    ax = fig.add_subplot(gs[0, 1])
    mc_df = df[df.panel == "mc"]
    dem_ref = np.linspace(0, dem_grid.max(), 50)
    ax.plot(dem_ref, dem_ref * BASE.l / BASE.v, color=CAT["yellow"], lw=2.6,
            ls=(0, (2, 2)), zorder=1,
            label=T("1인 승차 작업량 (수요×평균 차내시간)",
                    "single-occupancy workload (demand × mean ride time)"))
    for fname, capa_s, rule, flex, color, ls in mc_cfgs:
        sub = mc_df[mc_df.form == fname].sort_values("demand")
        ax.plot(sub.demand, sub.val, color=color, lw=2.2, ls=ls, label=fname)
    d_mark = 400.0
    s_taxi = mc_df[mc_df.form == mc_cfgs[0][0]].sort_values("demand")
    s_drt = mc_df[mc_df.form == mc_cfgs[2][0]].sort_values("demand")
    v_taxi = np.interp(d_mark, s_taxi.demand, s_taxi.val)
    v_drt = np.interp(d_mark, s_drt.demand, s_drt.val)
    ax.annotate("", xy=(d_mark, v_drt), xytext=(d_mark, v_taxi),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=2.2))
    ax.text(d_mark - 12, (v_taxi + v_drt) / 2, f"-{(1 - v_drt / v_taxi) * 100:.0f}%",
            fontsize=12, color=INK, ha="right", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_xlim(0, dem_grid.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel(T("수요 [건/시]", "demand [trips/h]"))
    ax.set_ylabel(T("최소 차량 대수 [대]", "minimum fleet size [veh]"))
    _panel_tag(ax, "b", corner="upper right")
    ax.legend(fontsize=8.0, frameon=False, loc="upper left")
    _style_axes(ax)
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_E1():
    #
    rho_grid = np.linspace(0.15, 0.95, 17)
    p_grid = np.linspace(0.05, 0.95, 19)
    capas = [1, 2, 3, 5, 10]
    capa_colors = [CAT["red"], CAT["orange"], CAT["aqua"], CAT["blue"], CAT["violet"]]
    capas_bot = [1, 3, 10]
    bot_colors = [CAT["red"], CAT["aqua"], CAT["violet"]]
    r_bot = [(0.0, (0, (5, 2))), (0.4, "-")]
    p0, r0, m = 0.6, 0.4, M_V2
    rho0 = 0.85

    def _metrics(capa, p, r, lam):
        mdl = HybridDRTModel(replace_base(capa=capa, m=m, p=p, r=r, lam=lam))
        res = mdl.solve()
        if _collapsed(res):
            return np.nan, np.nan, np.nan
        idle_pct = (res["n_AT"][(0, 0)] + res["n_HT"][(0, 0)]) / m * 100.0
        mid = sum(mdl.w[n] * res["n_HT"][n] for n in mdl.w if n != (0, 0))
        phi_mid = mid / res["n_eff"] * 100.0 if res["n_eff"] > 0 else 0.0
        return idle_pct, (res["sigma"] or 0.0), phi_mid

    rows = []
    for rho in rho_grid:
        lam = _lam_for(rho, m)
        for capa in capas:
            idle, sig, phi = _metrics(capa, p0, r0, lam)
            rows.append(dict(panel="rho", x=rho, capa=capa, r=r0,
                             idle_pct=idle, sigma_S=sig, phi_mid=phi))
    lam0 = _lam_for(rho0, m)
    for pv in p_grid:
        for capa in capas_bot:
            for rv, _ in r_bot:
                idle, sig, phi = _metrics(capa, float(pv), rv, lam0)
                rows.append(dict(panel="p", x=pv, capa=capa, r=rv,
                                 idle_pct=idle, sigma_S=sig, phi_mid=phi))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
    metrics = [("idle_pct", T("완전 유휴 차량 비율", "fully idle vehicle share"),
                T("유휴 비율 [%]", "idle share [%]")),
               ("sigma_S", T("street-hail 서비스율", "street-hail service probability σ_S"),
                T("street-hail 서비스율", "street-hail service probability σ_S")),
               ("phi_mid", T("운행중 흡수 채널의 기여 비중 (식 24)",
                             "in-service absorption channel share (Eq. 24)"),
                T("운행중 채널 비중 [%]", "in-service channel share [%]"))]
    marks = ["①②③", "④⑤⑥"]
    for ci, (ycol, title, ylab) in enumerate(metrics):
        ax = axes[0, ci]
        for capa, color in zip(capas, capa_colors):
            sub = df[(df.panel == "rho") & (df.capa == capa)].sort_values("x")
            ax.plot(sub.x, sub[ycol], "o-", color=color, lw=1.9, ms=3.4,
                    label=T(f"정원={capa}", f"capa = {capa}"))
        ax.set_xlabel(T("부하율 (수요 주행부하 ÷ 차량 공급)",
                        "load factor η (demand workload ÷ vehicle supply)"))
        ax.set_ylabel(ylab)
        ax.set_title(T(f"{marks[0][ci]} {title} — e비율={p0}·AT비율={r0} 고정",
                       f"{marks[0][ci]} {title} — p={p0}, r={r0} fixed"),
                     fontsize=10.2, color=INK)
        _style_axes(ax)
    for ci, (ycol, title, ylab) in enumerate(metrics):
        ax = axes[1, ci]
        for capa, color in zip(capas_bot, bot_colors):
            for rv, ls in r_bot:
                sub = df[(df.panel == "p") & (df.capa == capa)
                         & (df.r == rv)].sort_values("x")
                ax.plot(sub.x, sub[ycol], color=color, lw=1.9, ls=ls,
                        label=T(f"정원={capa}, AT비율={rv}", f"c={capa}, r={rv}"))
        ax.set_xlabel(T("e비율 (e-hail 수요 ÷ 전체 수요)",
                        "e-hail share p (e-hail demand ÷ total demand)"))
        ax.set_ylabel(ylab)
        ax.set_title(T(f"{marks[1][ci]} {title} — 부하율={rho0} 고정",
                       f"{marks[1][ci]} {title} — ρ={rho0} fixed"),
                     fontsize=10.2, color=INK)
        _style_axes(ax)
    axes[0, 0].legend(fontsize=8.5, frameon=False, title=T("정원", "capacity c"), ncol=2)
    axes[1, 1].legend(fontsize=7.6, frameon=False, ncol=2,
                      title=T("색=정원, 파선=AT비율 0 · 실선=AT비율 0.4",
                              "color = capacity; dashed r=0 · solid r=0.4"))
    fig.suptitle(T("E1 · [Task 2] 흡수가 활성화되는 체제 — 부하율(윗줄)과 e비율(아랫줄) 두 축에서 본 정원 효과 "
                   f"(차량대수={m:.0f})",
                   "E1 · [Task 2] Regimes that activate absorption — capacity effect along load factor (top) "
                   f"and e-hail share (bottom) (m={m:.0f})"),
                fontsize=12.8, y=1.0, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_D2():
    from drtmodel.optimize import find_m_c
    capas = [1, 2, 3, 5, 10]
    capa_colors = [CAT["red"], CAT["orange"], CAT["aqua"], CAT["blue"], CAT["violet"]]
    rho_grid = np.linspace(0.15, 0.95, 17)
    m_fixed = M_V2

    rows = []
    for rho in rho_grid:
        lam = _lam_for(rho, m_fixed)
        for capa in capas:
            base_i = replace_base(p=0.999, r=0.0, capa=capa, lam=lam)
            try:
                mc = find_m_c(base_i, TwE_target=5.0 / 60.0, sigmaE_target=0.9,
                              rtol=5e-3, m_hi=900.0)
            except RuntimeError:
                mc = np.nan
            rows.append(dict(panel="mc", rho=rho, capa=capa, m_c=mc))

    surf_capas = list(range(1, 11))
    surf_rhos = np.linspace(0.02, 0.98, 16)
    Zmc = np.full((len(surf_rhos), len(surf_capas)), np.nan)
    Cocc = np.full((len(surf_rhos), len(surf_capas)), np.nan)
    for a, rho_s in enumerate(surf_rhos):
        lam = _lam_for(rho_s, m_fixed)
        for b, capa in enumerate(surf_capas):
            base_i = replace_base(p=0.999, r=0.0, capa=capa, lam=lam)
            try:
                mc_here = find_m_c(base_i, TwE_target=5.0 / 60.0, sigmaE_target=0.9,
                                   rtol=5e-3, m_hi=900.0)
            except RuntimeError:
                continue
            res = HybridDRTModel(replace_base(p=0.999, r=0.0, capa=capa, lam=lam,
                                              m=mc_here)).solve()
            occupied = sum(res["n_AT"][(i, j)] + res["n_HT"][(i, j)]
                           for (i, j) in res["n_AT"] if i >= 1)
            Zmc[a, b] = mc_here
            Cocc[a, b] = res["onboard"] / occupied if occupied > 1e-9 else np.nan
            rows.append(dict(panel="surf", rho=rho_s, capa=capa, m_c=mc_here,
                             mean_occ=Cocc[a, b]))
    df = pd.DataFrame(rows)
    mc_df = df[df.panel == "mc"]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    Cg, Rg = np.meshgrid(surf_capas, surf_rhos)
    occ_hi = float(np.nanmax(Cocc))
    warp = _mcolors.PowerNorm(gamma=D2_GAMMA, vmin=1.0, vmax=occ_hi)
    band_reps = [1.1, 1.3, 1.5, 1.7, 1.9, (2.0 + occ_hi) / 2.0]
    band_colors = [D2_CMAP(float(warp(v))) for v in band_reps]
    cs = ax.contourf(Cg, Zmc, np.clip(Cocc, 1.0, None), levels=D2_LEVELS,
                     colors=band_colors, extend="max", zorder=1)
    ax.contour(Cg, Zmc, np.clip(Cocc, 1.0, None), levels=D2_LEVELS[1:],
               colors="white", linewidths=0.7, alpha=0.8, zorder=2)
    cb = fig.colorbar(cs, ax=ax, pad=0.02, spacing="uniform",
                      label=T("평균 재차인원 [인/차량]",
                              "average occupancy [pax/veh]"))
    cb.set_ticks(D2_LEVELS)
    cb.ax.set_yticklabels([f"{t:.1f}" for t in D2_LEVELS[:-1]] + [T("2.0 이상", "2.0+")])

    xs = np.array(surf_capas, dtype=float)
    for idx, ls, lab in ((0, "-", T("부하율 0 (하한)", "load factor 0 (lower bound)")),
                         (-1, (0, (2, 1.3)), T("부하율 1 (상한)", "load factor 1 (upper bound)"))):
        ax.plot(xs, Zmc[idx], color="white", lw=4.6, solid_capstyle="round", zorder=3)
        ax.plot(xs, Zmc[idx], color=D2_LINE, lw=2.4, ls=ls, solid_capstyle="round",
                zorder=4, label=lab)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.92, edgecolor="none",
              loc="upper right")
    ax.set_xticks(surf_capas)
    ax.set_xlim(1, 10)
    ax.set_ylim(0, np.nanmax(Zmc) * 1.08)
    ax.set_xlabel(T("정원 (capa)", "capacity ($capa$)"))
    ax.set_ylabel(T("최소 차량 대수 [대]", "minimum fleet size [veh]"))
    _style_axes(ax)
    ax.grid(False)
    fig.tight_layout()

    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_D3():
    capa, r = 1, 0.0
    DIVERGE = 1e4
    m_floor = BASE.lam * BASE.R * BASE.l / BASE.v

    p_map = np.linspace(0.05, 0.95, 10)
    m_map = np.linspace(56.0, 112.0, 14)
    demand = BASE.lam * BASE.R
    U = np.full((len(m_map), len(p_map)), np.nan)
    rows = []
    for b, p in enumerate(p_map):
        for a, m in enumerate(m_map):
            res = evaluate(BASE, p=float(p), r=r, capa=capa, m=float(m), balking=True)
            ok = res["converged"] and res["min_state"] > -1e-8
            if ok:
                U[a, b] = (res["unmet_E"] + res["unmet"]) / demand * 100
            rows.append(dict(panel="map", p=p, m=m, unmet_pct=U[a, b]))

    def delay_diverged(p, m):
        res = evaluate(BASE, p=float(p), r=r, capa=capa, m=float(m), balking=False)
        ok = res["converged"] and res["min_state"] > -1e-8
        return (not ok) or (res["T_w_S"] or 0.0) * 60 > DIVERGE

    m_cliff = []
    for p in p_map:
        lo, hi = m_floor + 1.0, 170.0
        if delay_diverged(p, hi):
            m_cliff.append(np.nan)
            continue
        while hi - lo > 1.0:
            mid = 0.5 * (lo + hi)
            if delay_diverged(p, mid):
                lo = mid
            else:
                hi = mid
        m_cliff.append(hi)
        rows.append(dict(panel="cliff", p=p, m=hi, unmet_pct=np.nan))

    m_grid = np.linspace(25.0, 160.0, 271)
    p_cuts = [(0.2, CAT["violet"]), (0.6, CAT["aqua"]), (0.9, CAT["orange"])]
    for p, _ in p_cuts:
        for m in m_grid:
            for balk in (True, False):
                res = evaluate(BASE, p=p, r=r, capa=capa, m=float(m), balking=balk)
                ok = res["converged"] and res["min_state"] > -1e-8
                if balk:
                    Tw = (res["T_w_S_avg"] or 0.0) * 60 if ok else np.nan
                    diverged = not ok
                else:
                    Tw = (res["T_w_S"] or 0.0) * 60 if ok else np.nan
                    diverged = (not ok) or Tw > DIVERGE
                rows.append(dict(panel="cut", p=p, m=m, balking=balk,
                                 T_w_min=np.nan if diverged else Tw, unmet_pct=np.nan))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    from matplotlib.colors import ListedColormap, BoundaryNorm
    levels = [0.0, 0.5, 2, 5, 10, 20, 35, 50, 70, 100]
    band_colors = ["#ffffff", "#c3d9f0", "#8fbce4", "#5d9bd5", "#3579c2",
                   "#1f5aa5", "#123f83", "#0a2a60", "#04143a"]
    nb_cmap = ListedColormap(band_colors)
    nb_norm = BoundaryNorm(levels, nb_cmap.N)
    ax = axes[0]
    Pg, Mg = np.meshgrid(p_map, m_map)
    cf = ax.contourf(Pg, Mg, U, levels=levels, cmap=nb_cmap, norm=nb_norm, extend="max")
    cb = fig.colorbar(cf, ax=ax)
    _colorbar_title(cb, T("발킹 있음:\n미충족 수요 [%]", "with balking:\nunmet demand [%]"),
                    fontsize=7, ticklabelsize=7)
    ax.plot(p_map, m_cliff, color=CAT["red"], lw=2.4,
            label=T("발킹 없음", "without balking"))
    ax.set_xlabel(T("e-hailing 비율 (p)", "e-hail proportion ($p$)"), fontsize=9)
    ax.set_ylabel(T("차량 대수 (m)", "fleet size ($m$)"), fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    _panel_tag(ax, "a", linespacing=1.3, labelpad=3)
    _style_axes(ax)
    ax.tick_params(labelsize=8.5)
    ax = axes[1]
    ax.set_yscale("log")
    y_top = 70.0
    FLAT_CUT = 59.0
    with_handles, without_handles = [], []
    with_labels, without_labels = [], []
    for p, color in p_cuts:
        sub = df[(df.panel == "cut") & (df.p == p) & (df.balking == True)].sort_values("m")  # noqa: E712
        below = sub.T_w_min < FLAT_CUT
        if below.any():
            first_pos = below.to_numpy().argmax()
            keep_from = max(first_pos - 1, 0)
            sub_trunc = sub.iloc[keep_from:]
        else:
            sub_trunc = sub
        line, = ax.plot(sub_trunc.m, sub_trunc.T_w_min, color=color, lw=2.1,
                        label=T(f"발킹 있음, p={p}", f"with balking, $p$={p}"))
        with_handles.append(line)
        with_labels.append(T(f"발킹 있음, p={p}", f"with balking, $p$={p}"))
        sub = df[(df.panel == "cut") & (df.p == p) & (df.balking == False)].sort_values("m")  # noqa: E712
        line, = ax.plot(sub.m, sub.T_w_min, color=color, lw=2.1, ls=(0, (4, 2)),
                        label=T(f"발킹 없음, p={p}", f"without balking, $p$={p}"))
        without_handles.append(line)
        without_labels.append(T(f"발킹 없음, p={p}", f"without balking, $p$={p}"))
    ax.set_ylim(top=y_top)
    ax.set_yticks([5, 10, 20, 40, 60])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(T("차량 대수 (m)", "fleet size ($m$)"), fontsize=9)
    ax.set_ylabel(T("전체 대기시간 [분]", "total waiting time [min]"), fontsize=9)
    _panel_tag(ax, "b", linespacing=1.3, labelpad=3)
    _style_axes(ax)
    ax.tick_params(labelsize=8.5)
    handles, labels = [], []
    for wh, wl, oh, ol in zip(with_handles, with_labels, without_handles, without_labels):
        handles += [wh, oh]
        labels += [wl, ol]
    fig.tight_layout(rect=(0, 0.13, 1, 0.97))
    b_cx = (axes[1].get_position().x0 + axes[1].get_position().x1) / 2.0
    fig.legend(handles, labels, ncol=3, frameon=False, fontsize=7,
               loc="upper center", bbox_to_anchor=(b_cx, 0.11))
    _recenter_colorbar_title(fig, cb)
    return df, fig


# ══════════════════════════════════════════════════════════════════════
#      analysis_plan.md §6.5.
# ══════════════════════════════════════════════════════════════════════
def fig_E2():
    capa_grid = [1, 2, 3, 5, 10]
    rho_levels = [(0.4, T("0.40 (저부하)", "0.40 (low)")),
                  (0.6, T("0.60 (중부하)", "0.60 (medium)")),
                  (0.85, T("0.85 (고부하)", "0.85 (high)"))]
    rho_colors = [CAT["blue"], CAT["aqua"], CAT["red"]]
    pr_cases = [(0.3, 0.1), (0.3, 0.7), (0.9, 0.1), (0.9, 0.7)]
    m = M_V2

    rows = []
    for p, r in pr_cases:
        for rho, rlabel in rho_levels:
            lam = _lam_for(rho, m)
            for capa in capa_grid:
                mdl = HybridDRTModel(replace_base(capa=capa, m=m, p=p, r=r, lam=lam))
                res = mdl.solve()
                bad = _collapsed(res)
                mid_ht = sum(res["n_HT"][(i, j)] for (i, j) in res["n_HT"]
                             if 0 < i + j < capa)
                demand_s = (1 - p) * lam * BASE.R
                rows.append(dict(p=p, r=r, rho=rho, rlabel=rlabel, capa=capa,
                                 mid_ht=np.nan if bad else mid_ht,
                                 n_eff=np.nan if bad else res["n_eff"],
                                 sigma_S=np.nan if bad else (res["sigma"] or 0.0),
                                 unmet_S_pct=np.nan if bad else
                                 res["unmet"] / demand_s * 100))
    df = pd.DataFrame(rows)

    stages = [("mid_ht", T("① 흡수의 원료 — 운행 중이며 빈자리가 있는 HT 차량\n"
                           "(정원=1에서는 정의상 0 — 태우면 만석)",
                           "① Raw material of absorption — in-service HT with spare seats\n"
                           "(identically 0 at c=1 — one pickup fills the vehicle)"),
               T("빈자리 있는 운행중 HT [대]", "in-service HT with spare seats [veh]")),
              ("n_eff", T("② 유효 가용 대수 (식 13)", "② effective available vehicles (Eq. 13)"),
               T("유효 가용 대수 [대]", "effective available vehicles [veh]")),
              ("sigma_S", T("③ street-hail 서비스율 (식 15)", "③ street-hail service probability σ_S (Eq. 15)"),
               T("street-hail 서비스율", "street-hail service probability σ_S")),
              ("unmet_S_pct", T("④ street-hail 미충족 비율 (식 21)", "④ street-hail unmet share (Eq. 21)"),
               T("미충족 비율 [%]", "unmet share [%]"))]
    fig, axes = plt.subplots(4, 4, figsize=(18.0, 13.6), sharex=True)
    for ri, (p, r) in enumerate(pr_cases):
        for ci, (ycol, title, ylab) in enumerate(stages):
            ax = axes[ri, ci]
            for (rho, rlabel), color in zip(rho_levels, rho_colors):
                sub = df[(df.p == p) & (df.r == r) & (df.rho == rho)].sort_values("capa")
                ax.plot(sub.capa, sub[ycol], "o-", color=color, lw=1.9, ms=4,
                        label=T(f"부하율 {rlabel}", f"ρ = {rlabel}"))
            ax.set_xticks(capa_grid)
            if ci == 0:
                ax.set_ylabel(T(f"e비율={p} · AT비율={r}\n{ylab}",
                                f"p={p}, r={r}\n{ylab}"), fontsize=9.2)
            else:
                ax.set_ylabel(ylab, fontsize=9.2)
            if ri == 0:
                ax.set_title(title, fontsize=10.2, color=INK)
            if ri == len(pr_cases) - 1:
                ax.set_xlabel(T("정원", "capacity c"))
            _style_axes(ax)
    axes[0, 0].legend(fontsize=8.2, frameon=False, title=T("부하율", "load factor ρ"))
    fig.suptitle(T("E2 · [Task 2] 흡수의 인과 사슬 — e비율×AT비율 2×2 요인설계 "
                   f"(행=조합, 열=식 13→14→15→21 사슬, 차량대수={m:.0f})",
                   "E2 · [Task 2] Causal chain of absorption — 2×2 factorial in p×r "
                   f"(rows = cases, columns = Eq. 13→14→15→21 chain, m={m:.0f})"),
                fontsize=12.8, y=0.998, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_E3():
    # ────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────
    rho_grid = np.linspace(0.15, 0.95, 17)
    capas = [1, 2, 3, 5, 10]
    capa_colors = [CAT["red"], CAT["orange"], CAT["aqua"], CAT["blue"], CAT["violet"]]
    p_hyb, r, m = 0.6, 0.4, 40.0

    rows = []
    for rho in rho_grid:
        lamE = _lam_for(rho, m)
        lam_tot = lamE / p_hyb
        for capa in capas:
            before = evaluate(BASE, capa=capa, m=m, p=0.999, r=r, lam=lamE)
            after = evaluate(BASE, capa=capa, m=m, p=p_hyb, r=r, lam=lam_tot)
            bad = _collapsed(before) or _collapsed(after)
            rows.append(dict(
                rho=rho, capa=capa, bad=bad,
                dTwE_pct=(after["T_w_E"] - before["T_w_E"]) / before["T_w_E"] * 100,
                detour_pct=after["delta_bar"] * 100,
                sigma_S=(after["sigma"] or 0.0),
                WS_hyb=(after["T_w_S_avg"] or 0.0) * 60,
                dz_pct=(after["z_beta"] - before["z_beta"]) / before["z_beta"] * 100))
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(8.6, 8.2))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.20,
                          top=0.91, bottom=0.12, left=0.10, right=0.97)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    panels = [
        (ax_a, "a", "dTwE_pct",
         T("대기시간 증가율 [%]", "waiting time increase [%]"), True, "upper right"),
        (ax_b, "b", "detour_pct",
         T("차내시간 증가율 [%]", "in-vehicle time increase [%]"), True, "upper left"),
        (ax_c, "c", "sigma_S",
         T("서비스율 (σₛ)", "service probability ($\\sigma_S$)"), False, "lower left"),
        (ax_d, "d", "WS_hyb",
         T("대기시간 [분]", "waiting time [min]"), False, "upper right"),
        (ax_e, "e", "dz_pct",
         T("사회적 비용 증가율 [%]", "social cost increase [%]"), True, "upper left"),
    ]
    for ax, letter, ycol, ylab, zeroline, corner in panels:
        good_all = df[~df.bad]
        for capa, color in zip(capas, capa_colors):
            sub = df[df.capa == capa].sort_values("rho")
            good = sub[~sub.bad]
            ext = sub[sub.bad & (sub.rho > good.rho.max())].head(1) \
                if len(good) else sub.head(0)
            plot_sub = pd.concat([good, ext])
            ax.plot(plot_sub.rho, plot_sub[ycol], "o-", color=color, lw=1.9, ms=3.4,
                    label=T(f"정원={capa}", f"capa = {capa}"))
        lo, hi = good_all[ycol].min(), good_all[ycol].max()
        pad = 0.03 * (hi - lo)
        ax.set_ylim(min(lo - pad, -pad if zeroline else lo - pad), hi + pad)
        if zeroline:
            ax.axhline(0.0, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_ylabel(ylab)
        ax.set_xlabel(T("e-hail 부하율 (η)", "e-hail load factor (η)"))
        _panel_tag(ax, letter, corner=corner)
        _style_axes(ax)
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, fontsize=9.0,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))

    for ax, title in ((ax_a, T("e-hail 이용자", "e-hail users")),
                      (ax_c, T("street-hail 이용자", "street-hail users")),
                      (ax_e, T("전체", "total"))):
        pos = ax.get_position()
        fig.text(0.015, (pos.y0 + pos.y1) / 2, title, rotation=90,
                  ha="left", va="center", fontsize=13, fontweight="bold", color=INK2)

    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_D0():
    k = BASE.kappa
    pi_do = 100.0
    lam_do = pi_do * BASE.v / BASE.R**1.5
    m_c = 3 * (k * pi_do / 2) ** (2 / 3) + k * pi_do

    rows = []
    m_grid = np.linspace(m_c * 1.05, m_c * 3.2, 16)
    for m in m_grid:
        res = evaluate(BASE, p=1.0, r=0.0, capa=1, m=float(m), lam=lam_do, balking=False)
        n00 = res["n_HT"][(0, 0)]
        n01_model, n10_model = res["n_HT"][(0, 1)], res["n_HT"][(1, 0)]
        n01_ref = k * pi_do / np.sqrt(n00) if n00 > 0 else np.nan
        n10_ref = k * pi_do
        rows.append(dict(panel="do3_n01", x=m, model=n01_model, ref=n01_ref))
        rows.append(dict(panel="do3_n10", x=m, model=n10_model, ref=n10_ref))
    kp = BASE.kappa * 100.0
    n_t = 20.0
    den = 2 * kp + n_t**1.5
    do_ref = {(0, 0): n_t * (kp + n_t**1.5) / den, (0, 1): kp * n_t / den,
              (0, 2): kp**2 / (2 * kp * np.sqrt(n_t) + n_t**2),
              (1, 1): kp**2 / (2 * kp * np.sqrt(n_t) + n_t**2),
              (1, 0): kp * (kp + n_t**1.5) / den, (2, 0): kp**2 / (np.sqrt(2) * den)}
    m_do = sum(do_ref.values())
    pr_do = HybridDRTParams(m=m_do, r=0.0, p=1.0, lam=100.0, R=1.0, v=1.0,
                            capa=2, kappa=BASE.kappa, balking=False,
                            availability_rule="b", flexible_dropoff=False)
    res_do = HybridDRTModel(pr_do).solve(s1_init=n_t)
    do_states = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (2, 0)]
    for st in do_states:
        rows.append(dict(panel="do19", x=str(st), model=res_do["n_HT"][st], ref=do_ref[st]))
    df = pd.DataFrame(rows)

    OUR = T("본 모형", "Our model")
    DO = "Daganzo and Ouyang (2019)"
    N01 = T("n01", "$n_{01}$")
    N10 = T("n10", "$n_{10}$")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.1))
    ax = axes[0]
    sub01 = df[df.panel == "do3_n01"]
    sub10 = df[df.panel == "do3_n10"]
    ax.plot(sub01.x, sub01.model, color=CAT["blue"], lw=2.2, linestyle="-",
           label=f"{OUR}, {N01}")
    ax.plot(sub01.x, sub01.ref, "o", mfc="none", mec=CAT["red"], ms=7, mew=1.6,
           label=f"{DO}, {N01}")
    ax.plot(sub10.x, sub10.model, color=CAT["blue"], lw=2.2, linestyle="--",
           label=f"{OUR}, {N10}")
    ax.plot(sub10.x, sub10.ref, "s", mfc="none", mec=CAT["red"], ms=7, mew=1.6,
           label=f"{DO}, {N10}")
    err1 = np.nanmax(np.abs(np.concatenate([sub01.model - sub01.ref, sub10.model - sub10.ref])) /
                     np.concatenate([sub01.ref, sub10.ref]))
    ax.set_xlabel(T("차량 대수 (m)", "fleet size ($m$)"))
    ax.set_ylabel(T("차량 대수 (n)", "number of vehicles ($n_{ij}$)"))
    ax.legend(fontsize=8.5, frameon=False, loc="center right")
    _panel_tag(ax, "a")
    _style_axes(ax)
    ax = axes[1]
    sub = df[df.panel == "do19"]
    xpos = np.arange(len(sub))
    ax.bar(xpos - 0.18, sub.model, width=0.36, color=CAT["blue"], label=OUR)
    ax.bar(xpos + 0.18, sub.ref, width=0.36, color="none", edgecolor=CAT["red"], lw=1.6, label=DO)
    err2 = np.nanmax(np.abs(sub.model.to_numpy() - sub.ref.to_numpy()) / sub.ref.to_numpy())
    ax.set_xticks(xpos); ax.set_xticklabels([s for s in sub.x], fontsize=9)
    ax.set_xlabel(T("차량 상태 (탑승 i, 배정대기 j)", "vehicle state (onboard $i$, assigned $j$)"))
    ax.set_ylabel(T("차량 대수 (n)", "number of vehicles ($n_{ij}$)"))
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor="none",
             loc="upper right")
    _panel_tag(ax, "b")
    _style_axes(ax)
    fig.tight_layout()
    print(f"D0 max relative error - (a) taxi: {err1:.2e}, (b) shared taxi: {err2:.2e}")
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_D4():
    from dataclasses import replace as dc_replace
    lam = _lam_for(0.6, 40.0)
    R, v, l = BASE.R, BASE.v, BASE.l
    drive = l / v

    DRT_BLUES = {2: "#7db6e8", 3: "#2a78d6", 5: "#123f83"}
    ST_GREEN = "#7fbf7f"
    cases = [
        ("taxi", dict(capa=1), MUTED, T("택시", "taxi")),
        ("st2", dict(capa=2, availability_rule="b", flexible_dropoff=False),
         ST_GREEN, T("합승택시", "shared-taxi")),
        ("drt2", dict(capa=2), DRT_BLUES[2], T("DRT (정원=2)", "DRT (capa = 2)")),
        ("drt3", dict(capa=3), DRT_BLUES[3], T("DRT (정원=3)", "DRT (capa = 3)")),
        ("drt5", dict(capa=5), DRT_BLUES[5], T("DRT (정원=5)", "DRT (capa = 5)")),
    ]
    s1_grid = np.geomspace(0.12, 400.0, 220)
    rows = []
    for key, kw, _, _ in cases:
        pr = dc_replace(BASE, m=1.0, p=1.0, r=0.0, lam=lam, balking=False, **kw)
        mdl = HybridDRTModel(pr)
        idx, elig = mdl.idx, mdl.eligible
        for s1 in s1_grid:
            phi = mdl._solve_fleet(1.0, True, float(s1), 1.0)
            if phi is None or np.min(phi) < -1e-9:
                continue
            g = sum(phi[idx[n]] for n in elig)
            if g <= 1e-12:
                continue
            M = s1 / g
            occ = sum((i + j) * phi[idx[(i, j)]] for (i, j) in idx) * M
            ft = occ / (lam * R) / drive
            rows.append(dict(mode=key, m=M, f_t=ft, s1=s1))
    for m in np.linspace(7.0, 255.0, 120):
        rows.append(dict(mode="transit", m=m, f_t=1.0 + 4.0 * np.sqrt(5.0 * R / m) / l))
    m_auto = lam * R * drive
    rows.append(dict(mode="auto", m=m_auto, f_t=1.0))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    X_MAX = 120.0
    lower, upper = {}, {}
    for key, _, color, label in cases:
        sub = df[df["mode"] == key].sort_values("s1").reset_index(drop=True)
        i_c = int(sub.m.idxmin())
        lower[key] = sub.iloc[i_c:][sub.iloc[i_c:].m <= X_MAX]
        upper[key] = sub.iloc[:i_c + 1][sub.iloc[:i_c + 1].m <= X_MAX]
        full = sub[sub.m <= X_MAX]
        ax.plot(full.m, full.f_t, color=color, ls="-", lw=2.1)
    tr = df[df["mode"] == "transit"].sort_values("m")
    ax.plot(tr.m, tr.f_t, color=CAT["violet"], ls="-", lw=2.1)
    m_env = np.linspace(7.0, X_MAX - 2.0, 400)
    env = np.interp(m_env, tr.m.to_numpy(), tr.f_t.to_numpy())
    for s in lower.values():
        interp = np.interp(m_env, s.m.to_numpy(), s.f_t.to_numpy(),
                           left=np.inf, right=np.inf)
        env = np.minimum(env, interp)
    from matplotlib import patheffects as _pe
    X_SPAN, Y_SPAN = X_MAX, 5.0 - 0.8
    m_n = m_env / X_SPAN
    env_n = env / Y_SPAN
    slope_n = np.gradient(env_n, m_n)
    norm = np.sqrt(slope_n ** 2 + 1.0)
    dir_x, dir_y = slope_n / norm, -1.0 / norm
    OFFSET = 0.022
    m_off = m_env + dir_x * OFFSET * X_SPAN
    env_off = env + dir_y * OFFSET * Y_SPAN
    frontier_line, = ax.plot(m_off, env_off, color=CAT["red"], ls=(0, (4, 2)),
                              lw=3.2, alpha=1.0, zorder=1.6)
    frontier_line.set_path_effects(
        [_pe.Stroke(linewidth=5.5, foreground="white"), _pe.Normal()])
    ax.plot([m_auto], [1.0], "o", color=INK, ms=9)
    ax.axhline(1.0, color=AXIS, lw=1.0, ls=(0, (1, 2)))
    TXT_HALO = [_pe.Stroke(linewidth=3.2, foreground="white"), _pe.Normal()]

    def label_on_upper(key, f_target, label, color, dx=4.0, dy=0.0):
        u = upper[key].sort_values("f_t")
        m_at = float(np.interp(f_target, u.f_t.to_numpy(), u.m.to_numpy()))
        ax.text(m_at + dx, f_target + dy, label, fontsize=12.5, fontweight="bold",
                color=color, va="center", path_effects=TXT_HALO)
    label_on_upper("drt5", 4.8, T("DRT (정원=5)", "DRT (capa = 5)"), DRT_BLUES[5])
    label_on_upper("drt3", 4.2, T("DRT (정원=3)", "DRT (capa = 3)"), DRT_BLUES[3])
    label_on_upper("drt2", 3.6, T("DRT (정원=2)", "DRT (capa = 2)"), DRT_BLUES[2])
    # (dx 4.0→1.0).
    label_on_upper("st2", 2.6, T("합승택시", "shared-taxi"), ST_GREEN, dx=1.0)
    label_on_upper("taxi", 1.75, T("택시", "taxi"), MUTED, dx=5.0)
    ax.text(84, 1.0 + 4.0 * np.sqrt(5.0 * R / 84) / l + 0.12,
            T("재래 대중교통", "conventional transit"), fontsize=12.5, fontweight="bold",
            color=CAT["violet"], path_effects=TXT_HALO)
    ax.text(m_auto + 5, 0.985, T("승용차", "auto"), fontsize=12.5, fontweight="bold",
            color=INK, va="center", path_effects=TXT_HALO)
    ax.text(80, 0.95, T("Pareto frontier", "Pareto frontier"), fontsize=12.5,
            fontweight="bold", color=CAT["red"], ha="center", path_effects=TXT_HALO)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0.8, 5.0)
    ax.set_xlabel(T("차량 대수 (m)", "fleet size ($m$)"))
    ax.set_ylabel(T("이용자 통행시간", "user travel time"))
    _style_axes(ax)
    fig.tight_layout()
    kpi = BASE.kappa * lam * R * np.sqrt(R) / v
    mc_do = 3 * (kpi / 2) ** (2 / 3) + kpi
    sub = df[df["mode"] == "taxi"]
    print(f"D4 check: taxi m_c(parametric)={sub.m.min():.2f} vs D&O closed form={mc_do:.2f}")
    for key in ["st2", "drt2", "drt3", "drt5"]:
        print(f"D4 m_c {key}: {df[df['mode'] == key].m.min():.1f}")
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_E4():
    p_ref, r_ref = 0.6, 0.4
    rho, m = 0.6, 40.0
    lam = _lam_for(rho, m)
    capa_cols = [1, 5]

    rows = []
    p_grid = np.linspace(0.05, 0.95, 19)
    r_grid = np.linspace(0.0, 0.95, 20)
    for capa_a in capa_cols:
        for pv in p_grid:
            res = evaluate(BASE, capa=capa_a, m=m, p=float(pv), r=r_ref, lam=lam)
            bad = _collapsed(res)
            TwE = np.nan if bad else (res["T_w_E"] or 0.0) * 60
            WS = np.nan if bad else (res["T_w_S_avg"] or 0.0) * 60
            rows.append(dict(panel="user", sweep="p", capa=capa_a, x=pv,
                             TwE_min=TwE, WS_min=WS,
                             overall_min=pv * TwE + (1 - pv) * WS))
        for rv in r_grid:
            res = evaluate(BASE, capa=capa_a, m=m, p=p_ref, r=float(rv), lam=lam)
            bad = _collapsed(res)
            TwE = np.nan if bad else (res["T_w_E"] or 0.0) * 60
            WS = np.nan if bad else (res["T_w_S_avg"] or 0.0) * 60
            rows.append(dict(panel="user", sweep="r", capa=capa_a, x=rv,
                             TwE_min=TwE, WS_min=WS,
                             overall_min=p_ref * TwE + (1 - p_ref) * WS))
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(8.4, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.25,
                          top=0.86, bottom=0.15, left=0.09, right=0.97)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1], sharey=ax00)
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1], sharey=ax10)
    axes = np.array([[ax00, ax01], [ax10, ax11]])
    xlabels = {"p": T("e-hailing 비율 (p)", "e-hail proportion ($p$)"),
               "r": T("AV 비율 (r)", "AV proportion ($r$)")}
    letters = {("p", 1): "a", ("p", 5): "b", ("r", 1): "c", ("r", 5): "d"}
    for ri, sweep in enumerate(["p", "r"]):
        for ci, capa_a in enumerate(capa_cols):
            ax = axes[ri, ci]
            sub = df[(df.sweep == sweep) & (df.capa == capa_a)].sort_values("x")
            ax.plot(sub.x, sub.TwE_min, marker="o", color=CAT["red"], lw=2.0, ms=3.4,
                    label="e-hail")
            ax.plot(sub.x, sub.WS_min, marker="s", color=CAT["green"], lw=2.0, ms=3.4,
                    label="street-hail")
            ax.plot(sub.x, sub.overall_min, color=INK, lw=2.3, ls=(0, (4, 2)),
                    label=T("전체", "total"))
            ax.set_xlabel(xlabels[sweep])
            if ci == 0:
                ax.set_ylabel(T("대기시간 [분]", "waiting time [min]"))
            _panel_tag(ax, letters[(sweep, capa_a)])
            _style_axes(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, fontsize=9.5,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    pos0, pos1 = axes[0, 0].get_position(), axes[0, 1].get_position()
    col0_cx = (pos0.x0 + pos0.x1) / 2.0
    col1_cx = (pos1.x0 + pos1.x1) / 2.0
    gap_x = (pos0.x1 + pos1.x0) / 2.0
    y_top = max(pos0.y1, pos1.y1)
    y_header = y_top + 0.025
    fig.text(col0_cx, y_header, T("정원 = 1", "capa = 1"), ha="center",
             va="bottom", fontsize=12, color=INK)
    fig.text(col1_cx, y_header, T("정원 = 5", "capa = 5"), ha="center",
             va="bottom", fontsize=12, color=INK)
    y_bottom = min(axes[1, 0].get_position().y0, axes[1, 1].get_position().y0)
    fig.add_artist(plt.Line2D([gap_x, gap_x], [y_bottom, y_header - 0.005],
                              transform=fig.transFigure, color=INK2, lw=1.0,
                              ls=(0, (3, 3))))
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_E4b():
    rho, m, capa = 0.6, 40.0, 5
    lam = _lam_for(rho, m)

    rows = []
    p_grid = np.linspace(0.1, 0.9, 9)
    r_grid2 = np.linspace(0.0, 0.95, 20)
    Z = np.full((len(r_grid2), len(p_grid)), np.nan)
    for b, pv in enumerate(p_grid):
        for a, rv in enumerate(r_grid2):
            res = evaluate(BASE, capa=capa, m=m, p=float(pv), r=float(rv), lam=lam)
            if not _collapsed(res):
                Z[a, b] = res["z_beta"]
            rows.append(dict(panel="map", p=pv, r=rv, z_beta=Z[a, b]))
    r_star = [r_grid2[int(np.nanargmin(Z[:, b]))] for b in range(len(p_grid))]
    for pv, rs in zip(p_grid, r_star):
        rows.append(dict(panel="rstar", p=pv, r=rs))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    Rg, Pg = np.meshgrid(r_grid2, p_grid)
    cf = ax.contourf(Rg, Pg, Z.T, levels=12, cmap="Blues")
    cb = fig.colorbar(cf, ax=ax)
    _colorbar_title(cb, T("통행당 사회적\n비용 z/β [h]", "social cost per trip\n$z/\\beta$ [h]"),
                    fontsize=8.5, ticklabelsize=7)
    ax.plot(r_star, p_grid, "o--", color=CAT["red"], lw=2.0, ms=4.5,
            label=T("비용 최소 AV 비율 r*(p)", "cost-minimizing AV proportion $r^*(p)$"))
    ax.set_xlabel(T("AV 비율 (r)", "AV proportion ($r$)"))
    ax.set_ylabel(T("e-hail 비율 (p)", "e-hail proportion ($p$)"))
    ax.legend(fontsize=8.5, frameon=False, loc="upper center")
    _style_axes(ax)
    ax.grid(False)
    fig.tight_layout()
    _recenter_colorbar_title(fig, cb)
    return df, fig


# ══════════════════════════════════════════════════════════════════════
#      analysis_plan.md §6.5.
# ══════════════════════════════════════════════════════════════════════
def fig_E5():
    from drtmodel.model import catchment_EN, expected_insertion_detour
    rho, p, r, m, capa = 0.6, 0.6, 0.4, M_V2, 3
    lam = _lam_for(rho, m)
    sqrtR = np.sqrt(BASE.R)

    de_grid = np.linspace(0.3, 2.8, 26)
    rows = []
    for de in de_grid:
        pi2 = de / sqrtR
        EN = catchment_EN(pi2)
        d_ins = expected_insertion_detour(pi2) * sqrtR
        base_i = replace_base(capa=capa, m=m, p=p, r=r, lam=lam, de_max=float(de))
        mdl = HybridDRTModel(base_i)
        res = mdl.solve()
        delta = mdl.detour_inflation(res)
        rows.append(dict(de_max=de, EN_pct=EN * 100, d_ins_km=d_ins,
                         sigma_S=res["sigma"] or 0.0, delta_pct=delta * 100))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    ax = axes[0]
    ax.plot(df.de_max, df.EN_pct, color=CAT["blue"], lw=2.2,
            label=T("흡수 확률 (캐치먼트, 식 12)", "absorption probability (catchment, Eq. 12)"))
    ax.set_ylabel(T("흡수 확률 [%]", "absorption probability [%]"), color=CAT["blue"])
    ax.tick_params(axis="y", labelcolor=CAT["blue"])
    ax2 = ax.twinx()
    ax2.plot(df.de_max, df.d_ins_km, color=CAT["orange"], lw=2.2, ls=(0, (4, 2)),
            label=T("흡수 1건당 기대 우회거리 (식 23)", "expected detour per absorption (Eq. 23)"))
    ax2.set_ylabel(T("기대 우회거리 [km]", "expected detour [km]"), color=CAT["orange"])
    ax2.tick_params(axis="y", labelcolor=CAT["orange"])
    ax.axvline(1.0, color=MUTED, lw=1.0, ls="--", zorder=0)
    ax.text(1.03, 0.95, T("기준값 1km", "baseline 1 km"), transform=ax.get_xaxis_transform(),
            fontsize=8.5, color=MUTED, va="top")
    ax.set_xlabel(T("허용 우회거리 [km]", "detour allowance de_max [km]"))
    ax.set_title(T("① 허용 우회거리의 트레이드오프 (모형 순수 함수)",
                   "① Detour-allowance trade-off (pure functions)"),
                 fontsize=10.8, color=INK)
    _style_axes(ax)
    ax = axes[1]
    ax.plot(df.de_max, df.sigma_S, color=CAT["green"], lw=2.2,
            label=T("street-hail 서비스율", "street-hail service probability σ_S"))
    ax.set_ylabel(T("street-hail 서비스율", "street-hail service probability σ_S"), color=CAT["green"])
    ax.tick_params(axis="y", labelcolor=CAT["green"])
    ax.set_ylim(0, max(df.sigma_S) * 1.3)
    ax2 = ax.twinx()
    ax2.plot(df.de_max, df.delta_pct, color=CAT["violet"], lw=2.2, ls=(0, (4, 2)),
            label=T("차내시간 증가율", "in-vehicle time inflation"))
    ax2.set_ylabel(T("차내시간 증가율 [%]", "in-vehicle time inflation [%]"), color=CAT["violet"])
    ax2.tick_params(axis="y", labelcolor=CAT["violet"])
    ax.set_xlabel(T("허용 우회거리 [km]", "detour allowance de_max [km]"))
    ax.set_title(T(f"② 시스템 반응 (부하율 {rho} 작동점) — 흡수량은 둔감, 우회 비용만 반응",
                   f"② System response (ρ={rho}) — only detour cost reacts"),
                 fontsize=10.8, color=INK)
    _style_axes(ax)
    ax = axes[2]
    w_grid = np.linspace(0.0, 3.0, 200)
    ax.plot(w_grid, 1 - np.exp(-w_grid), color=INK, lw=2.0,
            label=T("보편 곡선 (식 15)", "universal curve (Eq. 15)"))
    ax.axhline(0.5, color=MUTED, lw=1.0, ls="--", zorder=0)
    ax.text(2.95, 0.51, T("서비스율 50%", "service probability 50%"), fontsize=8.5, color=MUTED,
            ha="right", va="bottom")
    marks = [(T("대기 5분, 100대", "wt_max=5 min, m=100"), 5.0, 100.0, CAT["blue"]),
             (T("대기 15분, 100대", "wt_max=15 min, m=100"), 15.0, 100.0, CAT["aqua"]),
             (T("대기 60분, 100대 (기준)", "wt_max=60 min, m=100 (base)"), 60.0, 100.0, CAT["red"]),
             (T("대기 60분, 300대", "wt_max=60 min, m=300"), 60.0, 300.0, CAT["orange"])]
    for label, wt, mm, color in marks:
        res = evaluate(BASE, capa=capa, m=mm, p=p, r=r, lam=lam, wt_max=wt / 60.0)
        w_val = BASE.v * res["n_eff"] * (wt / 60.0) / BASE.L
        ax.plot([w_val], [res["sigma"] or 0.0], "o", color=color, ms=8, zorder=5)
        ax.annotate(label, (w_val, res["sigma"] or 0.0), textcoords="offset points",
                    xytext=(8, -4), fontsize=8.5, color=color)
    ax.set_xlabel(T("조우 강도 (차량 통과율 × 최대 대기시간)",
                    "encounter intensity (vehicle passing rate × patience)"))
    ax.set_ylabel(T("street-hail 서비스율", "street-hail service probability σ_S"))
    ax.set_title(T("③ street-hail 성립의 보편 곡선 — 조우 강도 하나로 결정",
                   "③ Universal street-hail viability curve"),
                 fontsize=10.8, color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    _style_axes(ax)
    fig.suptitle(T(f"E5 · [Task 2] 운영 레버 — 허용 우회거리·최대 대기시간의 역할 분담 (부하율 {rho}, 정원={capa})",
                   f"E5 · [Task 2] Operational levers — division of roles between detour allowance and patience (ρ={rho}, c={capa})"),
                fontsize=13, y=1.03, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_E6():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    sigmaS_target = 0.3
    lam = _lam_for(0.6, 40.0)
    capas_e6 = [1, 5]
    p_grid = np.linspace(0.05, 0.95, 10)
    r_grid = np.linspace(0.0, 0.9, 10)
    P, Rm = np.meshgrid(p_grid, r_grid)

    def _feasible(capa, p, r, m):
        res = evaluate(BASE, capa=capa, m=float(m), p=p, r=r, lam=lam)
        return (not _collapsed(res)) and (res["sigma"] or 0.0) >= sigmaS_target

    def _m_min(capa, p, r):
        lo, hi = 2.0, 60.0
        tries = 0
        while not _feasible(capa, p, r, hi):
            hi *= 2.0
            tries += 1
            if tries > 6:
                return np.nan
        while hi - lo > 3e-3 * hi:
            mid = 0.5 * (lo + hi)
            if _feasible(capa, p, r, mid):
                hi = mid
            else:
                lo = mid
        return hi

    rows = []
    surf_by_capa = {}
    for capa in capas_e6:
        Mc = np.zeros_like(P)
        for a in range(P.shape[0]):
            for b in range(P.shape[1]):
                pv, rv = float(P[a, b]), float(Rm[a, b])
                mc = _m_min(capa, pv, rv)
                res = evaluate(BASE, capa=capa, m=mc, p=pv, r=rv, lam=lam)
                Mc[a, b] = mc
                rows.append(dict(capa=capa, p=pv, r=rv, m_c=mc, z_beta=res["z_beta"]))
        surf_by_capa[capa] = Mc
    df = pd.DataFrame(rows)

    mc_lo, mc_hi = df.m_c.min(), df.m_c.max()
    cnorm = matplotlib.colors.PowerNorm(gamma=0.45, vmin=mc_lo, vmax=mc_hi)
    cb_ticks = [20, 40, 60, 80, 120, 160]
    fig = plt.figure(figsize=(8.8, 3.8))
    axes3d, cbs = [], []
    for cidx, capa in enumerate(capas_e6):
        Mc = surf_by_capa[capa]
        ax = fig.add_subplot(1, 2, cidx + 1, projection="3d")
        surf = ax.plot_surface(P, Rm, Mc, cmap="viridis", edgecolor="k",
                               linewidth=0.25, antialiased=True, norm=cnorm)
        ax.set_xlabel(T("e-hailing 비율 (p)", "e-hail proportion ($p$)"),
                      fontsize=8.5, labelpad=2)
        ax.set_ylabel(T("AV 비율 (r)", "AV proportion ($r$)"),
                      fontsize=8.5, labelpad=2)
        ax.tick_params(labelsize=7.5, pad=1)
        ax.set_zlim(mc_lo, mc_hi)
        cb = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, ticks=cb_ticks)
        _colorbar_title(cb, T("최소 차량 대수 (m)", "minimum fleet size ($m$)"),
                        fontsize=8.5, ticklabelsize=7)
        axes3d.append(ax)
        cbs.append(cb)
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, cb, letter, capa in zip(axes3d, cbs, ["a", "b"], capas_e6):
        bbox_disp = ax.get_tightbbox(renderer)
        inv = fig.transFigure.inverted()
        x0, y0 = inv.transform((bbox_disp.x0, bbox_disp.y0))
        x1, y1 = inv.transform((bbox_disp.x1, bbox_disp.y1))
        cx = (x0 + x1) / 2.0
        fig.text(cx, y1 + 0.02, T(f"정원 = {capa}", f"capa = {capa}"),
                 ha="center", va="bottom", fontsize=10.5, color=INK)
        fig.text(cx, y0 - 0.06, f"({letter})",
                 ha="center", va="top", fontsize=9.5, color=INK)
        _recenter_colorbar_title(fig, cb)
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_E6b():
    sigmaS_target = 0.5
    lam = _lam_for(0.6, 40.0)

    p_curves = [(0.2, CAT["violet"]), (0.6, CAT["aqua"]), (0.9, CAT["orange"])]
    r_grid = np.linspace(0.0, 0.9, 19)
    capa_cols = [1, 3]

    def _feasible(capa, p, r, m):
        res = evaluate(BASE, capa=capa, m=float(m), p=p, r=r, lam=lam)
        return (not _collapsed(res)) and (res["sigma"] or 0.0) >= sigmaS_target

    def _m_min(capa, p, r):
        lo, hi = 2.0, 60.0
        tries = 0
        while not _feasible(capa, p, r, hi):
            hi *= 2.0
            tries += 1
            if tries > 6:
                return np.nan
        while hi - lo > 3e-3 * hi:
            mid = 0.5 * (lo + hi)
            if _feasible(capa, p, r, mid):
                hi = mid
            else:
                lo = mid
        return hi

    rows = []
    for capa in capa_cols:
        for p, _ in p_curves:
            for rv in r_grid:
                rows.append(dict(capa=capa, p=p, r=float(rv),
                                 m_min=_m_min(capa, p, float(rv))))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, capa in zip(axes, capa_cols):
        for p, color in p_curves:
            sub = df[(df.capa == capa) & (df.p == p)].sort_values("r")
            ax.plot(sub.r, sub.m_min, color=color, lw=2.1,
                    label=T(f"p={p}", f"$p$={p}"))
        ax.set_xlabel(T("AV 비율 (r)", "AV proportion ($r$)"), fontsize=9)
        _panel_tag(ax, "a" if capa == 1 else "b")
        _style_axes(ax)
        ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel(T("최소 차량 대수 (m)", "minimum fleet size ($m$)"),
                       fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    fig.legend(handles, labels, ncol=3, frameon=False, fontsize=7.5,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    for ax, capa in zip(axes, capa_cols):
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2.0, pos.y1 + 0.03,
                 T(f"정원 = {capa}", f"capa = {capa}"),
                 ha="center", va="bottom", fontsize=9.5, color=INK)
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_E7():
    p_grid = np.linspace(0.05, 0.95, 19)
    m = M_V2
    rho_cases = [0.4, 0.6, 0.85]
    combos = [(1, 0.2, CAT["red"], "-"), (1, 0.6, CAT["red"], (0, (4, 2))),
              (3, 0.2, CAT["aqua"], "-"), (3, 0.6, CAT["aqua"], (0, (4, 2))),
              (10, 0.2, CAT["violet"], "-"), (10, 0.6, CAT["violet"], (0, (4, 2)))]

    rows = []
    for rho in rho_cases:
        lam = _lam_for(rho, m)
        for p in p_grid:
            for capa, r, _, _ in combos:
                res = evaluate(BASE, capa=capa, m=m, p=float(p), r=r, lam=lam)
                bad = _collapsed(res)
                rows.append(dict(rho=rho, p=p, capa=capa, r=r,
                                 T_w_E_min=np.nan if bad else (res["T_w_E"] or 0.0) * 60,
                                 sigma_S=np.nan if bad else (res["sigma"] or 0.0),
                                 unmet_total=np.nan if bad else
                                 res["unmet_E"] + res["unmet"]))
    df = pd.DataFrame(rows)

    metrics = [("T_w_E_min", T("e-hail 대기시간 (호출→탑승)", "e-hail waiting time (request to pickup)"),
                T("e-hail 대기시간 [분]", "e-hail waiting time [min]")),
               ("sigma_S", T("street-hail 서비스율", "street-hail service probability σ_S"),
                T("street-hail 서비스율", "street-hail service probability σ_S")),
               ("unmet_total", T("미충족 수요 총량", "total unmet demand"),
                T("미충족 수요 [건/시]", "unmet demand [trips/h]"))]
    marks = ["①②③", "④⑤⑥", "⑦⑧⑨"]
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 12.6), sharex=True)
    for ri, (ycol, title, ylab) in enumerate(metrics):
        vals = df[ycol].to_numpy()
        y_top = np.nanpercentile(vals, 99.5) * 1.08 if np.isfinite(vals).any() else None
        for ci, rho in enumerate(rho_cases):
            ax = axes[ri, ci]
            for capa, r, color, ls in combos:
                sub = df[(df.rho == rho) & (df.capa == capa)
                         & (df.r == r)].sort_values("p")
                ax.plot(sub.p, sub[ycol], color=color, lw=1.9, ls=ls,
                        label=T(f"정원={capa}, AT비율={r}", f"c={capa}, r={r}"))
            ax.set_ylabel(ylab, fontsize=9.4)
            ax.set_title(T(f"{marks[ri][ci]} {title} — 부하율={rho}",
                           f"{marks[ri][ci]} {title} — ρ={rho}"), fontsize=10.2, color=INK)
            if y_top is not None:
                ax.set_ylim(0, y_top)
            if ri == len(metrics) - 1:
                ax.set_xlabel(T("e-hailing 비율", "e-hail share p"))
            _style_axes(ax)
    axes[0, 0].legend(fontsize=7.8, frameon=False, ncol=2)
    fig.suptitle(T("E7 · [Task 2] e-hailing 비율×AT 비율 구분 제시 ([T-ITS] 문법) — 정원 3수준 × 부하율 3수준 "
                   f"(차량대수={m:.0f})",
                   "E7 · [Task 2] e-hail share × AT share disaggregation ([T-ITS] grammar) — three capacities × three load factors "
                   f"(m={m:.0f})"),
                fontsize=12.8, y=0.999, color=INK)
    fig.tight_layout()
    return df, fig


FIGS = {"A1": fig_A1, "A2": fig_A2, "A3": fig_A3, "A4": fig_A4,
       "A5": fig_A5, "A6": fig_A6,
       "B1": fig_B1, "B2": fig_B2, "B3": fig_B3, "B4": fig_B4, "B5": fig_B5,
       "B6": fig_B6, "B7": fig_B7, "B8": fig_B8,
       "D0": fig_D0, "D1": fig_D1, "D2": fig_D2, "D3": fig_D3, "D4": fig_D4,
       "E1": fig_E1, "E2": fig_E2, "E3": fig_E3, "E4": fig_E4, "E4B": fig_E4b,
       "E5": fig_E5,
       "E6": fig_E6, "E6B": fig_E6b, "E7": fig_E7}


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--en"]
    if "--en" in sys.argv[1:]:
        set_lang("en")
    target = argv[0] if argv else "all"
    todo = list(FIGS) if target == "all" else [target]
    for fid in todo:
        df, fig = FIGS[fid]()
        _save(fig, fid, df)
