import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
# -*- coding: utf-8 -*-
"""Figures 11-12 and the robustness table (Table 6) of the paper. Usage: python sensitivity_figures.py F7 --en"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

plt.rcParams["axes.unicode_minus"] = False

import analysis_figures as _af
from analysis_figures import (  # noqa: E402
    BASE, CAT, GRID, INK, INK2, M_V2, MUTED, SURFACE, T, _apply_panel_tags,
    _collapsed, _lam_for, _panel_tag, _style_axes, replace_base, set_lang,
)
from drtmodel.model import HybridDRTModel  # noqa: E402
from drtmodel.optimize import evaluate, find_m_c  # noqa: E402

# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════

M_F = M_V2
RHO_LEVELS = [0.30, 0.60, 0.85]
RHO_LABELS = ["부하율 (η) = 0.30", "부하율 (η) = 0.60", "부하율 (η) = 0.85"]
RHO_LABELS_EN = ["load factor (η) = 0.30", "load factor (η) = 0.60",
                 "load factor (η) = 0.85"]
RHO_COLORS = [CAT["blue"], CAT["aqua"], CAT["red"]]


def _rho_labels():
    """_rho_labels (see the paper for the definitions)."""
    return RHO_LABELS_EN if _af.LANG == "en" else RHO_LABELS

BASE_PT = dict(p=0.6, r=0.4, capa=3, m=M_F,
               wt_max=BASE.wt_max, te_max=BASE.te_max, de_max=BASE.de_max)

PARAMS = [
    ("p", "e비율 p", 0.6, 0.05, 0.95, lambda v: f"{v:.2f}"),
    ("r", "AT비율 r", 0.4, 0.00, 0.90, lambda v: f"{v:.2f}"),
    ("wt_max", "인내 wt_max", BASE.wt_max, 5 / 60, 60 / 60, lambda v: f"{v * 60:.0f}분"),
    ("te_max", "배차한도 te_max", BASE.te_max, 2 / 60, 60 / 60, lambda v: f"{v * 60:.0f}분"),
    ("de_max", "허용우회 de_max", BASE.de_max, 0.25, 2.50, lambda v: f"{v:.2f}km"),
]
PARAM_KEYS = [k for k, *_ in PARAMS]
PARAM_NAMES = {k: n for k, n, *_ in PARAMS}
PARAM_NAMES_EN = {"p": "e-hail share p", "r": "AT share r",
                  "wt_max": "patience wt_max", "te_max": "dispatch limit te_max",
                  "de_max": "detour allowance de_max"}
PARAM_BOX = {k: (lo, hi) for k, _, _, lo, hi, _ in PARAMS}


def _pname(k):
    """_pname (see the paper for the definitions)."""
    return PARAM_NAMES_EN[k] if _af.LANG == "en" else PARAM_NAMES[k]

OUTPUTS = [
    ("TwE", "e-hail 대기"),
    ("sigS", "street-hail 서비스율"),
    ("unmet", "미충족 총량"),
    ("z", "사회적 비용"),
    ("delta", "차내시간 팽창"),
]
OUT_NAMES = {k: n for k, n in OUTPUTS}
OUT_NAMES_EN = {"TwE": "e-hail waiting", "sigS": "street-hail service probability",
                "unmet": "total unmet demand", "z": "social cost",
                "delta": "in-vehicle time inflation"}
OUT_COLORS = [CAT["red"], CAT["aqua"], CAT["blue"], CAT["violet"], CAT["orange"]]


def _oname(k):
    """_oname (see the paper for the definitions)."""
    return OUT_NAMES_EN[k] if _af.LANG == "en" else OUT_NAMES[k]


def _outputs(res):
    """_outputs (see the paper for the definitions)."""
    def _pos(x):
        return float(x) if (x is not None and np.isfinite(x) and abs(x) > 1e-12) else np.nan
    return {
        "TwE": _pos((res["T_w_E"] or np.nan) * 60.0),
        "sigS": _pos(res["sigma"]),
        "unmet": _pos((res["unmet_E"] or 0.0) + (res["unmet"] or 0.0)),
        "z": _pos(res["z_beta"]),
        "delta": _pos(res["delta_bar"] * 100.0),
        "sigE": _pos(res["sigma_E"]),
    }


def _solve(rho, **over):
    """_solve (see the paper for the definitions)."""
    kw = dict(BASE_PT)
    kw.update(over)
    kw["lam"] = _lam_for(rho, kw["m"])
    return evaluate(BASE, **kw)


def elasticity(rho, key, h=0.05, **over):
    """elasticity (see the paper for the definitions)."""
    kw = dict(BASE_PT)
    kw.update(over)
    x0 = kw[key] if key in kw else BASE_PT[key]
    y0 = _outputs(_solve(rho, **{**over, key: x0}))
    yp = _outputs(_solve(rho, **{**over, key: x0 * (1 + h)}))
    ym = _outputs(_solve(rho, **{**over, key: x0 * (1 - h)}))
    return {k: (yp[k] - ym[k]) / y0[k] / (2 * h) for k in y0}


def elasticity_rho(rho, h=0.05, **over):
    """elasticity_rho (see the paper for the definitions)."""
    y0 = _outputs(_solve(rho, **over))
    yp = _outputs(_solve(rho * (1 + h), **over))
    ym = _outputs(_solve(rho * (1 - h), **over))
    return {k: (yp[k] - ym[k]) / y0[k] / (2 * h) for k in y0}


def arc_elasticity_capa(rho, c_lo, c_hi, **over):
    """arc_elasticity_capa (see the paper for the definitions)."""
    ylo = _outputs(_solve(rho, capa=c_lo, **over))
    yhi = _outputs(_solve(rho, capa=c_hi, **over))
    out = {}
    for k in ylo:
        ybar = 0.5 * (ylo[k] + yhi[k])
        cbar = 0.5 * (c_lo + c_hi)
        out[k] = ((yhi[k] - ylo[k]) / ybar) / ((c_hi - c_lo) / cbar)
    return out


def _save(fig, fid, df=None):
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "outputs", "figures")
    os.makedirs(out, exist_ok=True)
    suffix = "_en" if _af.LANG == "en" else ""
    _apply_panel_tags(fig)
    fig.savefig(os.path.join(out, f"{fid}{suffix}.png"), dpi=140, bbox_inches="tight",
                facecolor="white")
    exp = os.environ.get("FIG_EXPORT_DIR")
    if exp:
        os.makedirs(exp, exist_ok=True)
        for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
            fig.savefig(os.path.join(exp, f"{fid}{suffix}.{ext}"), bbox_inches="tight",
                        facecolor="white", **kw)
    if df is not None:
        df.to_csv(os.path.join(out, f"{fid}{suffix}.csv"), index=False,
                  encoding="utf-8-sig")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_F0():
    """fig_F0 (see the paper for the definitions)."""
    rows = [
        (T("wt_max\n(승객 인내)", "wt_max\n(patience)"), r"$w_S = v\,wt_{max}\,n_{eff} / L$",
         "(M13) $\\sigma_S = 1-e^{-w_S}$", T("v / L 과 별칭", "aliased with v / L"), CAT["red"]),
        (T("te_max\n(배차 도달한도)", "te_max\n(dispatch limit)"),
         r"$w_E = \frac{\pi}{4\kappa^2} n^E (v\,te_{max})^2 / R$",
         "$\\sigma_E = 1-e^{-w_E}$", T("$v^2/R$ 과 별칭", "aliased with $v^2/R$"), CAT["orange"]),
        (T("de_max\n(허용 우회거리)", "de_max\n(detour allowance)"),
         r"$\pi_2 = de_{max} / \sqrt{R}$",
         "(M8) $E[N]$, (M17) $\\delta_{ins}$",
         T(r"$\sqrt{R}$ 과 별칭", r"aliased with $\sqrt{R}$"), CAT["violet"]),
        (T("λ, m\n(수요·공급)", "λ, m\n(demand & supply)"), r"$\rho = \lambda R \ell / (v\,m)$",
         T("전 방정식 (체제축)", "all equations (regime axis)"),
         T("개별 식별 불가", "not individually identified"), CAT["blue"]),
        (T("p, r, capa\n(구성·기술)", "p, r, capa\n(composition & technology)"),
         T("그 자체로 무차원", "dimensionless as-is"),
         T("(M4)(M7) 등", "(M4)(M7) etc."), T("별칭 없음", "no alias"), CAT["aqua"]),
    ]

    fig, ax = plt.subplots(figsize=(15.0, 7.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    xs = [4, 30, 60, 84]
    heads = [T("원 파라미터", "raw parameter"),
             T("무차원 군 (모형이 실제로 보는 것)", "dimensionless group (what the model sees)"),
             T("등장 수식", "appears in"), T("네트워크 별칭", "network alias")]
    for x, h in zip(xs, heads):
        ax.text(x, 93, h, fontsize=11.5, color=INK, fontweight="bold")

    y0, dy = 82, 15.0
    for i, (raw, group, eq, alias, color) in enumerate(rows):
        y = y0 - i * dy
        ax.add_patch(mpatches.FancyBboxPatch(
            (xs[0], y - 5.6), 22, 11.2, boxstyle="round,pad=0.4,rounding_size=1.2",
            facecolor=color, alpha=0.16, edgecolor=color, lw=1.6))
        ax.text(xs[0] + 11, y, raw, fontsize=10.2, color=INK, ha="center", va="center")
        ax.annotate("", xy=(xs[1] - 1.2, y), xytext=(xs[0] + 22.6, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0))
        ax.add_patch(mpatches.FancyBboxPatch(
            (xs[1], y - 5.6), 27.5, 11.2, boxstyle="round,pad=0.4,rounding_size=1.2",
            facecolor=color, alpha=0.30, edgecolor=color, lw=2.0))
        ax.text(xs[1] + 13.7, y, group, fontsize=10.6, color=INK, ha="center", va="center")
        ax.text(xs[2], y, eq, fontsize=10.0, color=INK2, va="center")
        ax.text(xs[3], y, alias, fontsize=9.6, color=MUTED, va="center", style="italic")

    ax.add_patch(mpatches.FancyBboxPatch(
        (3, 1.5), 94, 12.5, boxstyle="round,pad=0.6,rounding_size=1.5",
        facecolor=SURFACE, edgecolor=GRID, lw=1.4))
    ax.text(50, 10.6,
            T("명제 — 세 한도 파라미터(wt_max·te_max·de_max)는 각각 정확히 하나의 무차원 군을 통해서만 모형에 들어간다"
              " (코드 수준에서 확인: 각 변수의 등장 위치가 단 한 곳).",
              "Proposition — each limit parameter (wt_max, te_max, de_max) enters the model through exactly one"
              " dimensionless group (verified at code level)."),
            fontsize=10.6, color=INK, ha="center", va="center")
    ax.text(50, 5.2,
            T("따름정리 1: 네트워크 변수(R, v, κ, Δ)는 '제외'되는 것이 아니라 같은 군으로 흡수된다 — 고정 네트워크에서의 행태 스윕은 그 군에 대해 완전하다.    "
              "따름정리 2: 개별 파라미터는 식별되지 않는다 — de_max를 2배로 늘리는 것과 면적을 1/4로 줄이는 것은 모형상 같은 실험이다.",
              "Corollary 1: network variables (R, v, κ, Δ) are not 'excluded' but absorbed into the same groups — a fixed-network behavioral sweep is complete for them.\n"
              "Corollary 2: parameters are not identified individually — doubling de_max equals quartering the area."),
            fontsize=9.6, color=INK2, ha="center", va="center")

    fig.suptitle(T("F0 · [민감도] 무차원 군으로의 환원 — 왜 행태 파라미터 5개면 충분한가",
                   "F0 · [Sensitivity] Reduction to dimensionless groups — why five behavioral parameters suffice"),
                 fontsize=13.5, y=0.975, color=INK)
    return None, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_F1():
    """fig_F1 (see the paper for the definitions)."""
    rows = []
    for rho in RHO_LEVELS:
        for key in PARAM_KEYS:
            eps = elasticity(rho, key)
            for ok, val in eps.items():
                rows.append(dict(rho=rho, param=key, kind="연속", out=ok, eps=val))
        for ok, val in elasticity_rho(rho).items():
            rows.append(dict(rho=rho, param="rho", kind="연속", out=ok, eps=val))
        for c_lo, c_hi, tag in ((1, 2, "capa1to2"), (2, 3, "capa2to3")):
            for ok, val in arc_elasticity_capa(rho, c_lo, c_hi).items():
                rows.append(dict(rho=rho, param=tag, kind="이산", out=ok, eps=val))
    df = pd.DataFrame(rows)

    order = PARAM_KEYS + ["rho", "capa1to2", "capa2to3"]
    labels = [_pname(k) for k in PARAM_KEYS] + \
             [T("부하율 ρ", "load factor ρ"),
              T("정원 1→2 (이산)", "capacity 1→2 (discrete)"),
              T("정원 2→3 (이산)", "capacity 2→3 (discrete)")]
    out_keys = [k for k, _ in OUTPUTS]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.6), sharex=True)
    nb = len(out_keys)
    hgt = 0.78 / nb
    for ax, rho, rlab in zip(axes, RHO_LEVELS, _rho_labels()):
        sub = df[df.rho == rho]
        for bi, (ok, color) in enumerate(zip(out_keys, OUT_COLORS)):
            vals = [sub[(sub.param == p) & (sub.out == ok)].eps.mean() for p in order]
            ypos = np.arange(len(order)) + (bi - (nb - 1) / 2) * hgt
            ax.barh(ypos, vals, height=hgt * 0.92, color=color,
                    label=_oname(ok) if ax is axes[0] else None,
                    edgecolor="white", lw=0.4)
        ax.axvline(0, color=INK2, lw=1.1)
        ax.axhline(len(PARAM_KEYS) - 0.5, color=MUTED, lw=1.0, ls=(0, (4, 3)))
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(labels, fontsize=9.4)
        ax.invert_yaxis()
        ax.set_xlabel(T("정규화 탄력도  ε = (ΔY/Y) / (Δx/x)",
                        "normalized elasticity  ε = (ΔY/Y) / (Δx/x)"))
        ax.set_title(f"{rlab}", fontsize=11.2, color=INK)
        ax.set_xlim(-2.6, 2.6)
        _style_axes(ax)
    axes[0].legend(fontsize=8.6, frameon=False, loc="lower left",
                   title=T("출력 지표", "output metric"))
    axes[0].text(-2.5, len(PARAM_KEYS) - 0.75,
                 T("↑ 대상 파라미터   ↓ 참조축", "↑ target parameters   ↓ reference axes"),
                 fontsize=8.4, color=MUTED, va="center")
    fig.suptitle(T("F1 · [민감도] 탄력도 토네이도 — 무엇이 중요한가, 그리고 그 순위는 체제에 따라 바뀌는가 "
                   f"(기준점: e비율 0.6·AT비율 0.4·정원 3·차량 {M_F:.0f}대, 중심차분 ±5%)",
                   "F1 · [Sensitivity] Elasticity tornado — what matters, and does the ranking change with regime "
                   f"(base point: p=0.6, r=0.4, c=3, m={M_F:.0f}; central difference ±5%)"),
                 fontsize=12.4, y=1.0, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_F2():
    """fig_F2 (see the paper for the definitions)."""
    p_fix = 0.5
    rho_grid = np.linspace(0.30, 1.80, 16)
    r_grid = np.linspace(0.00, 0.88, 12)
    rows = []
    for ir, rv in enumerate(r_grid):
        for ig, rho in enumerate(rho_grid):
            eps = elasticity(rho, "p", p=p_fix, r=float(rv))
            base = _solve(rho, p=p_fix, r=float(rv))
            TwS = base["T_w_S"]  # = 1/q_eff [h]
            wS = (BASE.wt_max / TwS) if (TwS and np.isfinite(TwS) and TwS > 0) else np.nan
            rows.append(dict(panel="rho_map", rho=rho, r=rv, ir=ir, ig=ig,
                             eps_p_sigS=eps["sigS"], w_S=wS,
                             sigma_S=base["sigma"]))
    p_grid = np.linspace(0.10, 0.92, 14)
    PR_RHOS = (1.5, 0.6)
    for rho in PR_RHOS:
        for ir, rv in enumerate(r_grid):
            for ip, pv in enumerate(p_grid):
                eps = elasticity(rho, "p", p=float(pv), r=float(rv))
                rows.append(dict(panel=f"pr_map_{rho}", rho=rho, p=pv, r=rv,
                                 ir=ir, ip=ip, eps_p_sigS=eps["sigS"]))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(8.9, 6.8))

    ax = axes[0, 0]
    sub = df[df.panel == "rho_map"]
    Z = np.full((len(r_grid), len(rho_grid)), np.nan)
    Z[sub.ir.to_numpy(dtype=int), sub.ig.to_numpy(dtype=int)] = sub.eps_p_sigS.to_numpy()
    G, Rm = np.meshgrid(rho_grid, r_grid)
    lim = np.nanpercentile(np.abs(Z), 98)
    pm = ax.pcolormesh(G, Rm, Z, cmap="RdBu_r", vmin=-lim, vmax=lim,
                       shading="gouraud")
    cs = ax.contour(G, Rm, Z, levels=[0.0], colors=[INK], linewidths=2.6)
    ax.clabel(cs, fmt={0.0: T("반전 경계 ε=0", "reversal boundary ε=0")}, fontsize=8.6)
    for rho, rlab in zip(RHO_LEVELS, _rho_labels()):
        ax.axvline(rho, color=MUTED, lw=1.1, ls=":")
        ax.text(rho + 0.012, 0.03, rlab, ha="left", va="bottom", fontsize=7.6,
                color=MUTED, rotation=90)
    fig.colorbar(pm, ax=ax, pad=0.02, label="ε(σ_S, p)")
    ax.set_xlabel(T("부하율 (η) — 차량 100대 고정, 수요 스윕",
                    "load factor (η), m = 100 fixed, demand swept"))
    ax.set_ylabel(T("AT비율 r", "AT share r"))
    _panel_tag(ax, "a")
    _style_axes(ax)
    ax.grid(False)

    ax = axes[0, 1]
    sc = ax.scatter(sub.w_S, sub.eps_p_sigS, c=sub.r, cmap="viridis",
                    s=22, lw=0)
    ax.axhline(0.0, color=INK, lw=1.4)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1.0, 10.0], ["0.1", "1", "10"])
    fig.colorbar(sc, ax=ax, pad=0.02, label=T("AT비율 r", "AT share r"))
    ax.set_xlabel(T("조우 강도 w_S = q_eff·wt_max (σ_S = 1-exp(-w_S))",
                    "encounter intensity w_S = q_eff*wt_max (σ_S = 1-exp(-w_S))"))
    ax.set_ylabel("ε(σ_S, p)")
    _panel_tag(ax, "b")
    _style_axes(ax)

    P, Rm2 = np.meshgrid(p_grid, r_grid)
    for ci, rho in enumerate(PR_RHOS):
        ax = axes[1, ci]
        sub2 = df[df.panel == f"pr_map_{rho}"]
        Z2 = np.full((len(r_grid), len(p_grid)), np.nan)
        Z2[sub2.ir.to_numpy(dtype=int), sub2.ip.to_numpy(dtype=int)] = \
            sub2.eps_p_sigS.to_numpy()
        lim2 = max(np.nanpercentile(np.abs(Z2), 98), 1e-6)
        pm = ax.pcolormesh(P, Rm2, Z2, cmap="RdBu_r", vmin=-lim2, vmax=lim2,
                           shading="gouraud")
        has_flip = np.nanmin(Z2) < 0 < np.nanmax(Z2)
        if has_flip:
            cs = ax.contour(P, Rm2, Z2, levels=[0.0], colors=[INK],
                            linewidths=2.4)
            ax.clabel(cs, fmt={0.0: T("임계 r*", "critical r*")}, fontsize=8.4)
        fig.colorbar(pm, ax=ax, pad=0.02, label="ε(σ_S, p)")
        ax.set_xlabel(T("e비율 p", "e-hail share p"))
        ax.set_ylabel(T("AT비율 r", "AT share r"))
        _panel_tag(ax, "c" if ci == 0 else "d")
        _style_axes(ax)
        ax.grid(False)

    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_F3():
    """fig_F3 (see the paper for the definitions)."""
    rho_grid = np.linspace(0.15, 0.95, 17)
    specs = [
        ("wt_max", np.linspace(3 / 60, 60 / 60, 16), "sigS", 60.0, T("분", "min"),
         T("① wt_max(승객 인내) → street-hail 서비스율",
           "① wt_max (patience) → street-hail service probability σ_S")),
        ("te_max", np.geomspace(0.5 / 60, 60 / 60, 18), "sigE", 60.0, T("분", "min"),
         T("② te_max(배차 도달한도) → e-hail 서비스율 σ_E",
           "② te_max (dispatch limit) → e-hail service probability σ_E")),
        ("de_max", np.linspace(0.2, 2.6, 16), "sigS", 1.0, "km",
         T("③ de_max(허용 우회) → street-hail 서비스율 (흡수량 레버?)",
           "③ de_max (detour allowance) → street-hail service probability (absorption lever?)")),
        ("de_max", np.linspace(0.2, 2.6, 16), "delta", 1.0, "km",
         T("④ de_max(허용 우회) → 차내시간 팽창 (우회 비용 레버?)",
           "④ de_max (detour allowance) → in-vehicle time inflation (detour-cost lever?)")),
    ]
    rows = []
    for key, vals, ycol, scale, unit, _ in specs:
        for rho in rho_grid:
            for xv in vals:
                eps = elasticity(rho, key, **{key: float(xv)})
                rows.append(dict(param=key, out=ycol, rho=rho,
                                 x=xv * scale, eps=eps[ycol]))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.0))
    for ax, (key, vals, ycol, scale, unit, title) in zip(axes.ravel(), specs):
        sub = df[(df.param == key) & (df.out == ycol)]
        piv = sub.pivot_table(index="x", columns="rho", values="eps")
        X, Y = np.meshgrid(piv.columns.to_numpy(), piv.index.to_numpy())
        Z = np.abs(piv.to_numpy())
        vmax = max(np.nanpercentile(Z, 99), 0.06)
        pm = ax.pcolormesh(X, Y, Z, cmap="YlOrRd", vmin=0.0, vmax=vmax,
                           shading="gouraud")
        fig.colorbar(pm, ax=ax, pad=0.02, label=T("|탄력도|", "|elasticity|"))
        if np.nanmax(Z) > 0.05 > np.nanmin(Z):
            cs = ax.contour(X, Y, Z, levels=[0.05], colors=[CAT["blue"]],
                            linewidths=2.6)
            ax.clabel(cs, fmt={0.05: T("구속 전선 |ε|=0.05", "binding front |ε|=0.05")},
                      fontsize=8.4)
        base_v = BASE_PT[key] * scale
        ax.axhline(base_v, color=INK, lw=1.6, ls=(0, (5, 2)))
        ax.text(0.965, base_v, T(f" 기준 {base_v:.0f}{unit}", f" base {base_v:.0f} {unit}"),
                fontsize=8.6, color=INK,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
        if key == "te_max":
            ax.set_yscale("log")
            ax.set_yticks([0.5, 1, 2, 5, 10, 20, 60])
            ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "60"])
        ax.set_xlabel(T("부하율 ρ", "load factor ρ"))
        ax.set_ylabel(f"{_pname(key)} [{unit}]"
                      + (T(" · 로그축", " · log scale") if key == "te_max" else ""))
        note = (T("전 구간 여유(비구속)", "slack everywhere (non-binding)")
                if np.nanmax(Z) < 0.05
                else T("전선 아래가 구속 영역", "binding region below the front"))
        ax.set_title(T(f"{title}\n{note} — 최대 |ε| = {np.nanmax(Z):.3f}",
                       f"{title}\n{note} — max |ε| = {np.nanmax(Z):.3f}"),
                     fontsize=10.2, color=INK)
        _style_axes(ax)
        ax.grid(False)
    fig.suptitle(T("F3 · [민감도] 제약 구속 전선 — 한도 파라미터는 기울기가 아니라 '문턱'을 갖는다 "
                   f"(정원 3·e비율 0.6·AT비율 0.4·차량 {M_F:.0f}대)",
                   "F3 · [Sensitivity] Constraint-binding fronts — limit parameters have thresholds, not slopes "
                   f"(c=3, p=0.6, r=0.4, m={M_F:.0f})"),
                 fontsize=12.4, y=0.995, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def _g(w):
    """_g (see the paper for the definitions)."""
    w = np.asarray(w, dtype=float)
    out = np.where(w < 1e-8, 1.0, w * np.exp(-w) / (1.0 - np.exp(-w)))
    return out


def fig_F4():
    """fig_F4 (see the paper for the definitions)."""
    TE_ACT = 2.0 / 60.0
    rho_grid = np.linspace(0.15, 0.95, 17)
    h = 0.05
    rows = []
    for rho in rho_grid:
        res0 = _solve(rho)
        pr = res0["params"]
        wS = pr.wt_max * pr.v * res0["n_eff"] / pr.L
        cE = np.pi / (4 * pr.kappa**2)
        wE_base = cE * res0["n_E"] * (pr.v * pr.te_max) ** 2 / pr.R
        sp = _solve(rho, wt_max=pr.wt_max * (1 + h))
        sm = _solve(rho, wt_max=pr.wt_max * (1 - h))
        tot_S = (sp["sigma"] - sm["sigma"]) / res0["sigma"] / (2 * h)
        fb_S = (sp["n_eff"] - sm["n_eff"]) / res0["n_eff"] / (2 * h)
        ea = _solve(rho, te_max=TE_ACT)
        wE_act = cE * ea["n_E"] * (pr.v * TE_ACT) ** 2 / pr.R
        ep = _solve(rho, te_max=TE_ACT * (1 + h))
        em = _solve(rho, te_max=TE_ACT * (1 - h))
        tot_E = (ep["sigma_E"] - em["sigma_E"]) / ea["sigma_E"] / (2 * h)
        fb_E = (ep["n_E"] - em["n_E"]) / ea["n_E"] / (2 * h)
        rows.append(dict(rho=rho, wS=wS, wE_base=wE_base, wE_act=wE_act,
                         sigS=res0["sigma"], sigE_act=ea["sigma_E"],
                         dir_S=1.0 * float(_g(wS)), tot_S=tot_S, fb_S=fb_S,
                         dir_E=2.0 * float(_g(wE_act)), tot_E=tot_E, fb_E=fb_E))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.6))
    ax = axes[0]
    wg = np.geomspace(0.02, 3000.0, 600)
    ax.plot(wg, _g(wg), color=INK, lw=2.4,
            label=T(r"보편 곡선  $g(w)=w\,e^{-w}/(1-e^{-w})$",
                    r"universal curve  $g(w)=w\,e^{-w}/(1-e^{-w})$"))
    ax.scatter(df.wS, _g(df.wS), s=34, color=CAT["red"], zorder=5,
               label=T("street-hail 작동점 (기준 wt_max=60분)",
                       "street-hail operating points (base wt_max=60 min)"))
    ax.scatter(df.wE_act, _g(df.wE_act), s=34, color=CAT["blue"], zorder=5,
               label=T("e-hail 작동점 (te_max=2분, 구속 수준)",
                       "e-hail operating points (te_max=2 min, binding level)"))
    ax.scatter(df.wE_base, _g(df.wE_base), s=34, color=MUTED, zorder=5,
               marker="s", label=T("e-hail 작동점 (기준 te_max=60분)",
                                   "e-hail operating points (base te_max=60 min)"))
    ax.annotate(T("기준 te_max 는 완전 포화\n(g 거의 0 — 늘려도 아무 효과 없음)",
                  "base te_max is fully saturated\n(g near 0 — raising it does nothing)"),
                (df.wE_base.mean(), 0.06), textcoords="offset points",
                xytext=(-140, 40), fontsize=8.6, color=MUTED,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.1", "1", "10", "100", "1000"])
    ax.set_xlabel(T("무차원 강도 w  (로그축)", "dimensionless intensity w (log scale)"))
    ax.set_ylabel("ε(σ, w)")
    ax.set_ylim(0, 1.05)
    ax.set_title(T("① 보편 곡선 — 서비스율 탄력도는 w 하나로 결정\n"
                   "(w가 클수록 포화: 한도를 늘려도 소용없어진다)",
                   "① Universal curve — service-probability elasticity set by w alone\n"
                   "(larger w = saturation: raising the limit stops helping)"),
                 fontsize=10.2, color=INK)
    ax.legend(fontsize=7.8, frameon=False, loc="upper right")
    _style_axes(ax)
    ax = axes[1]
    ax.plot(df.rho, df.dir_S, color=CAT["red"], lw=2.2, ls=(0, (5, 2)),
            label=T("wt_max → σ_S : 직접  1·g(w_S)", "wt_max → σ_S: direct  1·g(w_S)"))
    ax.plot(df.rho, df.tot_S, "o-", color=CAT["red"], lw=2.2, ms=3.6,
            label=T("wt_max → σ_S : 총(되먹임 포함)", "wt_max → σ_S: total (with feedback)"))
    ax.plot(df.rho, df.dir_E, color=CAT["blue"], lw=2.2, ls=(0, (5, 2)),
            label=T("te_max → σ_E : 직접  2·g(w_E)   [te_max=2분]",
                    "te_max → σ_E: direct  2·g(w_E)  [te_max=2 min]"))
    ax.plot(df.rho, df.tot_E, "o-", color=CAT["blue"], lw=2.2, ms=3.6,
            label=T("te_max → σ_E : 총(되먹임 포함) [te_max=2분]",
                    "te_max → σ_E: total (with feedback) [te_max=2 min]"))
    ax.set_xlabel(T("부하율 ρ", "load factor ρ"))
    ax.set_ylabel(T("탄력도", "elasticity"))
    ax.set_title(T("② 직접 vs 총 탄력도 — 되먹임이 얼마나 깎는가\n"
                   "(파선=폐형식 예측, 실선=모형 수치)",
                   "② Direct vs total elasticity — how much feedback shaves off\n"
                   "(dashed = closed-form prediction, solid = model)"),
                 fontsize=10.2, color=INK)
    ax.legend(fontsize=7.8, frameon=False)
    _style_axes(ax)
    ax = axes[2]
    ax.plot(df.rho, df.tot_S / _g(df.wS), "o-", color=CAT["red"], lw=2.2, ms=3.6,
            label=T("street-hail : ε_total / g(w_S)", "street-hail: ε_total / g(w_S)"))
    ax.plot(df.rho, df.tot_E / _g(df.wE_act), "o-", color=CAT["blue"], lw=2.2, ms=3.6,
            label=T("e-hail : ε_total / g(w_E)  [te_max=2분]",
                    "e-hail: ε_total / g(w_E)  [te_max=2 min]"))
    ax.axhline(1.0, color=CAT["red"], lw=1.3, ls=(0, (4, 3)))
    ax.axhline(2.0, color=CAT["blue"], lw=1.3, ls=(0, (4, 3)))
    ax.text(0.16, 1.04, T("이론 지수 1 (wt_max 는 선형)", "theoretical exponent 1 (wt_max is linear)"),
            fontsize=8.4, color=CAT["red"])
    ax.text(0.16, 2.04, T("이론 지수 2 (te_max 는 제곱)", "theoretical exponent 2 (te_max is squared)"),
            fontsize=8.4, color=CAT["blue"])
    ax.set_xlabel(T("부하율 ρ", "load factor ρ"))
    ax.set_ylabel(T("총탄력도 ÷ g(w)", "total elasticity ÷ g(w)"))
    ax.set_ylim(0, 2.4)
    ax.set_title(T("③ 구조 지수의 검증 — te_max 는 wt_max 의 2배\n"
                   "(이론선 아래로 내려간 만큼이 되먹임 감쇠)",
                   "③ Verifying the structural exponents — te_max doubles wt_max\n"
                   "(shortfall below the theory line = feedback damping)"),
                 fontsize=10.2, color=INK)
    ax.legend(fontsize=8.0, frameon=False, loc="center right")
    _style_axes(ax)
    fig.suptitle(T(r"F4 · [민감도, 해석적 결과] 서비스율 탄력도의 폐형식 — $\sigma=1-e^{-w}$ 라는 공통 구조에서 나오는 것 "
                   f"(정원 3·e비율 0.6·AT비율 0.4·차량 {M_F:.0f}대)",
                   r"F4 · [Sensitivity, analytical] Closed-form service-probability elasticity — what follows from the shared structure $\sigma=1-e^{-w}$ "
                   f"(c=3, p=0.6, r=0.4, m={M_F:.0f})"),
                 fontsize=12.4, y=1.03, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def _eval_box(X, rho):
    """_eval_box (see the paper for the definitions)."""
    lo = np.array([PARAM_BOX[k][0] for k in PARAM_KEYS])
    hi = np.array([PARAM_BOX[k][1] for k in PARAM_KEYS])
    V = lo + X * (hi - lo)
    out = {k: np.empty(len(X)) for k, _ in OUTPUTS}
    nfail = 0
    for i, row in enumerate(V):
        over = {k: float(v) for k, v in zip(PARAM_KEYS, row)}
        res = _solve(rho, **over)
        if not res["converged"]:
            nfail += 1
        o = _outputs(res)
        for k in out:
            out[k][i] = o[k]
    return out, nfail


def _sobol(fA, fB, fAB):
    """_sobol (see the paper for the definitions)."""
    k, N = fAB.shape
    V = np.var(np.concatenate([fA, fB]), ddof=1)
    S = np.empty(k)
    ST = np.empty(k)
    for i in range(k):
        S[i] = np.mean(fB * (fAB[i] - fA)) / V
        ST[i] = 0.5 * np.mean((fA - fAB[i]) ** 2) / V
    return S, ST


def fig_F5(n_sobol=512, n_morris=40):
    """fig_F5 (see the paper for the definitions)."""
    from scipy.stats import qmc

    k = len(PARAM_KEYS)
    rows = []
    sob = {}
    mor = {}
    fails = {}
    for rho in RHO_LEVELS:
        eng = qmc.Sobol(d=2 * k, scramble=True, seed=20260815)
        base = eng.random(n_sobol)
        A, B = base[:, :k], base[:, k:]
        Xs = [A, B]
        for i in range(k):
            AB = A.copy()
            AB[:, i] = B[:, i]
            Xs.append(AB)
        allX = np.vstack(Xs)
        out, nfail = _eval_box(allX, rho)
        fails[rho] = nfail / len(allX) * 100
        sob[rho] = {}
        for ok, _ in OUTPUTS:
            f = out[ok]
            med = np.nanmedian(f)
            f = np.where(np.isfinite(f), f, med)
            fA, fB = f[:n_sobol], f[n_sobol:2 * n_sobol]
            fAB = f[2 * n_sobol:].reshape(k, n_sobol)
            S, ST = _sobol(fA, fB, fAB)
            sob[rho][ok] = (S, ST)
            for i, pk in enumerate(PARAM_KEYS):
                rows.append(dict(rho=rho, method="sobol", out=ok, param=pk,
                                 S1=S[i], ST=ST[i], inter=max(ST[i] - S[i], 0.0)))
        rng = np.random.default_rng(20260815)
        levels, delta = 4, 2.0 / 3.0
        ee = {ok: [[] for _ in range(k)] for ok, _ in OUTPUTS}
        traj_X, traj_meta = [], []
        for _ in range(n_morris):
            x = rng.integers(0, levels // 2 + 1, size=k) / (levels - 1)
            order = rng.permutation(k)
            pts = [x.copy()]
            for i in order:
                x = x.copy()
                x[i] = min(x[i] + delta, 1.0) if x[i] <= 0.5 else max(x[i] - delta, 0.0)
                pts.append(x.copy())
            traj_X.extend(pts)
            traj_meta.append(order)
        outm, _ = _eval_box(np.array(traj_X), rho)
        for ti, order in enumerate(traj_meta):
            b = ti * (k + 1)
            for step, i in enumerate(order):
                for ok, _ in OUTPUTS:
                    y0, y1 = outm[ok][b + step], outm[ok][b + step + 1]
                    if np.isfinite(y0) and np.isfinite(y1) and abs(y0) > 1e-12:
                        ee[ok][i].append((y1 - y0) / y0 / delta)
        mor[rho] = {ok: (np.array([np.mean(np.abs(v)) if v else np.nan
                                   for v in ee[ok]]),
                         np.array([np.std(v) if v else np.nan for v in ee[ok]]))
                    for ok, _ in OUTPUTS}
        for ok, _ in OUTPUTS:
            mu, sd = mor[rho][ok]
            for i, pk in enumerate(PARAM_KEYS):
                rows.append(dict(rho=rho, method="morris", out=ok, param=pk,
                                 mu_star=mu[i], sigma=sd[i]))
    df = pd.DataFrame(rows)

    labels = [_pname(k_) for k_ in PARAM_KEYS]
    x = np.arange(len(PARAM_KEYS))
    fig, axes = plt.subplots(3, 3, figsize=(17.0, 13.2))
    for ci, (rho, rlab) in enumerate(zip(RHO_LEVELS, _rho_labels())):
        for ri, ok in enumerate(("sigS", "unmet")):
            ax = axes[ri, ci]
            S, ST = sob[rho][ok]
            S = np.clip(S, 0, None)
            inter = np.clip(ST - S, 0, None)
            ax.bar(x, S, width=0.62, color=CAT["blue"],
                   label=T("1차 지수 S₁ (단독 기여)", "first-order index S₁ (solo contribution)"))
            ax.bar(x, inter, width=0.62, bottom=S, color=CAT["orange"],
                   label=T("교호작용 몫 (S_T 빼기 S₁)", "interaction share (S_T minus S₁)"))
            for xi, (s, it) in enumerate(zip(S, inter)):
                if s + it > 0.02:
                    ax.text(xi, s + it + 0.015, f"{s + it:.2f}", ha="center",
                            fontsize=8.0, color=INK2)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8.6, rotation=18, ha="right")
            ax.set_ylabel(T("분산 기여도", "variance contribution"))
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{'①②③'[ci] if ri == 0 else '④⑤⑥'[ci]} Sobol — "
                         f"{_oname(ok)} · {rlab}", fontsize=10.2, color=INK)
            _style_axes(ax)
        ax = axes[2, ci]
        mu, sd = mor[rho]["sigS"]
        mu_max = np.nanmax(mu) if np.isfinite(mu).any() else 1.0
        nsmall = 0
        for i, (m_, s_) in enumerate(zip(mu, sd)):
            ax.scatter([m_], [s_], s=90, color=OUT_COLORS[i % len(OUT_COLORS)],
                       zorder=5, edgecolor="white", lw=1.0)
            if m_ < 0.05 * mu_max:
                off = (10, 6 + 13 * nsmall)
                nsmall += 1
            else:
                off = (8, 4)
            ax.annotate(labels[i], (m_, s_), textcoords="offset points",
                        xytext=off, fontsize=8.4, color=INK2)
        top = max(np.nanmax(mu), np.nanmax(sd)) * 1.25 + 1e-9
        ax.plot([0, top], [0, top], color=MUTED, lw=1.1, ls=(0, (4, 3)))
        ax.text(top * 0.62, top * 0.72,
                T("σ = μ*  위쪽 = 비선형·교호작용 강함",
                  "σ = μ*  above: strong nonlinearity/interactions"),
                fontsize=8.0, color=MUTED, rotation=38)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel(T("μ*  (영향 크기)", "μ* (effect magnitude)"))
        ax.set_ylabel(T("σ  (효과의 산포)", "σ (effect dispersion)"))
        ax.set_title(T(f"{'⑦⑧⑨'[ci]} Morris 스크리닝 — σ_S · {rlab}",
                       f"{'⑦⑧⑨'[ci]} Morris screening — σ_S · {rlab}"),
                     fontsize=10.2, color=INK)
        _style_axes(ax)
    axes[0, 0].legend(fontsize=8.4, frameon=False, loc="upper left")
    fail_txt = " · ".join(f"ρ={r_:.2f}: {v:.2f}%" for r_, v in fails.items())
    fig.suptitle(T("F5 · [민감도] 전역 민감도 — 일변량으로는 보이지 않는 교호작용의 분산분해 "
                   f"(Sobol N={n_sobol}·Saltelli, Morris {n_morris}궤적, 정원 3·차량 {M_F:.0f}대)\n"
                   f"표본 상자: p∈[0.05,0.95] · r∈[0,0.9] · wt_max∈[5,60]분 · te_max∈[2,60]분 · de_max∈[0.25,2.5]km    "
                   f"|  비수렴 표본 비율 {fail_txt}",
                   "F5 · [Sensitivity] Global sensitivity — variance decomposition of interactions invisible to univariate sweeps "
                   f"(Sobol N={n_sobol}, Saltelli; Morris {n_morris} trajectories; c=3, m={M_F:.0f})\n"
                   f"sample box: p∈[0.05,0.95] · r∈[0,0.9] · wt_max∈[5,60] min · te_max∈[2,60] min · de_max∈[0.25,2.5] km    "
                   f"|  non-converged sample share {fail_txt}"),
                 fontsize=11.8, y=1.0, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_F6(n_lhs=100, n_mc=100):
    """fig_F6 (see the paper for the definitions)."""
    from scipy.stats import qmc

    k = len(PARAM_KEYS)
    lo = np.array([PARAM_BOX[p][0] for p in PARAM_KEYS])
    hi = np.array([PARAM_BOX[p][1] for p in PARAM_KEYS])
    sampler = qmc.LatinHypercube(d=k, seed=20260815)
    X = lo + sampler.random(n_lhs) * (hi - lo)
    Xm = lo + qmc.LatinHypercube(d=k, seed=77).random(n_mc) * (hi - lo)

    rows = []
    for rho in RHO_LEVELS:
        for row in X:
            over = {p: float(v) for p, v in zip(PARAM_KEYS, row)}
            p_hyb = over.pop("p")
            lamE = _lam_for(rho, M_F)
            for capa in (1, 2, 3, 5, 10):
                before = evaluate(BASE, **{**BASE_PT, **over, "capa": capa,
                                           "p": 0.999, "lam": lamE})
                after = evaluate(BASE, **{**BASE_PT, **over, "capa": capa,
                                          "p": p_hyb, "lam": lamE / p_hyb})
                if not (before["converged"] and after["converged"]):
                    continue
                if not (before["T_w_E"] and before["T_w_E"] > 0):
                    continue
                dT = (after["T_w_E"] - before["T_w_E"]) / before["T_w_E"] * 100
                rows.append(dict(claim="C23", rho=rho, capa=capa, p=p_hyb,
                                 r=over["r"], dTwE=dT))
            eps = elasticity(rho, "de_max", p=p_hyb, **over)
            rows.append(dict(claim="C4", rho=rho, p=p_hyb, r=over["r"],
                             eps_sigS=abs(eps["sigS"]), eps_delta=abs(eps["delta"])))
    for rho in RHO_LEVELS:
        lam = _lam_for(rho, M_F)
        for row in Xm:
            over = {p: float(v) for p, v in zip(PARAM_KEYS, row)}
            mcs = {}
            ok = True
            for capa in (1, 2, 10):
                try:
                    mcs[capa] = find_m_c(replace_base(**{**over, "capa": capa,
                                                         "lam": lam}),
                                         sigmaE_target=0.9, rtol=6e-3, m_hi=1200.0)
                except RuntimeError:
                    ok = False
                    break
            if not ok or mcs[1] - mcs[10] <= 1e-6:
                continue
            share = (mcs[1] - mcs[2]) / (mcs[1] - mcs[10])
            rows.append(dict(claim="C1", rho=rho, p=over["p"], r=over["r"],
                             share=share))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.0))
    # ── C1 ──
    ax = axes[0, 0]
    c1 = df[df.claim == "C1"]
    for rho, rlab, color in zip(RHO_LEVELS, _rho_labels(), RHO_COLORS):
        v = c1[c1.rho == rho].share.dropna().to_numpy()
        if len(v):
            ax.hist(np.clip(v, 0, 1.2), bins=np.linspace(0, 1.2, 37), alpha=0.55,
                    color=color, label=T(f"{rlab}  (중앙값 {np.median(v):.3f})",
                                         f"{rlab}  (median {np.median(v):.3f})"))
    ax.axvline(0.9, color=INK, lw=1.8, ls=(0, (5, 2)))
    ax.text(0.895, ax.get_ylim()[1] * 0.94, "0.9 ", fontsize=8.6, color=INK, ha="right")
    frac = (c1.share >= 0.9).mean() * 100 if len(c1) else np.nan
    ax.set_xlabel(T("(m_c(1) - m_c(2)) / (m_c(1) - m_c(10))   — 1이면 정원 2에서 이미 전부",
                    "(m_c(1) - m_c(2)) / (m_c(1) - m_c(10))   — 1 means capacity 2 already captures all"))
    ax.set_ylabel(T("표본 수", "number of samples"))
    ax.set_title(T(f"C1 · 정원의 가치는 1→2에서 전부인가\n"
                   f"0.9 이상인 표본 {frac:.1f}%  (n={len(c1)})",
                   f"C1 · Is the value of capacity all in 1→2?\n"
                   f"{frac:.1f}% of samples at 0.9 or above  (n={len(c1)})"),
                 fontsize=10.6, color=INK)
    ax.legend(fontsize=8.2, frameon=False)
    _style_axes(ax)
    # ── C2 ──
    ax = axes[0, 1]
    c23 = df[df.claim == "C23"]
    capas = [1, 2, 3, 5, 10]
    pos_frac = [(c23[c23.capa == c].dTwE > 0).mean() * 100 for c in capas]
    ax.bar(np.arange(len(capas)), pos_frac, width=0.6, color=CAT["red"])
    for i, v in enumerate(pos_frac):
        ax.text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=9.0, color=INK2)
    ax.axhline(100, color=INK2, lw=1.0, ls=(0, (4, 3)))
    ax.set_xticks(np.arange(len(capas)))
    ax.set_xticklabels([T(f"정원={c}", f"c={c}") for c in capas])
    ax.set_ylim(0, 112)
    ax.set_ylabel(T("Δ(e-hail 대기) > 0 인 표본 비율 [%]",
                    "share of samples with Δ(e-hail wait) > 0 [%]"))
    ax.set_title(T("C2 · street-hail 도입은 e-hail 승객에게 순부담인가\n"
                   "(100%면 상자 전체에서 예외 없이 성립)",
                   "C2 · Is street-hail introduction a net burden on e-hail riders?\n"
                   "(100% = holds without exception across the box)"),
                 fontsize=10.6, color=INK)
    _style_axes(ax)
    # ── C3 ──
    ax = axes[1, 0]
    for c, color in zip(capas, [CAT["red"], CAT["orange"], CAT["aqua"],
                                CAT["blue"], CAT["violet"]]):
        v = c23[c23.capa == c].dTwE.dropna().to_numpy()
        if len(v):
            ax.hist(np.clip(v, -2, 60), bins=np.linspace(-2, 60, 63), histtype="step",
                    lw=2.0, color=color,
                    label=T(f"정원={c} (중앙값 {np.median(v):+.2f}%)",
                            f"c={c} (median {np.median(v):+.2f}%)"))
    ax.axvline(3.0, color=INK, lw=1.8, ls=(0, (5, 2)))
    sub3 = c23[c23.capa >= 3]
    f3 = (sub3.dTwE <= 3.0).mean() * 100
    ax.set_xlabel(T("Δ(e-hail 대기) [%] — 도입 전 대비",
                    "Δ(e-hail wait) [%] — vs before introduction"))
    ax.set_ylabel(T("표본 수", "number of samples"))
    ax.set_yscale("log")
    ax.set_title(T(f"C3 · 정원 3 이상이면 부담이 3% 이하인가\n"
                   f"capa≥3 표본 중 {f3:.1f}% 가 3% 이하 (검은 파선=3%)",
                   f"C3 · Does capacity 3+ keep the burden at or below 3%?\n"
                   f"{f3:.1f}% of capa≥3 samples at or below 3% (black dashed = 3%)"),
                 fontsize=10.6, color=INK)
    ax.legend(fontsize=7.8, frameon=False)
    _style_axes(ax)
    # ── C4 ──
    ax = axes[1, 1]
    c4 = df[df.claim == "C4"].dropna(subset=["eps_sigS", "eps_delta"])
    ax.scatter(c4.eps_sigS, c4.eps_delta, s=14, alpha=0.45, color=CAT["violet"],
               edgecolor="none")
    top = max(c4.eps_delta.max(), c4.eps_sigS.max()) * 1.1
    ax.plot([0, top], [0, top], color=INK, lw=1.6, ls=(0, (5, 2)))
    ax.text(top * 0.52, top * 0.60, "y = x", fontsize=8.8, color=INK, rotation=41)
    win = (c4.eps_delta > c4.eps_sigS).mean() * 100
    ax.set_xlabel(T("|ε(σ_S, de_max)|  — 흡수량 쪽", "|ε(σ_S, de_max)|  — absorption side"))
    ax.set_ylabel(T("|ε(δ, de_max)|  — 우회 비용 쪽", "|ε(δ, de_max)|  — detour-cost side"))
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_title(T(f"C4 · de_max 는 우회 비용 레버인가\n"
                   f"우회 쪽 탄력도가 더 큰 표본 {win:.1f}%  (대각선 위쪽)",
                   f"C4 · Is de_max a detour-cost lever?\n"
                   f"detour-side elasticity larger in {win:.1f}% of samples (above the diagonal)"),
                 fontsize=10.6, color=INK)
    _style_axes(ax)
    fig.suptitle(T("F6 · [민감도] 결론의 강건성 — 논문의 주장을 파라미터 상자 전체에서 검증 "
                   f"(LHS {n_lhs}표본 × 체제 3수준, C1은 m_c 비용 때문에 {n_mc}표본)",
                   "F6 · [Sensitivity] Robustness of conclusions — the paper's claims tested across the parameter box "
                   f"(LHS {n_lhs} samples × 3 regimes; C1 uses {n_mc} samples due to m_c cost)"),
                 fontsize=12.4, y=0.995, color=INK)
    fig.tight_layout()
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_F7():
    capa_f7 = 5
    m_ref = 40.0
    de_grid = np.linspace(0.25, 2.5, 20)
    te_grid = np.linspace(1 / 60, 10 / 60, 19)
    wt_grid = np.linspace(5 / 60, 60 / 60, 20)

    rows = []
    for rho in RHO_LEVELS:
        for de in de_grid:
            res = _solve(rho, m=m_ref, de_max=float(de), capa=capa_f7)
            rows.append(dict(panel="de", rho=rho, x=de,
                             sigma_S=res["sigma"] or 0.0,
                             delta_pct=(res["delta_bar"] or 0.0) * 100))
        for te in te_grid:
            res = _solve(rho, m=m_ref, te_max=float(te), capa=capa_f7)
            rows.append(dict(panel="te", rho=rho, x=te * 60,
                             sigma_E=res["sigma_E"] or 0.0))
        for wt in wt_grid:
            res = _solve(rho, m=m_ref, wt_max=float(wt), capa=capa_f7)
            rows.append(dict(panel="wt", rho=rho, x=wt * 60,
                             sigma_S=res["sigma"] or 0.0))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.6))
    rho_labels = _rho_labels()

    def _plot_panel(ax, panel, ycol, letter):
        for rho, color, lab in zip(RHO_LEVELS, RHO_COLORS, rho_labels):
            sub = df[(df.panel == panel) & (df.rho == rho)].sort_values("x")
            ax.plot(sub.x, sub[ycol], color=color, lw=2.1, label=lab)
        _panel_tag(ax, letter, linespacing=1.7, labelpad=4)
        _style_axes(ax)

    _plot_panel(axes[0, 0], "de", "sigma_S", "a")
    axes[0, 0].set_xlabel(T("허용 우회 (d_e^max) [km]",
                            "detour allowance ($d_e^{max}$) [km]"))
    axes[0, 0].set_ylabel(T("street-hail 서비스율 (σ_S)",
                            "street-hail service probability ($\\sigma_S$)"))

    _plot_panel(axes[0, 1], "de", "delta_pct", "b")
    axes[0, 1].set_xlabel(T("허용 우회 (d_e^max) [km]",
                            "detour allowance ($d_e^{max}$) [km]"))
    axes[0, 1].set_ylabel(T("차내시간 증가율 [%]", "in-vehicle time increase [%]"))

    _plot_panel(axes[1, 0], "te", "sigma_E", "c")
    axes[1, 0].set_xlim(0, 10)
    axes[1, 0].set_xlabel(T("인내 한계 (t_E^max) [분]",
                            "patience limit ($t_E^{max}$) [min]"))
    axes[1, 0].set_ylabel(T("e-hail 서비스율 (σ_E)",
                            "e-hail service probability ($\\sigma_E$)"))

    _plot_panel(axes[1, 1], "wt", "sigma_S", "d")
    axes[1, 1].set_xlabel(T("인내 한계 (t_S^max) [분]",
                            "patience limit ($t_S^{max}$) [min]"))
    axes[1, 1].set_ylabel(T("street-hail 서비스율 (σ_S)",
                            "street-hail service probability ($\\sigma_S$)"))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, fontsize=9.5,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1.0))
    fig.subplots_adjust(hspace=0.5, wspace=0.32)
    return df, fig


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def fig_F8():
    lam = _lam_for(0.6, 40.0)
    targets = np.linspace(0.0, 0.90, 19)
    pts = [(0.95, 0.0), (0.6, 0.4), (0.95, 0.9)]
    pt_colors = [CAT["red"], CAT["aqua"], CAT["blue"]]

    def _feasible(capa, p, r, m, tgt):
        res = evaluate(BASE, capa=capa, m=float(m), p=p, r=r, lam=lam)
        return (not _collapsed(res)) and (res["sigma"] or 0.0) >= tgt

    def _m_min(capa, p, r, tgt):
        lo, hi = 2.0, 60.0
        tries = 0
        while not _feasible(capa, p, r, hi, tgt):
            hi *= 2.0
            tries += 1
            if tries > 8:
                return np.nan
        while hi - lo > 3e-3 * hi:
            mid = 0.5 * (lo + hi)
            if _feasible(capa, p, r, mid, tgt):
                hi = mid
            else:
                lo = mid
        return hi

    rows = []
    for p, r in pts:
        for tgt in targets:
            m1 = _m_min(1, p, r, float(tgt))
            m5 = _m_min(5, p, r, float(tgt))
            gap = ((m1 - m5) / m1 * 100.0
                   if np.isfinite(m1) and np.isfinite(m5) else np.nan)
            rows.append(dict(p=p, r=r, target=tgt, m1=m1, m5=m5, gap_pct=gap))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for (p, r), color in zip(pts, pt_colors):
        sub = df[(df.p == p) & (df.r == r)].sort_values("target")
        ax.plot(sub.target, sub.gap_pct, color=color, lw=2.1,
                label=T(f"p = {p:g}, r = {r:g}", f"$p$ = {p:g}, $r$ = {r:g}"))
    ax.set_xlabel(T("street-hail 서비스율 (σₛ)",
                    "street-hail service probability ($\\sigma_S$)"))
    ax.set_ylabel(T("정원 5의 최소 fleet 절감률 [%]",
                    "minimum-fleet saving from capa = 5 [%]"))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9.2)
    _style_axes(ax)
    fig.tight_layout()
    return df, fig


FIGS_F = {"F0": fig_F0, "F1": fig_F1, "F2": fig_F2, "F3": fig_F3,
          "F4": fig_F4, "F5": fig_F5, "F6": fig_F6, "F7": fig_F7,
          "F8": fig_F8}


if __name__ == "__main__":
    import time
    argv = [a for a in sys.argv[1:] if a != "--en"]
    if "--en" in sys.argv[1:]:
        set_lang("en")
    target = argv[0] if argv else "all"
    todo = list(FIGS_F) if target == "all" else [target]
    for fid in todo:
        t = time.perf_counter()
        df, fig = FIGS_F[fid]()
        _save(fig, fid, df)
        print(f"{fid}  ({time.perf_counter() - t:.1f}s)")
