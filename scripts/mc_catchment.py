import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / "figures"))
"""Monte Carlo check of the catchment probability, Eq. (9) (Section 3.3, Table 4).

Single-leg paths test Eq. (9) itself; pooled routes of state (i, j) give state-dependent absorption
probabilities, which are then substituted into the model to measure the effect on its outputs.
Output: outputs/tables/mc_catchment.csv, mc_catchment_model.csv
"""
from pathlib import Path
import dataclasses

import numpy as np
import pandas as pd

from drtmodel.model import HybridDRTModel, catchment_EN

OUT = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260929)


def l1(a, b):
    return np.abs(a[..., 0] - b[..., 0]) + np.abs(a[..., 1] - b[..., 1])


def min_insertion(route, O, D):
    """min_insertion (see the paper for the definitions)."""
    k = len(route)
    n = len(O)
    best = np.full(n, np.inf)
    for a in range(k):
        pa = route[a]
        na = route[a + 1] if a + 1 < k else None
        seg = l1(pa, O) + l1(O, D)
        if na is not None:
            seg = seg + l1(D, na) - l1(pa, na)
        best = np.minimum(best, seg)
        o_add = l1(pa, O) + (l1(O, na) - l1(pa, na) if na is not None else 0.0)
        for b in range(a + 1, k):
            pb = route[b]
            nb = route[b + 1] if b + 1 < k else None
            d_add = l1(pb, D) + (l1(D, nb) - l1(pb, nb) if nb is not None else 0.0)
            best = np.minimum(best, o_add + d_add)
    return best


def greedy_route(start, drops, picks):
    """greedy_route (see the paper for the definitions)."""
    pos = start
    route = [start]
    pending_drop = [d for d in drops]
    pending_pick = [p for p in picks]
    while pending_drop or pending_pick:
        cands = [("d", idx, d) for idx, d in enumerate(pending_drop)] + \
                [("p", idx, p[0]) for idx, p in enumerate(pending_pick)]
        kind, idx, pt = min(cands, key=lambda c: abs(c[2][0] - pos[0]) + abs(c[2][1] - pos[1]))
        route.append(pt)
        pos = pt
        if kind == "d":
            pending_drop.pop(idx)
        else:
            o, d = pending_pick.pop(idx)
            pending_drop.append(d)
    return np.array(route)


def mc_single_leg(pi2, n_route=4000, n_od=4000):
    """mc_single_leg (see the paper for the definitions)."""
    hits = 0.0
    for _ in range(n_route):
        A, B = RNG.random(2), RNG.random(2)
        O, D = RNG.random((n_od, 2)), RNG.random((n_od, 2))
        hits += np.mean(min_insertion(np.array([A, B]), O, D) <= pi2)
    return hits / n_route


def mc_state(i, j, pi2, n_route=1500, n_od=3000):
    """mc_state (see the paper for the definitions)."""
    if i + j == 0:
        return np.nan
    hits = 0.0
    for _ in range(n_route):
        start = RNG.random(2)
        drops = [RNG.random(2) for _ in range(i)]
        picks = [(RNG.random(2), RNG.random(2)) for _ in range(j)]
        route = greedy_route(start, drops, picks)
        O, D = RNG.random((n_od, 2)), RNG.random((n_od, 2))
        hits += np.mean(min_insertion(route, O, D) <= pi2)
    return hits / n_route


def main():
    from analysis_figures import BASE
    pi2_ref = BASE.pi2
    rows = []
    for pi2 in (0.05, 0.1, pi2_ref, 0.25, 0.35):
        emp = mc_single_leg(pi2)
        rows.append(dict(kind="single_leg", i=np.nan, j=np.nan, pi2=pi2, mc=emp,
                         formula=catchment_EN(pi2)))
        print(f"single leg π₂={pi2:.3f}: MC={emp:.4f}  식9={catchment_EN(pi2):.4f}", flush=True)
    for i in range(0, 5):
        for j in range(0, 5 - i):
            if i + j == 0:
                continue
            emp = mc_state(i, j, pi2_ref)
            rows.append(dict(kind="state", i=i, j=j, pi2=pi2_ref, mc=emp,
                             formula=catchment_EN(pi2_ref)))
            print(f"state ({i},{j}): MC={emp:.4f}  식9={catchment_EN(pi2_ref):.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mc_catchment.csv", index=False)

    wmc = {(int(r.i), int(r.j)): r.mc for r in df[df.kind == "state"].itertuples()}
    R, v, l = BASE.R, BASE.v, BASE.l
    out = []
    for capa in (2, 3, 5):
        for eta in (0.3, 0.6, 0.9):
            for p, r in ((0.6, 0.4), (0.3, 0.0)):
                lam = eta * v * 40.0 / (R * l)
                pr = dataclasses.replace(BASE, m=40.0, capa=capa, p=p, r=r, lam=lam)
                base_m = HybridDRTModel(pr)
                a = base_m.solve()
                alt = HybridDRTModel(pr)
                for n in alt.w:
                    if n != (0, 0):
                        alt.w[n] = wmc[n]
                b = alt.solve()
                out.append(dict(capa=capa, eta=eta, p=p, r=r,
                                sigma_S=a["sigma"], sigma_S_mc=b["sigma"],
                                TwS_avg=a["T_w_S_avg"] * 60, TwS_avg_mc=b["T_w_S_avg"] * 60,
                                TwE=a["T_w_E"] * 60, TwE_mc=b["T_w_E"] * 60,
                                ride=a["ride_time"] * 60, ride_mc=b["ride_time"] * 60))
    dm = pd.DataFrame(out)
    dm["dsigma_S_pp"] = (dm.sigma_S_mc - dm.sigma_S) * 100
    dm.to_csv(OUT / "mc_catchment_model.csv", index=False)
    print(dm.round(3).to_string())


if __name__ == "__main__":
    main()
