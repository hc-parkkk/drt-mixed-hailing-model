import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / "figures"))
"""Verification grid of Section 4: catchment approximation (4.2, Table 4) and uniqueness (4.3, Fig. 4).

Cases: capacities 2, 3, 5, 10; load factors 0.1-1.5; e-hail and AV shares 0-1 (steps of 0.1); fleets of
20 to 100 vehicles. Area, grid spacing, speed, and patience limits are the case-study values (BASE).
4.2: for each detour allowance pi2, the state-wise sampled probabilities (mc_states_extended.py) replace
     E[N(pi2)] in Eq. (10); the weighted ratio and the change in sigma_S and W_S are recorded.
4.3: the model is solved from 36 initial guesses (n^E and n^S at 2, 20, 40, 60, 80, 100%) and the
     distinct solutions are recorded with the number of guesses reaching each.

Usage: python verification_grid.py [uniqueness_only | catchment_only]
Output: outputs/tables/verif_catchment.csv, verif_uniqueness_detail.csv
"""
import dataclasses
import itertools
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
CAPAS = (2, 3, 5, 10)
LOADS = tuple(np.round(np.arange(0.1, 1.501, 0.1), 2))
SHARES = tuple(np.round(np.arange(0.0, 1.001, 0.1), 2))
FLEETS = (20.0, 40.0, 60.0, 80.0, 100.0)
STARTS = (0.02, 0.2, 0.4, 0.6, 0.8, 1.0)   # initial guesses: n^E as share of fleet, n^S as share of HVs


def _setup():
    from analysis_figures import BASE
    from drtmodel.model import HybridDRTModel, catchment_EN
    return BASE, HybridDRTModel, catchment_EN


def catchment_task(args):
    capa, eta, m = args
    BASE, Model, catchment_EN = _setup()
    st = pd.read_csv(OUT / "mc_catchment_states_by_pi2.csv")
    R, v, l = BASE.R, BASE.v, BASE.l
    lam = eta * v * m / (R * l)
    rows = []
    for pi2 in sorted(st.pi2.unique()):
        s = st[(st.pi2 - pi2).abs() < 1e-12]
        pmc = {(int(r.i), int(r.j)): r.mc for r in s.itertuples()}
        EN = catchment_EN(pi2)
        for p in SHARES:
            for r in SHARES:
                if p >= 1.0 or r >= 1.0:
                    continue
                pr = dataclasses.replace(BASE, m=m, capa=capa, p=float(p), r=float(r), lam=float(lam),
                                         de_max=float(pi2 * np.sqrt(R)))
                base = Model(pr)
                a = base.solve()
                row = dict(pi2=pi2, capa=capa, eta=eta, m=m, p=p, r=r)
                # efficient solution only: converged, no negative state, eligible pool not empty
                if (not a["converged"]) or a["n_E"] < 1e-3 * m or a["n_eff"] <= 1e-9:
                    row.update(ok=False)
                    rows.append(row)
                    continue
                enroute = [n for n in base.w if n != (0, 0)]
                nh = {n: a["n_HT"][n] for n in enroute}
                tot = sum(nh.values())
                pbar = sum(pmc[n] * nh[n] for n in enroute) / tot if tot > 1e-12 else np.nan
                alt = Model(pr)
                for n in enroute:
                    alt.w[n] = pmc[n]
                b = alt.solve()
                row.update(ok=bool(b["converged"]), ratio=pbar / EN, pbar=pbar, EN=EN,
                           idle_share_nS=a["n_HT"][(0, 0)] / a["n_eff"],
                           dsigma_pp=(b["sigma"] - a["sigma"]) * 100,
                           dWbar_min=(b["T_w_S_avg"] - a["T_w_S_avg"]) * 60)
                rows.append(row)
    return rows


def uniqueness_task(args):
    capa, eta, m = args
    BASE, Model, _ = _setup()
    R, v, l = BASE.R, BASE.v, BASE.l
    lam = eta * v * m / (R * l)
    rows = []
    for p in SHARES:
        for r in SHARES:
            pr = dataclasses.replace(BASE, m=m, capa=capa, p=float(p), r=float(r), lam=float(lam))
            mdl = Model(pr)
            sols = []
            n_conv = 0
            for f1 in STARTS:
                for f2 in STARTS:
                    res = mdl.solve(s1_init=f1 * m, s2_init=max(f2 * (1 - r) * m, 1e-6))
                    if not res["converged"] or res["min_state"] < -1e-8:
                        continue
                    n_conv += 1
                    key = np.array([res["n_E"], res["n_eff"]])
                    hit = [s for s in sols if np.allclose(key, s["key"], rtol=1e-6, atol=1e-9)]
                    if hit:
                        hit[0]["n"] += 1; hit[0]["f1s"].add(f1)
                    else:
                        sols.append(dict(key=key, f1=f1, n=1, f1s={f1}, sigma_E=res["sigma_E"],
                                         TwE=(res["T_w_E"] or np.nan) * 60))
            sols.sort(key=lambda s: -s["key"][0])
            # which solution the default initial guess (used for all reported results) reaches
            dflt = mdl.solve()
            dkey = np.array([dflt["n_E"], dflt["n_eff"]])
            dflt_is_top = bool(sols) and dflt["converged"] and np.allclose(dkey, sols[0]["key"], rtol=1e-6, atol=1e-9)
            row = dict(capa=capa, eta=eta, m=m, p=p, r=r, n_converged=n_conv, n_distinct=len(sols),
                       top_nE_frac=(sols[0]["key"][0] / m) if sols else np.nan,
                       top_sigma_E=sols[0]["sigma_E"] if sols else np.nan,
                       top_TwE_min=sols[0]["TwE"] if sols else np.nan,
                       top_n_starts=sols[0]["n"] if sols else 0,
                       top_f1_min=min(sols[0]["f1s"]) if sols else np.nan,
                       default_reaches_top=dflt_is_top)
            if len(sols) >= 2:
                lo = sols[-1]
                row.update(second_nE_frac=lo["key"][0] / m, second_sigma_E=lo["sigma_E"],
                           second_TwE_min=lo["TwE"], second_from_f1=lo["f1"],
                           second_n_starts=lo["n"], second_f1_max=max(lo["f1s"]))
            rows.append(row)
    return rows


def main():
    tasks = list(itertools.product(CAPAS, LOADS, FLEETS))
    with Pool(os.cpu_count()) as pool:
        if "catchment_only" not in sys.argv:
            uq = [r for chunk in pool.map(uniqueness_task, tasks) for r in chunk]
            pd.DataFrame(uq).to_csv(OUT / "verif_uniqueness_detail.csv", index=False)
            print("uniqueness done", len(uq), flush=True)
        if "uniqueness_only" not in sys.argv:
            ct = [r for chunk in pool.map(catchment_task, tasks) for r in chunk]
            pd.DataFrame(ct).to_csv(OUT / "verif_catchment.csv", index=False)
            print("catchment done", len(ct), flush=True)


if __name__ == "__main__":
    main()
