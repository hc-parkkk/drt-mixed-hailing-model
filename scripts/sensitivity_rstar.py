import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / "figures"))
"""Sensitivity of the cost-minimizing AV share (Section 6.2, Table 5).

Each parameter is set to 0.5 and 1.5 times its reference value, one at a time, with the load factor held at 0.6.

Output: outputs/tables/sensitivity_rstar.csv
"""
from pathlib import Path
import dataclasses

import numpy as np
import pandas as pd

from drtmodel.model import HybridDRTModel, HybridDRTParams
from analysis_figures import BASE

OUT = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
R_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
M_REF, ETA_REF, CAPA_REF, P_REF = 40.0, 0.6, 5, 0.6


def scan(label, value, alpha=20.0, beta=30.0, vot_ratio=1.0, **over):
    """scan (see the paper for the definitions)."""
    base = dataclasses.replace(BASE, **{"m": M_REF, "capa": CAPA_REF, "p": P_REF, **over})
    lam = ETA_REF * base.v * base.m / (base.R * base.l)
    zs, sS = [], []
    for r in R_GRID:
        pr = dataclasses.replace(base, r=float(r), lam=lam)
        mdl = HybridDRTModel(pr)
        res = mdl.solve()
        if res["converged"]:
            z = mdl.social_cost(res, alpha, beta)
            demand = pr.lam * pr.R
            fleet = (pr.r * pr.m * alpha / beta + (1 - pr.r) * pr.m * (alpha + beta) / beta) / demand
            zs.append(fleet + vot_ratio * (z - fleet))
        else:
            zs.append(np.nan)
        sS.append(res["sigma"] if res["converged"] else np.nan)
    zs = np.array(zs)
    k = int(np.nanargmin(zs))
    return dict(parameter=label, value=value, r_star=R_GRID[k], z_star_h=zs[k],
                z_r0_h=zs[0], z_r1_h=zs[-1], sigmaS_r0=sS[0], sigmaS_rstar=sS[k],
                saving_pct=(1 - zs[k] / zs[0]) * 100)


def main():
    rows = [scan("reference", "-")]
    for a in (10.0, 30.0):
        rows.append(scan("alpha_v [$/h] (beta = 30)", a, alpha=a))
    for b in (15.0, 45.0):
        rows.append(scan("beta [$/h] (alpha_v = 20)", b, beta=b))
    for g in (0.5, 1.5):
        rows.append(scan("value of time / driver wage", g, vot_ratio=g))
    for v in (16.5, 49.5):
        rows.append(scan("v [km/h]", v, v=v))
    for R in (0.5 * BASE.R, 1.5 * BASE.R):
        rows.append(scan("R [km2] (l = kappa sqrt R)", R, R=R))
    for t in (30.0, 90.0):
        rows.append(scan("t_E^max = t_S^max [min]", t, te_max=t / 60, wt_max=t / 60))
    for d in (0.5, 1.5):
        rows.append(scan("d_e^max [km]", d, de_max=d))
    for p in (0.3, 0.9):
        rows.append(scan("p", p, p=p))
    df = pd.DataFrame(rows)
    for c in ("z_star_h", "z_r0_h", "z_r1_h"):
        df[c.replace("_h", "_min")] = df[c] * 60
    df.to_csv(OUT / "sensitivity_rstar.csv", index=False)
    print(df[["parameter", "value", "r_star", "z_star_min", "z_r0_min", "z_r1_min",
              "saving_pct", "sigmaS_r0", "sigmaS_rstar"]].round(3).to_string())


if __name__ == "__main__":
    main()
