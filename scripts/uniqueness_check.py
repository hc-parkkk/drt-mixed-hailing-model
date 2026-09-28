import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / "figures"))
"""Multiple-start search for equilibria of the fixed-point iteration (Section 3.4).

Output: outputs/tables/uniqueness_check.csv
"""
from pathlib import Path
import dataclasses
import itertools

import numpy as np
import pandas as pd

from drtmodel.model import HybridDRTModel
from analysis_figures import BASE

OUT = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    R, v, l = BASE.R, BASE.v, BASE.l
    rows = []
    grid = itertools.product((1, 2, 3, 5, 10), (0.2, 0.6, 1.0), (0.0, 0.4, 0.8),
                             (0.2, 0.5, 0.8, 1.1, 1.4), (20.0, 40.0, 100.0))
    for capa, p, r, eta, m in grid:
        lam = eta * v * m / (R * l)
        pr = dataclasses.replace(BASE, m=m, capa=capa, p=p, r=r, lam=lam)
        mdl = HybridDRTModel(pr)
        sols = []
        n_conv = 0
        for f1 in (0.02, 0.2, 0.5, 0.9, 1.0):
            for f2 in (0.02, 0.3, 1.0):
                res = mdl.solve(s1_init=f1 * m, s2_init=max(f2 * (1 - r) * m, 1e-6))
                if not res["converged"] or res["min_state"] < -1e-8:
                    continue
                n_conv += 1
                key = np.array([res["n_E"], res["n_eff"]])
                if not any(np.allclose(key, s, rtol=1e-6, atol=1e-9) for s in sols):
                    sols.append(key)
        rows.append(dict(capa=capa, p=p, r=r, eta=eta, m=m, n_starts=15,
                         n_converged=n_conv, n_distinct=len(sols)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "uniqueness_check.csv", index=False)
    print(len(df), "configurations")
    print("distinct-solution counts:", df.n_distinct.value_counts().to_dict())
    print("configs with <15 converged starts:", int((df.n_converged < 15).sum()))
    print(df[df.n_distinct != 1].to_string())


if __name__ == "__main__":
    main()
