import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / "figures"))
"""State-wise catchment probabilities of pooled routes (Section 4.2, Table 4).

For every state (i, j) with i + j <= 9 (vehicles with up to ten seats) and five detour allowances pi2,
routes are sampled on the rectilinear unit square (4,000 routes and 3,000 street-hail users per route)
and the share of users that can be inserted within the detour allowance is recorded.
Output: outputs/tables/mc_catchment_states_by_pi2.csv
"""
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
PI2S = (0.05, 0.10, 1.0 / np.sqrt(33.04), 0.25, 0.35)   # 0.174 = case study (d_e^max = 1 km, R = 33.04 km2)
STATES = [(i, j) for i in range(10) for j in range(10 - i) if i + j > 0]


def task(args):
    pi2, i, j = args
    import mc_catchment as mc
    mc.RNG = np.random.default_rng(int(1e6 * pi2) * 1000 + 10 * i + j)
    return dict(pi2=pi2, i=i, j=j, mc=mc.mc_state(i, j, pi2, n_route=4000, n_od=3000))


def main():
    from drtmodel.model import catchment_EN
    tasks = [(pi2, i, j) for pi2 in PI2S for (i, j) in STATES]
    tasks.sort(key=lambda t: -(t[1] + 2 * t[2]))      # longest routes first
    with Pool(os.cpu_count()) as pool:
        rows = pool.map(task, tasks, chunksize=1)
    df = pd.DataFrame(rows)
    df["formula"] = df.pi2.map(catchment_EN)
    df.sort_values(["pi2", "i", "j"]).to_csv(OUT / "mc_catchment_states_by_pi2.csv", index=False)
    print(len(df), "rows")


if __name__ == "__main__":
    main()
