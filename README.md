# drt-mixed-hailing-model

Analytical model of a pooling-enabled demand-responsive transit (DRT) service that serves both
**e-hailing** and **street-hailing** users with a **heterogeneous fleet** of autonomous vehicles (AVs),
which serve e-hail requests only, and human-driven vehicles (HVs), which serve both. It accompanies the paper

> H. Park, S. Kang, *An analytical model of demand-responsive transit with mixed hailing and a heterogeneous
> fleet* (submitted to Transportation Research Part B).

## Model

Each vehicle is in a workload state (i, j): i passengers on board and j assigned but not yet picked up,
with i + j ≤ capa. Vehicles move between states through four links:

| Link | Transition | Who |
|---|---|---|
| e-hail assignment | (i, j) → (i, j+1) | AVs and HVs with a spare seat |
| pickup | (i, j) → (i+1, j−1) | vehicles with j ≥ 1 |
| drop-off | (i, j) → (i−1, j) | vehicles with i ≥ 1 |
| street-hail absorption | (i, j) → (i+1, j) | HVs with a spare seat (idle HVs fully, en-route HVs through a catchment probability) |

Users balk: an e-hail request is abandoned if the nearest eligible vehicle cannot arrive within
`te_max`, and a street-hail user leaves after `wt_max`. The steady state is the solution of the flow
balance of every state, found by a damped fixed-point iteration on the e-hail-eligible pool n^E and the
effective street-hail availability n^S. Outputs include the service probability, waiting time, and unmet
demand of each hailing mode, the mean in-vehicle time (Little's law), and a generalized social cost.

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Usage

```python
from drtmodel import HybridDRTModel, HybridDRTParams

pr = HybridDRTParams(m=40, r=0.4, p=0.6, lam=5.0, R=33.04, v=33.0, capa=5,
                     Delta1=0.586, Delta2=0.586, wt_max=1.0, te_max=1.0, de_max=1.0)
model = HybridDRTModel(pr)
res = model.solve()
print(res["sigma_E"], res["sigma"], res["T_w_E"], res["T_w_S"], res["ride_time"])
print("social cost per trip [h]:", model.social_cost(res))
```

Units are km and hours (speed in km/h, demand density in trips per km² per hour).
`HybridDRTParams(av_street_idle=True)` lets idle AVs also pick up street-hail users.

## Reproducing the paper

```bash
bash reproduce_all.sh
```

| Script | Paper |
|---|---|
| `figures/make_m1_diagram.py` | Fig. 1 |
| `figures/analysis_figures.py D3/D0/D1/D2/D4/E3/E4/E4B/E6 --en` | Figs. 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `figures/sensitivity_figures.py F7/F8/F6 --en` | Figs. 11, 12 and Table 6 |
| `scripts/mc_catchment.py` | Table 4 (Monte Carlo check of the catchment probability) |
| `scripts/uniqueness_check.py` | Section 3.4 (existence and uniqueness of the equilibrium) |
| `scripts/sensitivity_rstar.py` | Table 5 (sensitivity of the cost-minimizing AV share) |

Outputs are written to `outputs/`. The Dongtan case-study parameters are defined as `BASE` in
`figures/analysis_figures.py`. Figure labels are available in Korean (default) and English (`--en`).

## License

MIT — see `LICENSE`.
