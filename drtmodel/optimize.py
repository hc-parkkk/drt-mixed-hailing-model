"""Evaluation helpers: solve a configuration with its social cost, and the minimum-fleet search of Eq. (23)."""

import itertools
from dataclasses import replace

from .model import HybridDRTModel, HybridDRTParams


def evaluate(base: HybridDRTParams, alpha_veh=20.0, beta=30.0, use_detour=True, **overrides):
    """Solve `base` with some parameters overridden; return the results with 'delta_bar' and 'z_beta'."""
    pr = replace(base, **overrides) if overrides else base
    mdl = HybridDRTModel(pr)
    res = mdl.solve()
    delta_bar = mdl.detour_inflation(res) if use_detour else 0.0
    res["delta_bar"] = delta_bar
    res["z_beta"] = mdl.social_cost(res, alpha_veh, beta, delta_bar)
    res["params"] = pr
    return res


def is_feasible(res, TwE_target=None, TwS_target=None, sigmaE_target=None, sigmaS_target=None):
    """Converged, non-negative state counts, and optional service-level targets."""
    ok = res["converged"] and res["min_state"] > -1e-8
    if ok and TwE_target is not None and res["T_w_E"] is not None:
        ok = res["T_w_E"] <= TwE_target
    if ok and TwS_target is not None and res["T_w_S"] is not None:
        ok = res["T_w_S"] <= TwS_target
    if ok and sigmaE_target is not None and res["sigma_E"] is not None:
        ok = res["sigma_E"] >= sigmaE_target
    if ok and sigmaS_target is not None and res["sigma"] is not None:
        ok = res["sigma"] >= sigmaS_target
    return ok


def find_m_c(base: HybridDRTParams, TwE_target=None, TwS_target=None, sigmaE_target=None,
             sigmaS_target=None, rtol=1e-3, m_hi=None):
    """Minimum fleet meeting the targets, by bisection on m (Eq. 23).

    Lower bound: the single-occupancy workload lambda R l / v. Upper bound: `m_hi`, or four times the lower
    bound doubled until feasible.
    """
    lo = base.lam * base.R * base.l / base.v
    hi = m_hi if m_hi is not None else max(4.0 * lo, 10.0)

    def feasible(m):
        return is_feasible(evaluate(base, m=m), TwE_target, TwS_target, sigmaE_target, sigmaS_target)

    expand = 0
    while not feasible(hi):
        hi *= 2.0
        expand += 1
        if expand > 12:
            raise RuntimeError("could not find a feasible fleet; the targets appear unattainable")
    while hi - lo > rtol * hi:
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi


def sweep(base: HybridDRTParams, name: str, values, alpha_veh=20.0, beta=30.0):
    """Sweep one parameter; return a list of dicts with the main outputs."""
    rows = []
    for val in values:
        res = evaluate(base, alpha_veh=alpha_veh, beta=beta, **{name: val})
        rows.append({name: val, "converged": res["converged"], "T_w_E": res["T_w_E"], "T_w_S": res["T_w_S"],
                     "sigma": res["sigma"], "unmet": res["unmet"], "c": res["c"], "n_eff": res["n_eff"],
                     "delta_bar": res["delta_bar"], "z_beta": res["z_beta"]})
    return rows


def optimize_policy(base: HybridDRTParams, grids: dict, alpha_veh=20.0, beta=30.0):
    """Grid search for the combination minimizing z/beta; returns (best_overrides, best_result, rows)."""
    names = list(grids)
    best, best_res, rows = None, None, []
    for combo in itertools.product(*(grids[n] for n in names)):
        ov = dict(zip(names, combo))
        res = evaluate(base, alpha_veh=alpha_veh, beta=beta, **ov)
        rows.append({**ov, "z_beta": res["z_beta"], "converged": res["converged"],
                     "T_w_E": res["T_w_E"], "T_w_S": res["T_w_S"]})
        if res["converged"] and (best is None or res["z_beta"] < best_res["z_beta"]):
            best, best_res = ov, res
    return best, best_res, rows
