"""Analytical model of pooling-enabled DRT with mixed hailing and a heterogeneous fleet.

Equation numbers refer to the paper.

States:  n_ij^F, F in {AV, HV}: i passengers on board, j assigned and not yet picked up, i + j <= capa
Links:   assignment  (i,j) -> (i,j+1)   sigma_E p lambda R n_ij / n^E            eligible states, Eq. (6)
         pickup      (i,j) -> (i+1,j-1) n_ij sqrt(n^E) v / (kappa sqrt(R))      j >= 1, Eq. (7a)
         drop-off    (i,j) -> (i-1,j)   n_ij sqrt(i) v / (kappa sqrt(R))        i >= 1, Eq. (7b)
         absorption  (i,j) -> (i+1,j)   sigma_S (1-p) lambda R w_ij n_ij^HV / n^S   HVs, i + j < capa, Eq. (13)
Solution: for fixed (n^E, n^S) the flow balance of each fleet type is linear; the equilibrium is found by a
          damped fixed-point iteration on (n^E, n^S) (Section 2.5).

In the code, AV/HV are called AT/HT (autonomous / human-driven taxi) and n^S is called n_eff.
"""

from dataclasses import dataclass

import numpy as np


def catchment_EN(pi2: float) -> float:
    """Catchment probability of an en-route HV, Eq. (9).

    Expectation of the polynomial N(X, Y) of Ouyang, Yang and Daganzo (2021) when the rectilinear path
    components X and Y follow the triangular density 2(1 - x) on [0, 1]; valid for pi2 << 1.
    """
    return 1.0 / 144.0 + pi2 / 12.0 + 13.0 * pi2**2 / 72.0 + pi2**3 / 18.0


def expected_insertion_detour(pi2: float) -> float:
    """Expected insertion detour per successful absorption, in units of sqrt(R), Eq. (20).

    E[L'N] = (pi2^2/48)(2 + 52 pi2/9 + 2 pi2^2) and delta_ins = E[L'N] / E[N]; delta_ins < pi2 always.
    """
    ELpN = pi2**2 / 48.0 * (2.0 + 52.0 * pi2 / 9.0 + 2.0 * pi2**2)
    return ELpN / catchment_EN(pi2)


@dataclass
class HybridDRTParams:
    """Model parameters. Units: km and h (speed km/h, demand trips/(km^2 h))."""
    m: float            # fleet size
    r: float            # AV share of the fleet
    p: float            # e-hail share of demand
    lam: float          # demand density lambda
    R: float            # service area [km^2]
    v: float            # vehicle speed [km/h]
    capa: int           # seat capacity
    kappa: float = 0.63     # nearest-neighbor constant of a grid network
    Delta1: float = 0.35    # grid spacings [km]
    Delta2: float = 0.35
    wt_max: float = 5.0 / 60.0   # street-hail patience limit t_S^max [h]
    de_max: float = 1.0          # detour allowance d_e^max [km]
    availability_rule: str = "a"   # e-hail eligibility: "a" any state with i + j < capa (DRT),
    #                                 "b" only i = 0, j < capa (shared-taxi form of Daganzo and Ouyang, 2019)
    flexible_dropoff: bool = True  # True: drop-off allowed while pickups remain; False: only when j = 0
    balking: bool = True           # True: unserved users leave (service probabilities sigma_E, sigma_S);
    #                                 False: all demand served (limit used for the special-case checks)
    te_max: float = 10.0 / 60.0    # e-hail patience limit t_E^max (maximum pickup travel time) [h]
    av_street_idle: bool = False   # alternative scenario: idle AVs also pick up street-hail users (weight 1)

    @property
    def L(self) -> float:
        """Total road length of the grid network."""
        return 2.0 * self.R * (self.Delta1 + self.Delta2) / (self.Delta1 * self.Delta2)

    @property
    def pi2(self) -> float:
        """Catchment ratio d_e^max / sqrt(R)."""
        return self.de_max / np.sqrt(self.R)

    @property
    def l(self) -> float:
        """Mean trip length kappa sqrt(R)."""
        return self.kappa * np.sqrt(self.R)


class HybridDRTModel:
    def __init__(self, params: HybridDRTParams):
        self.pr = params
        capa = params.capa
        self.nodes = [(i, j) for i in range(capa + 1) for j in range(capa + 1 - i)]
        self.idx = {node: k for k, node in enumerate(self.nodes)}
        self.N = len(self.nodes)
        if params.availability_rule == "a":
            self.eligible = {(i, j) for (i, j) in self.nodes if i + j < capa}
        elif params.availability_rule == "b":
            self.eligible = {(i, j) for (i, j) in self.nodes if i == 0 and j < capa}
        else:
            raise ValueError("availability_rule must be 'a' or 'b'")
        # street-hail weights w_ij, Eq. (10): 1 for idle HVs, E[N] for en-route HVs with spare seats
        EN = catchment_EN(params.pi2)
        self.w = {}
        for (i, j) in self.nodes:
            if i + j < capa:
                self.w[(i, j)] = 1.0 if (i, j) == (0, 0) else EN

    # -- service probabilities -------------------------------------------------------------------
    def _sigma(self, s2: float) -> float:
        """Street-hail service probability sigma_S, Eq. (12); s2 = n^S."""
        return 1.0 - np.exp(-self.pr.wt_max * self.pr.v * s2 / self.pr.L)

    def _sigma_E(self, s1: float) -> float:
        """E-hail service probability sigma_E, Eq. (5); s1 = n^E."""
        pr = self.pr
        c = np.pi / (4.0 * pr.kappa**2)
        return 1.0 - np.exp(-c * s1 * (pr.v * pr.te_max) ** 2 / pr.R)

    def _coeffs(self, s1: float, s2: float, is_ht: bool):
        """Transition rates per vehicle in each state for fixed (n^E, n^S)."""
        pr = self.pr
        eh_flow = pr.p * pr.lam * pr.R
        if pr.balking:
            eh_flow *= self._sigma_E(s1)
        k_a = eh_flow / s1
        k_p = pr.v * np.sqrt(s1) / (pr.kappa * np.sqrt(pr.R))
        vk = pr.v / (pr.kappa * np.sqrt(pr.R))
        sh_flow = (1.0 - pr.p) * pr.lam * pr.R
        if pr.balking:
            sh_flow *= self._sigma(s2)
        out = {}
        for (i, j) in self.nodes:
            links = []  # (target state, rate per vehicle)
            if (i, j) in self.eligible:
                links.append(((i, j + 1), k_a))
            if j >= 1:
                links.append(((i + 1, j - 1), k_p))
            if i >= 1 and (pr.flexible_dropoff or j == 0):
                links.append(((i - 1, j), vk * np.sqrt(i)))
            if is_ht and (i, j) in self.w and pr.p < 1.0:
                links.append(((i + 1, j), sh_flow * self.w[(i, j)] / s2))
            elif (not is_ht) and pr.av_street_idle and (i, j) == (0, 0) and pr.p < 1.0:
                links.append(((1, 0), sh_flow * 1.0 / s2))
            out[(i, j)] = links
        return out

    def _solve_fleet(self, fleet_size: float, is_ht: bool, s1: float, s2: float):
        """Linear flow balance (Eq. 14) of one fleet type with its fleet constraint (Eq. 1)."""
        if fleet_size <= 0.0:
            return np.zeros(self.N)
        A = np.zeros((self.N, self.N))
        links = self._coeffs(s1, s2, is_ht)
        for (i, j), lst in links.items():
            src = self.idx[(i, j)]
            for target, coeff in lst:
                A[src, src] -= coeff
                A[self.idx[target], src] += coeff
        A[0, :] = 1.0          # one balance equation is redundant: replace it by the fleet constraint
        b = np.zeros(self.N)
        b[0] = fleet_size
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def solve(self, s1_init=None, s2_init=None, damping=0.5, tol=1e-13, max_iter=5000):
        """Damped fixed-point iteration on (n^E, n^S) (Section 2.5).

        The default start (n^E = 0.8 m) selects the efficient equilibrium; a nearly empty start can reach
        the inefficient one at heavy load (Section 4.3).
        """
        pr = self.pr
        m_at, m_ht = pr.r * pr.m, (1.0 - pr.r) * pr.m
        eps = 1e-12 * max(pr.m, 1.0)
        s1 = float(s1_init) if s1_init is not None else 0.8 * pr.m
        s2 = float(s2_init) if s2_init is not None else max(0.5 * m_ht, eps)
        converged = False
        for it in range(max_iter):
            xa = self._solve_fleet(m_at, False, s1, s2)
            xh = self._solve_fleet(m_ht, True, s1, s2)
            s1n = sum(xa[self.idx[n]] + xh[self.idx[n]] for n in self.eligible)
            s2n = sum(self.w[n] * xh[self.idx[n]] for n in self.w)
            if pr.av_street_idle:
                s2n += xa[self.idx[(0, 0)]]
            s1n, s2n = max(s1n, eps), max(s2n, eps)
            if abs(s1n - s1) <= tol * max(s1, 1.0) and abs(s2n - s2) <= tol * max(s2, 1.0):
                s1, s2, converged = s1n, s2n, True
                break
            s1 = (1.0 - damping) * s1 + damping * s1n
            s2 = (1.0 - damping) * s2 + damping * s2n
        xa = self._solve_fleet(m_at, False, s1, s2)
        xh = self._solve_fleet(m_ht, True, s1, s2)
        return self._results(xa, xh, s1, s2, converged, it + 1)

    # -- performance measures (Section 2.6) --------------------------------------------------------
    def _results(self, xa, xh, s1, s2, converged, iters):
        pr = self.pr
        neg = min(xa.min() if len(xa) else 0.0, xh.min() if len(xh) else 0.0)
        if neg < -1e-8 * max(pr.m, 1.0):
            converged = False  # negative state counts: infeasible configuration
        n_at = {n: float(xa[self.idx[n]]) for n in self.nodes}
        n_ht = {n: float(xh[self.idx[n]]) for n in self.nodes}
        j_sum = sum(j * (n_at[(i, j)] + n_ht[(i, j)]) for (i, j) in self.nodes)
        i_sum = sum(i * (n_at[(i, j)] + n_ht[(i, j)]) for (i, j) in self.nodes)
        demand = pr.lam * pr.R
        res = {"converged": converged, "iterations": iters, "n_AT": n_at, "n_HT": n_ht,
               "n_E": s1, "n_eff": s2, "min_state": float(neg),
               "residual": self._balance_residual(xa, xh, s1, s2)}
        if pr.p > 0:
            res["sigma_E"] = self._sigma_E(s1) if pr.balking else 1.0
            res["eh_served"] = res["sigma_E"] * pr.p * demand
            res["unmet_E"] = (1.0 - res["sigma_E"]) * pr.p * demand
            res["T_w_E"] = j_sum / res["eh_served"]                      # Eq. (15)
        else:
            res["sigma_E"], res["eh_served"], res["unmet_E"] = None, 0.0, 0.0
            res["T_w_E"] = None
        if pr.p < 1.0:
            q = pr.v * s2 / pr.L                                         # encounter rate q^S, Eq. (11)
            res["T_w_S_uncond"] = 1.0 / q
            res["sigma"] = self._sigma(s2)
            res["unmet"] = (1.0 - res["sigma"]) * (1.0 - pr.p) * demand
            res["sh_served"] = (res["sigma"] if pr.balking else 1.0) * (1.0 - pr.p) * demand
            if pr.balking:
                # waiting time of served users, Eq. (17): E[W | W <= t] = 1/q - t e^(-qt) / (1 - e^(-qt))
                qt = q * pr.wt_max
                if qt > 1e-8:
                    res["T_w_S"] = 1.0 / q - pr.wt_max * np.exp(-qt) / (-np.expm1(-qt))
                else:
                    res["T_w_S"] = pr.wt_max / 2.0
            else:
                res["T_w_S"] = res["T_w_S_uncond"]
            res["T_w_S_avg"] = res["sigma"] * res["T_w_S_uncond"]      # per arrival, Eq. (18)
        else:
            res["T_w_S"], res["sigma"], res["unmet"] = None, None, 0.0
            res["sh_served"], res["T_w_S_avg"], res["T_w_S_uncond"] = 0.0, None, None
        res["unmet_total"] = res["unmet_E"] + res["unmet"]
        res["c"] = (sum(n_at[n] for n in self.eligible) / s1) if s1 > 0 else None  # AV share of eligible pool
        res["onboard"] = i_sum
        res["implied_ride_time"] = i_sum / demand
        served = res["eh_served"] + res["sh_served"]
        res["ride_time"] = (i_sum / served) if served > 0 else pr.l / pr.v       # Eq. (16)
        vk = pr.v / (pr.kappa * np.sqrt(pr.R))
        res["dropoff_flow"] = sum(vk * np.sqrt(i) * (n_at[(i, j)] + n_ht[(i, j)])
                                  for (i, j) in self.nodes if i >= 1 and (pr.flexible_dropoff or j == 0))
        return res

    def _balance_residual(self, xa, xh, s1, s2):
        """Largest absolute flow-balance residual at the final (n^E, n^S), relative to lambda R."""
        pr = self.pr
        worst = 0.0
        for x, is_ht in ((xa, False), (xh, True)):
            if x.sum() <= 0:
                continue
            links = self._coeffs(s1, s2, is_ht)
            net = np.zeros(self.N)
            for (i, j), lst in links.items():
                src = self.idx[(i, j)]
                for target, coeff in lst:
                    flow = coeff * x[src]
                    net[src] -= flow
                    net[self.idx[target]] += flow
            worst = max(worst, float(np.abs(net).max()))
        return worst / (pr.lam * pr.R)

    def detour_inflation(self, res) -> float:
        """In-vehicle time added by absorption detours, relative to l/v, Eq. (21)."""
        pr = self.pr
        if pr.p >= 1.0 or res["n_eff"] <= 0:
            return 0.0
        mid = sum(self.w[n] * res["n_HT"][n] for n in self.w if n != (0, 0))
        phi_mid = mid / res["n_eff"]
        d_ins = expected_insertion_detour(pr.pi2) * np.sqrt(pr.R)
        served_frac = res["sigma"] if pr.balking else 1.0
        return served_frac * (1.0 - pr.p) * phi_mid * d_ins / pr.l

    def social_cost(self, res, alpha_veh=20.0, beta=30.0, delta_bar=0.0, unmet_penalty=0.0,
                    ride="little", ehail_unmet="patience"):
        """Generalized social cost per trip request z/beta [h], Eq. (22).

        Waiting: e-hail p [sigma_E T_w^E + (1 - sigma_E) t_E^max]; street-hail (1 - p) E[min(W, t_S^max)].
        In-vehicle time: ride="little" uses Eq. (16) (default); ride="detour" uses (l/v)(1 + delta_bar).
        ehail_unmet="none" drops the charge for abandoned e-hail requests.
        """
        pr = self.pr
        demand = pr.lam * pr.R
        z = (pr.r * pr.m / demand) * (alpha_veh / beta) \
            + ((1.0 - pr.r) * pr.m / demand) * ((alpha_veh + beta) / beta)
        if pr.p > 0 and res["T_w_E"] is not None:
            sE = res["sigma_E"] if res["sigma_E"] is not None else 1.0
            z += pr.p * sE * res["T_w_E"]
            if pr.balking and ehail_unmet == "patience":
                z += pr.p * (1.0 - sE) * pr.te_max
        if pr.p < 1.0 and res["T_w_S"] is not None:
            z += (1.0 - pr.p) * (res["T_w_S_avg"] if pr.balking else res["T_w_S"])
        z += res["ride_time"] if ride == "little" else (pr.l / pr.v) * (1.0 + delta_bar)
        z += unmet_penalty * res["unmet_total"] / demand
        return z
