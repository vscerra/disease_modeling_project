"""
===========================================================
fitting.py
Author: Veronica Scerra
Last Updated: 2025-10-11
===========================================================

Description:
    Estimate SIR parameters (beta, gamma) from time series by:
      1) coarse grid search
      2) local pattern-search refinement
    Supports fitting to either I(t) (prevalence) or daily incidence.

Example Usage:
    from epimodels.fitting import fit_beta_gamma
    result = fit_beta_gamma(
        t, y_obs, N=N, I0=I0, R0_init=0,
        observable="I",
        beta_bounds=(0.05, 0.7),
        gamma_bounds=(0.05, 0.5)
    )

Notes:
    - No SciPy; uses RK4 integrator.
    - Observations should align with t (same length).
    - Consider smoothing noisy incidence before fitting (moving average).
-----------------------------------------------------------
License: MIT
===========================================================
"""

from typing import Dict, Tuple, Literal, Callable
import numpy as np
import pandas as pd
from .sir import SIRModel

ObsType = Literal["I", "incidence"]

def _simulate_observable(
        N: int,
        beta: float,
        gamma: float,
        I0: int,
        R0_init: int, 
        t: np.ndarray,
        observable: ObsType
) -> np.ndarray:
    model = SIRModel(N=N, beta=beta, gamma=gamma)
    out = model.simulate(t=t, I0=I0, R0_init=R0_init)
    if observable == "I":
        return out["I"]
    elif observable == "incidence":
        return out["incidence"]
    else:
        raise ValueError("observable must be 'I' or 'incidence'")
    

def _mse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is not None:
        return float(np.mean(weights * (y_true - y_pred) **2))
    return float(np.mean((y_true - y_pred) ** 2))


def _bounded(val: float, lo: float, hi: float) -> float:
    return min(max(val, lo), hi)


def _pattern_search(
        start: Tuple[float, float],
        loss_fn: Callable[[float, float], float],
        bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        init_step: Tuple[float, float] = (0.05, 0.05),
        shrink: float = 0.05,
        min_step: float = 1e-3,
        max_iter: int = 300
) -> Tuple[float, float, float]: 
    """Simple coordinate pattern search (Nelder-Mead-like approach, but simpler)"""
    beta, gamma = start
    step_b, step_g = init_step
    best = loss_fn(beta, gamma)

    for _ in range(max_iter):
        improved = False
        for db, dg in [(+step_b, 0), (-step_b, 0), (0, +step_g), (0, -step_g)]:
            nb = _bounded(beta + db, bounds[0][0], bounds[0][1])
            ng = _bounded(gamma + dg, bounds[1][0], bounds[1][1])
            val = loss_fn(nb, ng)
            if val + 1e-12 < best:
                beta, gamma, best = nb, ng, val
                improved = True

            if not improved:
                # shrink steps
                step_b *= shrink
                step_g *= shrink
                if step_b < min_step and step_g < min_step:
                    break
    return beta, gamma, best


def fit_beta_gamma(
        t: np.ndarray,
        y_obs: np.ndarray,
        N: int,
        I0: int, 
        R0_init: int = 0,
        observable: ObsType = "I",
        beta_bounds: Tuple[float, float] = (0.05, 0.6),
        gamma_bounds: Tuple[float, float] = (0.05, 0.5),
        grid_size: Tuple[int, int] = (15, 15),
        weights: np.ndarray | None = None, 
        refine: bool = True,
        init_step: Tuple[float, float] = (0.05, 0.05),
        random_restarts: int = 2, 
        seed: int | None = 42
) -> Dict[str, float | np.ndarray]:
    """
    Estimate beta and gamma by minimizing MSE between observed and simulated series.

    Args:
        t, y_obs: time grid and observation vector (same length).
        observable: "I" (prevalence) or "incidence".
        weights: optional per-time weights (e.g., weight peak more, or downweight noisy tail).
        grid_size: coarse grid over bounds; increase for more exhaustive search.
        random_restarts: extra local refinements from random seeds inside bounds.

    Returns:
        dict with beta, gamma, R0, loss, y_fit, and simulation dict (t,S,I,R,incidence).
    """
    assert len(t) == len(y_obs), "t and y_obs must have equal lengths"
    if weights is not None:
        assert len(weights) == len(y_obs), "weights must match y length"

    # coarse grid
    b_vals = np.linspace(beta_bounds[0], beta_bounds[1], grid_size[0])
    g_vals = np.linspace(gamma_bounds[0], gamma_bounds[1], grid_size[1])

    def loss(beta: float, gamma: float) -> float:
        y_hat = _simulate_observable(N, beta, gamma, I0, R0_init, t, observable)
        # Optional scale aligning: if user observes *fractions*, but sim is counts (or vice versa),
        # scale y_hat to best match y_obs in least-squares sense.
        # This single scalar 'a' handles unit mismatch robustly.
        a = float(np.dot(y_obs, y_hat) / (np.dot(y_hat, y_hat) + 1e-12))
        y_hat_scaled = a * y_hat
        return _mse(y_obs, y_hat_scaled, weights)
    
    best = {"beta": None, "gamma": None, "loss": np.inf}
    for b in b_vals:
        for g in g_vals:
            val = loss(b, g)
            if val < best["loss"]:
                best = {"beta": float(b), "gamma": float(g), "loss": float(val)}

    # Local refinement
    rng = np.random.default_rng(seed)
    starts = [(best["beta"], best["gamma"])]
    if random_restarts > 0:
        for _ in range(random_restarts):
            starts.append((
                float(rng.uniform(*beta_bounds)),
                float(rng.uniform(*gamma_bounds)),
            ))

    if refine:
        for s in starts:
            b_opt, g_opt, l_opt = _pattern_search(
                s, loss_fn=loss,
                bounds=(beta_bounds, gamma_bounds),
                init_step=init_step
            )
            if l_opt < best["loss"]:
                best = {"beta": float(b_opt), "gamma": float(g_opt), "loss": float(l_opt)}

    # final simulate and scaling
    y_fit_raw = _simulate_observable(N, best["beta"], best["gamma"], I0, R0_init, t, observable)
    scale = float(np.dot(y_obs, y_fit_raw) / (np.dot(y_fit_raw, y_fit_raw) + 1e-12))
    y_fit = scale * y_fit_raw

    # full trajectories (so I can plot S, I, R)
    model = SIRModel(N=N, beta=best["beta"], gamma=best["gamma"])
    sim = model.simulate(t=t, I0=I0, R0_init=R0_init)

    return {
         "beta": best["beta"],
        "gamma": best["gamma"],
        "R0": best["beta"] / best["gamma"] if best["gamma"] > 0 else np.inf,
        "loss": best["loss"],
        "scale": scale,
        "y_fit": y_fit,
        "sim": sim,
    }


def fit_from_dataframe(df: pd.DataFrame, time_col: str, value_col: str, **kwargs):
    """To load data from a CSV with columns t, incidence (or I)"""
    t = df[time_col].to_numpy(dtype=float)
    y = df[value_col].to_numpy(dtype=float)
    return fit_beta_gamma(t, y, **kwargs)