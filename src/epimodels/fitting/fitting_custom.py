"""
===========================================================
fitting_custom.py
Author: Veronica Scerra
Last Updated: 2025-10-11
===========================================================

Description:
    Estimate SIR and SEIR parameters (beta, gamma) from time series by:
      1) coarse grid search
      2) local pattern-search refinement
    Supports fitting to either I(t) (prevalence) or daily incidence.
      3) Using scipy least_squares fitting

Example Usage:
    from epimodels.fitting import fit_beta_gamma
    result = fit_beta_gamma(
        t, y_obs, N=N, I0=I0, R0_init=0,
        observable="I",
        beta_bounds=(0.05, 0.7),
        gamma_bounds=(0.05, 0.5)
    )

Notes:
    - No SciPy usage in the .seir_custom fitting; uses RK4 integrator 
    - Observations should align with t (same length).
    - Consider smoothing noisy incidence before fitting (moving average).
-----------------------------------------------------------
License: MIT
===========================================================
"""

from typing import Dict, Tuple, Literal, Callable, Sequence
import numpy as np
import pandas as pd
from .sir_custom import SIRModel
from .sir_piecewise import SIRPiecewiseBeta
from .seir_custom import SEIRModel
from .seir_scipy_fit import SEIRParams, simulate_seir
from __future__ import annotations
from dataclasses import dataclass
from scipy.optimize import least_squares



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


## Added for piecewise beta fitting

def _simulate_observable_piecewise(
    N: int,
    gamma: float,
    edges: np.ndarray,
    betas: np.ndarray,
    I0: int,
    R0_init: int,
    t: np.ndarray,
    observable: ObsType
) -> np.ndarray:
    model = SIRPiecewiseBeta(N=N, gamma=gamma, edges=edges, betas=betas)
    out = model.simulate(t=t, I0=I0, R0_init=R0_init)
    return out["I"] if observable == "I" else out["incidence"]

def _loss_with_scale(y_obs: np.ndarray, y_hat: np.ndarray, weights: np.ndarray | None) -> float:
    a = float(np.dot(y_obs, y_hat) / (np.dot(y_hat, y_hat) + 1e-12))  # scale align
    return _mse(y_obs, a * y_hat, weights)

def fit_piecewise_beta(
    t: np.ndarray,
    y_obs: np.ndarray,
    N: int,
    I0: int,
    edges: Sequence[float],
    R0_init: int = 0,
    observable: ObsType = "incidence",
    beta_bounds: Tuple[float, float] = (0.05, 1.5),
    gamma_bounds: Tuple[float, float] = (1/21, 1/3),
    init_betas: Sequence[float] | None = None,
    init_gamma: float | None = None,
    weights: np.ndarray | None = None,
    lambda_smooth: float = 0.0,       # L2 penalty on adjacent β differences
    grid_gamma: int = 10,             # coarse grid over γ, then refine
    step_beta: float = 0.03,
    step_gamma: float = 0.01,
    shrink: float = 0.5,
    min_step: float = 1e-3,
    max_iter: int = 300,
    seed: int | None = 7
) -> Dict[str, float | np.ndarray]:
    """
    Fit piecewise β(t) SIR by minimizing MSE (with scale) + optional smoothness penalty.

    Args:
        edges: segment boundaries in same units as t (len = K+1).
        init_betas/gamma: optional initial guesses; otherwise mid-bounds.
        lambda_smooth: weight of L2 penalty on (β_{k+1}-β_k)^2 to discourage overfitting.

    Returns:
        dict with betas, gamma, R0_segment (betas/gamma), loss, scale, y_fit, sim
    """
    assert len(t) == len(y_obs)
    edges = np.asarray(edges, dtype=float)
    K = len(edges) - 1
    rng = np.random.default_rng(seed)

    # inits
    b_lo, b_hi = beta_bounds; g_lo, g_hi = gamma_bounds
    betas = np.full(K, 0.5*(b_lo + b_hi), dtype=float) if init_betas is None else np.asarray(init_betas, dtype=float)
    gamma = float(0.5*(g_lo + g_hi) if init_gamma is None else init_gamma)

    # loss wrapper
    def model_loss(betas_vec: np.ndarray, gamma_val: float) -> float:
        y_hat = _simulate_observable_piecewise(N, gamma_val, edges, betas_vec, I0, R0_init, t, observable)
        data_fit = _loss_with_scale(y_obs, y_hat, weights)
        if lambda_smooth > 0 and len(betas_vec) > 1:
            smooth = np.sum(np.diff(betas_vec)**2)
            return float(data_fit + lambda_smooth * smooth)
        return float(data_fit)

    # coarse grid over gamma only
    g_vals = np.linspace(g_lo, g_hi, grid_gamma)
    best = {"betas": betas.copy(), "gamma": gamma, "loss": np.inf}
    for g in g_vals:
        val = model_loss(betas, g)
        if val < best["loss"]:
            best = {"betas": betas.copy(), "gamma": float(g), "loss": float(val)}

    # coordinate pattern search over (betas, gamma)
    betas = best["betas"].copy()
    gamma = best["gamma"]
    step_b = np.full(K, step_beta, dtype=float)
    step_g = float(step_gamma)

    for _ in range(max_iter):
        improved = False

        # try adjusting each beta_k up/down
        for k in range(K):
            for d in (+step_b[k], -step_b[k]):
                trial = betas.copy()
                trial[k] = float(np.clip(trial[k] + d, b_lo, b_hi))
                val = model_loss(trial, gamma)
                if val + 1e-12 < best["loss"]:
                    best = {"betas": trial.copy(), "gamma": gamma, "loss": float(val)}
                    betas = trial
                    improved = True

        # try adjusting gamma up/down
        for d in (+step_g, -step_g):
            g_try = float(np.clip(gamma + d, g_lo, g_hi))
            val = model_loss(betas, g_try)
            if val + 1e-12 < best["loss"]:
                best = {"betas": betas.copy(), "gamma": g_try, "loss": float(val)}
                gamma = g_try
                improved = True

        if not improved:
            step_b *= shrink
            step_g *= shrink
            if np.all(step_b < min_step) and step_g < min_step:
                break

    # final simulate + scale and return
    y_hat = _simulate_observable_piecewise(N, best["gamma"], edges, best["betas"], I0, R0_init, t, observable)
    scale = float(np.dot(y_obs, y_hat) / (np.dot(y_hat, y_hat) + 1e-12))
    model = SIRPiecewiseBeta(N=N, gamma=best["gamma"], edges=edges, betas=best["betas"])
    sim = model.simulate(t=t, I0=I0, R0_init=R0_init)

    return {
        "betas": best["betas"],
        "gamma": best["gamma"],
        "R0_segments": best["betas"] / best["gamma"] if best["gamma"] > 0 else np.full(K, np.inf),
        "loss": best["loss"],
        "scale": scale,
        "y_fit": scale * y_hat,
        "sim": sim,
        "edges": edges,
    }


## Added for SEIR 

def fit_seir_custom(
    t: np.ndarray,
    y_obs: np.ndarray,
    N: int,
    I0: int, 
    E0: int = 0,
    R0_init: int = 0,
    observable: ObsType = "incidence",
    beta_bounds: Tuple[float, float]  = (0.05, 1.2),
    gamma_bounds: Tuple[float, float] = (1/14, 1/5),   # infectious period ~5–14d
    sigma_bounds: Tuple[float, float] = (1/7, 1/2),    # incubation ~2–7d
    grid_size: Tuple[int, int, int] = (10, 10, 8),
    weights: np.ndarray | None = None,
    refine: bool = True,
    init_steps: Tuple[float, float, float] = (0.03, 0.01, 0.01),
    random_restarts: int = 2,
    seed: int | None = 42) -> Dict[str, float | np.ndarray]:
    """
    Estimate (beta, gamma, sigma) for SEIR by minimizing MSE between observed and simulated series
    with an automatically learned global scale factor.
    """
    assert len(t) == len(y_obs)
    rng = np.random.default_rng(seed)

    def simulate(beta, gamma, sigma):
        m = SEIRModel(N=N, beta=beta, gamma=gamma, sigma=sigma)
        out = m.simulate(t=t, I0=I0, E0=E0, R0_init=R0_init)
        return out["I"] if observable=="I" else out["incidence"]
    
    def loss(beta, gamma, sigma):
        y_hat = simulate(beta, gamma, sigma)
        a = float(np.dot(y_obs, y_hat) / (np.dot(y_hat, y_hat) + 1e-12))
        return _mse(y_obs, a*y_hat, weights)
    
    # coarse grid
    b_vals = np.linspace(*beta_bounds, grid_size[0])
    g_vals = np.linspace(*gamma_bounds, grid_size[1])
    s_vals = np.linspace(*sigma_bounds, grid_size[2])

    best = {"beta": None, "gamma": None, "sigma": None, "loss": np.inf}
    for b in b_vals:
        for g in g_vals:
            for s in s_vals:
                val = loss(b,g,s)
                if val < best["loss"]:
                    best = {"beta": float(b), "gamma": float(g), "sigma": float(s), "loss": float(val)}

    # local pattern search (coordinate-wise)
    step_b, step_g, step_s = init_steps 
    def clip(x, lo, hi): return float(min(max(x, lo), hi))

    def try_dir(b, g, s, db, dg, ds):
        nb = clip(b + db, *beta_bounds)
        ng = clip(g + dg, *gamma_bounds)
        ns = clip(s + ds, *sigma_bounds)
        val = loss(nb, ng, ns)
        return nb, ng, ns, val

    if refine:
        b, g, s = best["beta"], best["gamma"], best["sigma"]
        while step_b > 1e-3 or step_g > 1e-3 or step_s > 1e-3:
            improved = False
            for (db, dg, ds) in [(+step_b,0,0),(-step_b,0,0),(0,+step_g,0),(0,-step_g,0),(0,0,+step_s),(0,0,-step_s)]:
                nb, ng, ns, val = try_dir(b,g,s,db,dg,ds)
                if val + 1e-12 < best["loss"]:
                    best = {"beta": nb, "gamma": ng, "sigma": ns, "loss": float(val)}
                    b, g, s = nb, ng, ns
                    improved = True
            if not improved:
                step_b *= 0.5; step_g *= 0.5; step_s *= 0.5

    # final simulate + scale
    y_hat = simulate(best["beta"], best["gamma"], best["sigma"])
    scale = float(np.dot(y_obs, y_hat) / (np.dot(y_hat, y_hat) + 1e-12))
    sim = SEIRModel(N=N, beta=best["beta"], gamma=best["gamma"], sigma=best["sigma"]).simulate(
        t=t, I0=I0, E0=E0, R0_init=R0_init
    )
    return {
        "beta": best["beta"], "gamma": best["gamma"], "sigma": best["sigma"],
        "R0": best["beta"]/best["gamma"] if best["gamma"]>0 else np.inf,
        "loss": best["loss"], "scale": scale, "y_fit": scale * y_hat, "sim": sim
    }

@dataclass
class FitConfig:
    # fixed clinical/vital parameters
    sigma: float     # E->I (per week)
    gamma: float     # I->R (per week)
    mu: float        # vital rate (per week)
    N: float         # population

    # initial-condition ranges (fractions/absolutes)
    S0_range: Tuple[float, float] = (0.05, 0.40)    # as fraction of N
    E0_range: Tuple[float, float] = (1.0, 5e4)
    I0_range: Tuple[float, float] = (1.0, 5e4)

def _predict_cases(theta, t_grid, cfg: FitConfig):
    beta0, beta1, phi, rho, lag, S0_scale, E0, I0 = theta
    S0 = np.clip(S0_scale * cfg.N, 1.0, cfg.N - 1.0)
    R0_init = cfg.N - (S0 + E0 + I0)
    if R0_init < 0:
        return np.full_like(t_grid, np.nan, dtype=float) 

    p = SEIRParams(beta0=beta0, beta1=beta1, phi=phi,
                   sigma=cfg.sigma, gamma=cfg.gamma, mu=cfg.mu, N=cfg.N)

    SEIR = simulate_seir(t_grid, [S0, E0, I0, R0_init], p)
    E_series = SEIR[1, :]
    incidence = cfg.sigma * E_series  # new infections/week

    if lag <= 1e-8:
        reported = rho * incidence
    else:
        t_shifted = t_grid - lag
        reported = rho * np.interp(t_grid, t_shifted, incidence, left=np.nan, right=np.nan)

    return reported 

def sqrt_poisson_resid(theta, t_obs, y_obs, cfg: FitConfig):
    y_hat = _predict_cases(theta, t_obs, cfg)
    if np.any(~np.isfinite(y_hat)):
        return np.full_like(y_obs, 1e6, dtype=float)
    return (np.sqrt(y_obs + 0.5) - np.sqrt(y_hat + 0.5))

def fit_seasonal_seir(t_obs, y_obs, cfg: FitConfig):
    # theta = [beta0, beta1, phi, rho, lag, S0_scale, E0, I0]
    theta0 = np.array([700.0, 0.15, 0.0, 0.30, 1.0, 0.15, 100.0, 100.0], dtype=float)
    lower  = np.array([ 50.0, 0.00,-0.5, 0.01, 0.0, cfg.S0_range[0], cfg.E0_range[0], cfg.I0_range[0]])
    upper  = np.array([3000.0, 0.60, 0.5, 1.00, 2.0, cfg.S0_range[1], cfg.E0_range[1], cfg.I0_range[1]])

    res = least_squares(
        sqrt_poisson_resid, theta0, bounds=(lower, upper),
        args=(t_obs, y_obs, cfg),
        xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=200
    )

    theta_hat = res.x
    beta0, beta1, phi, rho, lag, S0_scale, E0_hat, I0_hat = theta_hat
    S0_hat = S0_scale * cfg.N
    R0_min = (beta0 * (1 - beta1)) / (cfg.gamma + cfg.mu)
    R0_max = (beta0 * (1 + beta1)) / (cfg.gamma + cfg.mu)

    return {
        'theta': theta_hat,
        'S0': S0_hat,
        'E0': E0_hat,
        'I0': I0_hat,
        'R0_range': (R0_min, R0_max),
        'success': res.success,
        'message': res.message,
        'nfev': res.nfev,
        'cost': res.cost,
        'status': res.status
    }

# Public helper so notebooks can reuse prediction:
def predict_reported(theta, t_grid, cfg: FitConfig):
    return _predict_cases(theta, t_grid, cfg)