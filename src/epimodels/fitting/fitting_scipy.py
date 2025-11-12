"""
===========================================================
fitting_custom.py
Author: Veronica Scerra
Last Updated: 2025-10-25
===========================================================

Description:
    Estimate SIR and SEIR parameters (beta, gamma) from time series by
    using scipy least_squares fitting

Example Usage:
    from epimodels/fitting/fitting_scipy.py import SIRFitResults
    result = sir_fit_to_cumulative(
        ***
    )

Notes:
    - Observations should align with t (same length).
    - Consider smoothing noisy incidence before fitting (moving average).
-----------------------------------------------------------
License: MIT
===========================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Literal, Tuple
from scipy.optimize import least_squares
from epimodels.sir_scipy_fit import simulate_sir, SIRParams
from epimodels.seir_scipy_fit import simulate_seir, SEIRParams

def _to_timebase(df: pd.DataFrame, t_col="date") -> np.ndarray:
    t = pd.to_datetime(df[t_col]).astype("int64") // 10**9  # seconds
    t0 = t.iloc[0]
    return (t - t0) / (60*60*24)  # days since start

# ------------------ SIR ------------------

@dataclass
class SIRFitResult:
    beta: float; gamma: float; p_report: float
    I0: float; R0_init: float
    R0_basic: float
    t: np.ndarray; C_hat: np.ndarray

# --- SIR cumulative fit with C0 offset ---
def fit_sir_to_cumulative(df: pd.DataFrame, N: float,
                          t_col="date", y_col="cumulative_cases",
                          I0_guess=10.0, R0_guess=0.0,
                          beta_guess=0.25, gamma_guess=0.14,
                          p_report_guess=0.8,
                          C0_guess=None) -> SIRFitResult:
    # --- strictly NumPy inputs ---
    t = (pd.to_datetime(df[t_col]).astype("int64").to_numpy() / 1e9)  # seconds
    t = (t - t[0]) / (60*60*24)                                       # days
    t = np.asarray(t, dtype=float)

    y_obs = np.asarray(df[y_col].to_numpy(dtype=float))               # 1-D array

    y_obs = np.asarray(df[y_col].to_numpy(dtype=float))
    first = float(y_obs[0])
    if C0_guess is None:
        C0_guess = first  # start near the first observed cumulative

    def pack(theta: np.ndarray) -> np.ndarray:
        beta, gamma, p_report, I0, C0 = map(float, theta)
        S0 = float(N) - I0 - float(R0_guess)
        y0 = (S0, I0, float(R0_guess), 0.0)
        _, Y = simulate_sir(t, y0, SIRParams(beta, gamma, float(N)))
        C = np.asarray(Y[3], dtype=float) + C0
        return p_report * C

    def resid(theta: np.ndarray) -> np.ndarray:
        return pack(theta) - y_obs

    # Dry run
    _ = pack(np.array([beta_guess, gamma_guess, p_report_guess, I0_guess, C0_guess], dtype=float))
    assert _.shape == y_obs.shape

    # Bounds: 0<p<=1 (interpretability), C0 between 0 and ~2×first observed
    bounds = (np.array([1e-6, 1e-3, 1e-3, 1.0, 0.0], dtype=float),
              np.array([2.0,   1.0,  1.0,  N*1e-4, max(1.0, 2.0*first)], dtype=float))
    x0 = np.array([beta_guess, gamma_guess, p_report_guess, I0_guess, C0_guess], dtype=float)

    res = least_squares(resid, x0, bounds=bounds, loss="huber", method="trf", x_scale="jac")
    beta, gamma, p_report, I0, C0 = res.x.astype(float)
    C_hat = pack(res.x)
    R0_basic = beta / gamma

    # Return with C0 included (extend your dataclass if you like)
    return SIRFitResult(beta, gamma, p_report, I0, R0_guess, R0_basic, t, C_hat)

# ------------------ SEIR ------------------

@dataclass
class SEIRFitResult:
    beta: float; sigma: float; gamma: float; p_report: float
    E0: float; I0: float; R0_init: float
    R0_basic: float
    t: np.ndarray; C_hat: np.ndarray

# --- SEIR cumulative fit with C0 offset ---
def fit_seir_to_cumulative(df: pd.DataFrame, N: float,
                           t_col="date", y_col="cumulative_cases",
                           E0_guess=20.0, I0_guess=10.0, R0_guess=0.0,
                           beta_guess=0.25, sigma_guess=0.10, gamma_guess=0.17,
                           p_report_guess=0.8,
                           C0_guess=None) -> SEIRFitResult:
    # --- strictly NumPy inputs ---
    t = (pd.to_datetime(df[t_col]).astype("int64").to_numpy() / 1e9)  # seconds
    t = (t - t[0]) / (60*60*24)                                       # days
    t = np.asarray(t, dtype=float)

    y_obs = np.asarray(df[y_col].to_numpy(dtype=float))               # 1-D array

    y_obs = np.asarray(df[y_col].to_numpy(dtype=float))
    first = float(y_obs[0])
    if C0_guess is None:
        C0_guess = first

    def pack(theta: np.ndarray) -> np.ndarray:
        beta, sigma, gamma, p_report, E0, I0, C0 = map(float, theta)
        S0 = float(N) - E0 - I0 - float(R0_guess)
        y0 = (S0, E0, I0, float(R0_guess), 0.0)
        _, Y = simulate_seir(t, y0, SEIRParams(beta, sigma, gamma, float(N)))
        C = np.asarray(Y[4], dtype=float) + C0
        return p_report * C

    def resid(theta: np.ndarray) -> np.ndarray:
        return pack(theta) - y_obs

    _ = pack(np.array([beta_guess, sigma_guess, gamma_guess, p_report_guess,
                       E0_guess, I0_guess, C0_guess], dtype=float))
    assert _.shape == y_obs.shape

    # Keep σ≈1/9–1/12 and γ≈1/5–1/7 in-bounds; 0<p<=1; C0 in [0, 2*first]
    bounds = (np.array([1e-6,  1/20,  1/14, 1e-3, 1.0, 1.0, 0.0], dtype=float),
              np.array([2.0,   1/6,   1/3,  1.0,  N*1e-4, N*1e-4, max(1.0, 2.0*first)], dtype=float))
    x0 = np.array([beta_guess, sigma_guess, gamma_guess, p_report_guess,
                   E0_guess, I0_guess, C0_guess], dtype=float)

    res = least_squares(resid, x0, bounds=bounds, loss="huber", method="trf", x_scale="jac")
    beta, sigma, gamma, p_report, E0, I0, C0 = res.x.astype(float)
    C_hat = pack(res.x)
    R0_basic = beta / gamma

    return SEIRFitResult(beta, sigma, gamma, p_report, E0, I0, R0_guess, R0_basic, t, C_hat)