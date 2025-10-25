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

def fit_sir_to_cumulative(df: pd.DataFrame, N: float,
                          t_col="date", y_col="cumulative_cases",
                          I0_guess=10.0, R0_guess=0.0,
                          beta_guess=0.25, gamma_guess=0.14,
                          p_report_guess=1.0) -> SIRFitResult:
    # strictly numpy inputs
    t = _to_timebase(df, t_col).astype(float)
    y_obs = np.asarray(df[y_col].to_numpy(dtype=float))  # 1-D np.array

    def pack(theta: np.ndarray) -> np.ndarray:
        beta, gamma, p_report, I0 = map(float, theta)
        S0 = float(N) - I0 - float(R0_guess)
        y0 = (S0, I0, float(R0_guess), 0.0)
        tt, Y = simulate_sir(t, y0, SIRParams(beta, gamma, float(N)))
        C = np.asarray(Y[3], dtype=float)
        return float(p_report) * C

    def resid(theta: np.ndarray) -> np.ndarray:
        return pack(theta) - y_obs

    # quick dry-run to catch shape/type issues early
    _ = pack(np.array([beta_guess, gamma_guess, p_report_guess, I0_guess], dtype=float))
    assert _.ndim == 1 and _.shape == y_obs.shape, f"Residual shape mismatch: {_.shape} vs {y_obs.shape}"

    bounds = (np.array([1e-6, 1e-3, 1e-3, 1.0], dtype=float),
              np.array([2.0,   1.0,  5.0,  N*1e-4], dtype=float))
    x0 = np.array([beta_guess, gamma_guess, p_report_guess, I0_guess], dtype=float)

    res = least_squares(resid, x0, bounds=bounds, loss="huber", method="trf", x_scale="jac")
    beta, gamma, p_report, I0 = res.x.astype(float)

    C_hat = pack(res.x)
    R0_basic = beta / gamma
    return SIRFitResult(beta, gamma, p_report, I0, R0_guess, R0_basic, t, C_hat)

# ------------------ SEIR ------------------

@dataclass
class SEIRFitResult:
    beta: float; sigma: float; gamma: float; p_report: float
    E0: float; I0: float; R0_init: float
    R0_basic: float
    t: np.ndarray; C_hat: np.ndarray

def fit_seir_to_cumulative(df: pd.DataFrame, N: float,
                           t_col="date", y_col="cumulative_cases",
                           E0_guess=20.0, I0_guess=10.0, R0_guess=0.0,
                           beta_guess=0.25, sigma_guess=0.10, gamma_guess=0.17,
                           p_report_guess=1.0) -> SEIRFitResult:
    t = _to_timebase(df, t_col).astype(float)
    y_obs = np.asarray(df[y_col].to_numpy(dtype=float))

    def pack(theta: np.ndarray) -> np.ndarray:
        beta, sigma, gamma, p_report, E0, I0 = map(float, theta)
        S0 = float(N) - E0 - I0 - float(R0_guess)
        y0 = (S0, E0, I0, float(R0_guess), 0.0)
        tt, Y = simulate_seir(t, y0, SEIRParams(beta, sigma, gamma, float(N)))
        C = np.asarray(Y[4], dtype=float)
        return float(p_report) * C

    def resid(theta: np.ndarray) -> np.ndarray:
        return pack(theta) - y_obs

    _ = pack(np.array([beta_guess, sigma_guess, gamma_guess, p_report_guess, E0_guess, I0_guess], dtype=float))
    assert _.ndim == 1 and _.shape == y_obs.shape, f"Residual shape mismatch: {_.shape} vs {y_obs.shape}"

    bounds = (np.array([1e-6, 1e-3, 1e-3, 1e-3, 1.0, 1.0], dtype=float),
              np.array([2.0,   1.0,  1.0,  5.0,   N*1e-4, N*1e-4], dtype=float))
    x0 = np.array([beta_guess, sigma_guess, gamma_guess, p_report_guess, E0_guess, I0_guess], dtype=float)

    res = least_squares(resid, x0, bounds=bounds, loss="huber", method="trf", x_scale="jac")
    beta, sigma, gamma, p_report, E0, I0 = res.x.astype(float)

    C_hat = pack(res.x)
    R0_basic = beta / gamma
    return SEIRFitResult(beta, sigma, gamma, p_report, E0, I0, R0_guess, R0_basic, t, C_hat)