"""
===========================================================
seir_scipy_fit.py
Author: Veronica Scerra
Last Updated: 2025-10-21
===========================================================

Description:
    Deterministic SEIR (Susceptible-Exposed-Infectious-Recovered)
    model with constant rates and RK4 integration.
    SEIR with seasonal Beta(t) and links model to data via an observation
    model

Notes:
    - beta: transmission rate (per week)
    - gamma: recovery rate (per week) [1/gamma = infectious period]
    - sigma: progression rate E->I (per week)  [1/sigma = incubation period]

-----------------------------------------------------------
License: MIT
===========================================================
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple 
from scipy.integrate import solve_ivp 

@dataclass
class SEIRParams:
    beta0: float    # baseline transmission (per week)
    beta1: float    # seasonal amplitude [0, 1)
    phi: float      # seasonal phase (years)
    sigma: float    # E -> I rate (per week)
    gamma: float    # I -> R rate (per week)
    mu: float       # vital rate (per week)
    N: float        # total population

def beta_t(t: float, beta0: float, beta1: float, ph: float) -> float:
    # t in weeks; period ~ 52 weeks
    return beta0 * (1.0 + beta1 * np.cos(2.0 * np.pi*(t/52.0 - phi)))

def seir_rhs(t: float, y: Sequence[float], p: SEIRParams) -> Tuple[float, float, float, float]:
    S, E, I, R = y
    b = beta_t(t, p.beta0, p.beta1, p.phi)
    dS = p.mu*p.N - b*S*I/p.N - p.mu*S
    dE = b*S*I/p.N - (p.sigma + p.mu)*E
    dI = p.sigma*E - (p.gamma + p.mu)*I
    dR = p.gamma*I - p.mu*R
    return (dS, dE, dI, dR) 

def simulate_seir(t_eval: np.ndarray, y0: Sequence[float], p: SEIRParams,
                  r_tol: float=1e-6, atol: float=1e-9) -> np.ndarray:
    sol = solve_ivp(lambda t, y: seir_rhs(t, y, p),
                    (float(t_eval[0]), float(t_eval[-1])),
                    y0, t_eval=t_eval, dense_output=False,
                    r_tol=r_tol, atol=atol) 
    if not sol.success:
        raise RuntimeError("ODE integration failed: " + sol.message)
    return sol.y    # shape (4, len(t_eval))