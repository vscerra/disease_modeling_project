"""
===========================================================
seir_scipy_fit.py
Author: Veronica Scerra
Last Updated: 2025-10-21
===========================================================

Description:
    Deterministic SEIR (Susceptible-Exposed-Infectious-Recovered)
    model with scipy fitting.

Notes:
    - beta: transmission rate (per week)
    - gamma: recovery rate (per week) [1/gamma = infectious period]
    - sigma: progression rate E->I (per week)  [1/sigma = incubation period]

-----------------------------------------------------------
License: MIT
===========================================================
"""
# src/models/seir.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Sequence
from scipy.integrate import solve_ivp

@dataclass
class SEIRParams:
    beta: float     # transmission rate
    sigma: float    # 1/incubation
    gamma: float    # 1/infectious duration
    N: float        # population size

def _rhs(t, y, p: SEIRParams):
    S, E, I, R, C = y
    inf = p.beta * S * I / p.N       # force of infection
    dS = -inf
    dE =  inf - p.sigma * E
    dI =  p.sigma * E - p.gamma * I
    dR =  p.gamma * I
    dC =  p.sigma * E                # cumulative onset of infectious (≈ cases)
    return (dS, dE, dI, dR, dC)

def simulate_seir(t_eval: Sequence[float], y0: Tuple[float,float,float,float,float], p: SEIRParams):
    t_span = (t_eval[0], t_eval[-1])
    sol = solve_ivp(lambda t, y: _rhs(t, y, p),
                    t_span, y0, t_eval=np.asarray(t_eval), rtol=1e-6, atol=1e-8)
    return sol.t, sol.y  # [5 x len(t)]