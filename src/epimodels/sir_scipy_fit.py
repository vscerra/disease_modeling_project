"""
===========================================================
sir_scipy_fit.py
Author: Veronica Scerra
Last Updated: 2025-10-25
===========================================================

Description:
    Deterministic SIR (Susceptible-Infectious-Recovered)
    epidemiological model implemented using scipy fitters

Example Usage:
    from epimodels.sir_scipy_fit import ***
    model = ***
    results = ***

Notes:
    - Uses NumPy and SciPy fitting.
    - Designed for reproducible research and educational use.
-----------------------------------------------------------
License: MIT
===========================================================
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Sequence
from scipy.integrate import solve_ivp

@dataclass
class SIRParams:
    beta: float     # transmission rate
    gamma: float    # recovery/removal rate
    N: float        # population size

def _rhs(t, y, p: SIRParams):
    S, I, R, C = y
    inf = p.beta * S * I / p.N 
    dS = -inf 
    dI = inf - p.gamma * I 
    dR = p.gamma * I 
    dC = inf           # cumulative new infections
    return (dS, dI, dR, dC) 

def simulate_sir(t_eval: Sequence[float], y0: Tuple[float, float, float, float], p: SIRParams):
    t_span = (t_eval[0], t_eval[-1])
    sol = solve_ivp(lambda t, y: _rhs(t, y, p),
                    t_span, y0, t_eval=np.asarray(t_eval), rtol=1e-6, atol=1e-8)
    return sol.t, sol.y # (times, state matrix [4 x len(t)])


