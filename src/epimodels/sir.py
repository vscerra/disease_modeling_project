"""
===========================================================
sir.py
Author: Veronica Scerra
Last Updated: 2025-10-11
===========================================================

Description:
    Core deterministic SIR (Susceptible–Infectious–Recovered)
    epidemiological model implemented using a Runge–Kutta 4th
    order integrator.

    Defines:
        - sir_rhs(): Computes the ODE right-hand side.
        - sir_rk4(): Integrates the SIR system over time.
        - SIRModel: Class wrapper with convenience methods
                    for simulation and summary stats.

Example Usage:
    from epimodels.sir import SIRModel
    model = SIRModel(N=10000, beta=0.3, gamma=0.1)
    results = model.simulate(t, I0=10)

Notes:
    - Uses NumPy only (no SciPy dependency).
    - Designed for reproducible research and educational use.
-----------------------------------------------------------
License: MIT
===========================================================
"""
import numpy as np
from typing import Dict, Tuple

def sir_rhs(S, I, R, beta, gamma, N):
    """Right-hand side of the SIR equations"""
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return dS, dI, dR


def rk4_steps(S, I, R, h, beta, gamma, N):
    """single RK4 step"""
    k1 = sir_rhs(S, I, R, beta, gamma, N)
    k2 = sir_rhs(S + 0.5*h*k1[0], I + 0.5*h*k1[1], R + 0.5*h*k1[2], beta, gamma, N)
    k3 = sir_rhs(S + 0.5*h*k2[0], I + 0.5*h*k2[1], R + 0.5*h*k2[2], beta, gamma, N)
    k4 = sir_rhs(S + h*k3[0], I + h*k3[1], R + h*k3[2], beta, gamma, N)
    S_next = S + (h/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    I_next = I + (h/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    R_next = R + (h/6)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    return S_next, I_next, R_next


def sir_rk4(t: np.ndarray, N: int, beta: float, gamma: float, I0: int, R0: int=0) -> Dict[str, np.ndarray]:
    """Integrate SIR with RK4 over time grid t"""
    S0 = N - I0 - R0
    S, I, R = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    S[0], I[0], R[0] = S0, I0, R0

    incidence = np.zeros_like(t)
    for k in range(1, len(t)):
        h = t[k] - t[k-1]
        S[k], I[k], R[k] = rk4_steps(S[k-1], I[k-1], R[k-1], h, beta, gamma, N)
        incidence[k] = max(S[k-1] - S[k], 0.0)
    
    return {"t": t, "S": S, "I": I, "R": R, "incidence": incidence}


class SIRModel:
    def __init__(self, N, beta, gamma):
        self.N, self.beta, self.gamma = float(N), float(beta), float(gamma)

    @property
    def R0(self):
        return self.beta / self.gamma if self.gamma else np.inf
    
    def simulate(self, t, I0, R0_init=0):
        return sir_rk4(t, self.N, self.beta, self.gamma, I0, R0_init)
    
    @staticmethod
    def summary(outputs):
        t, I, R = outputs["t"], outputs["I"], outputs["R"]
        peak_idx = np.argmax(I)
        N = outputs["S"][0] + I[0] + R[0]
        return {
            "peak_day": float(t[peak_idx]),
            "peak_infected": float(I[peak_idx]),
            "peak_prevalence": float(I[peak_idx]/N),
            "final_size": float(R[-1]/N)
        }