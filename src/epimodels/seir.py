"""
===========================================================
seir.py
Author: Veronica Scerra
Last Updated: 2025-10-17
===========================================================

Description:
    Deterministic SEIR (Susceptible–Exposed–Infectious–Recovered)
    model with constant rates and RK4 integration.

API:
    SEIRModel(N, beta, gamma, sigma)
      - simulate(t, I0, E0=0, R0_init=0) -> dict(t,S,E,I,R,incidence)
      - summary(outputs) -> dict of peak day, prevalence, final size

Notes:
    - beta: transmission rate (per day)
    - gamma: recovery rate (per day)  [1/gamma = infectious period]
    - sigma: progression rate E->I (per day)  [1/sigma = incubation period]
    - incidence ≈ -ΔS per step
-----------------------------------------------------------
License: MIT
===========================================================
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

def _rhs(S: float, E: float, I: float, R: float, beta: float, gamma: float, sigma: float, N: float) -> Tuple[float, float, float, float]:
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I 
    dR = gamma * I 
    return dS, dE, dI, dR

def _rk4_step(S, E, I, R, h, beta, gamma, sigma, N):
    k1 = _rhs(S,E,I,R,beta,gamma,sigma,N)
    k2 = _rhs(S+0.5*h*k1[0], E+0.5*h*k1[1], I+0.5*h*k1[2], R+0.5*h*k1[3], beta,gamma,sigma,N)
    k3 = _rhs(S+0.5*h*k2[0], E+0.5*h*k2[1], I+0.5*h*k2[2], R+0.5*h*k2[3], beta,gamma,sigma,N)
    k4 = _rhs(S+h*k3[0],   E+h*k3[1],   I+h*k3[2],   R+h*k3[3],   beta,gamma,sigma,N)
    Sn = S + (h/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    En = E + (h/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    In = I + (h/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    Rn = R + (h/6.0)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
    return Sn, En, In, Rn

class SEIRModel:
    def __init__(self, N: int, beta: float, gamma: float, sigma: float):
        self.N = float(N)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.sigma = float(sigma)

    @property
    def R0(self) -> float:
        # for SEIR with homogeneous mixing: R0 - beta / gamma (same as SIR) when no births/deaths
        return self.beta / self.gamma if self.gamma > 0 else np.inf 
    
    def simulate(self, t: np.ndarray, I0: int, E0: int = 0, R0_init: int = 0) -> Dict[str, np.ndarray]:
        S0 = self.N - I0 - R0_init - E0
        S = np.empty_like(t, dtype=float)
        E = np.empty_like(t, dtype=float)
        I = np.empty_like(t, dtype=float)
        R = np.empty_like(t, dtype=float)
        S[0], E[0], I[0], R[0] = S0, float(E0), float(I0), float(R0_init)
        incidence = np.zeros_like(t, dtype=float)
        for k in range(1, len(t)):
            h = float(t[k]-t[k-1])
            Sn, En, In, Rn = _rk4_step(S[k-1], E[k-1], I[k-1], R[k-1], h, self.beta, self.gamma, self.sigma, self.N)
            S[k] = max(Sn, 0.0)
            E[k] = max(En, 0.0)
            I[k] = max(In, 0.0)
            R[k] = max(Rn, 0.0)
            incidence[k] = max(S[k-1] - S[k], 0.0)
        return {"t": t, "S": S, "E": E, "I": I, "R": R, "incidence": incidence}
    
    @staticmethod
    def summary(outputs: Dict[str, np.ndarray]) -> Dict[str, float]:
        t, I = outputs["t"], outputs["I"]
        S0 = outputs["S"][0]; E0 = outputs.get("E", np.array([0.0]))[0]; I0 = outputs["I"][0]; R0 = outputs["R"][0]
        N0 = S0 + E0 + I0 + R0
        peak_idx = int(np.argmax(I))
        return {
            "peak_day": float(t[peak_idx]),
            "peak_infected": float(I[peak_idx]),
            "peak_prevalence": float(I[peak_idx] / N0),
            "final_size": float(outputs["R"][-1] / N0),
        }
    
    