"""
===========================================================
sir_piecewise.py
Author: Veronica Scerra
Last Updated: 2025-10-15
===========================================================

Description:
    Deterministic SIR model with piecewise-constant transmission
    rate β(t). Useful when interventions/behavior change over time
    and a single β cannot fit multiple waves.

API:
    - SIRPiecewiseBeta(N, gamma, edges, betas)
      where:
        edges: 1D array-like of strictly increasing times (same units as t),
               length = K+1 for K segments (closed-open: [e0,e1),...,[e_{K-1},eK])
        betas: length-K array of β for each segment
    - simulate(t, I0, R0_init=0)
    - summary(outputs)

Notes:
    - β segment is chosen based on the *left* time of each ODE step.
    - Keep units consistent (daily t → per-day rates).
-----------------------------------------------------------
License: MIT
===========================================================
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Sequence, Tuple

def _sir_rhs(S: float, I: float, R: float, beta: float, gamma: float, N: float) -> Tuple[float, float, float]:
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return dS, dI, dR

def _rk4_step(S: float, I: float, R: float, h: float, beta: float, gamma: float, N: float) -> Tuple[float, float, float]:
    k1 = _sir_rhs(S, I, R, beta, gamma, N)
    k2 = _sir_rhs(S + 0.5*h*k1[0], I + 0.5*h*k1[1], R + 0.5*h*k1[2], beta, gamma, N)
    k3 = _sir_rhs(S + 0.5*h*k2[0], I + 0.5*h*k2[1], R + 0.5*h*k2[2], beta, gamma, N)
    k4 = _sir_rhs(S + h*k3[0], I + h*k3[1], R + h*k3[2], beta, gamma, N)
    Sn = S + (h/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    In = I + (h/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    Rn = R + (h/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    return Sn, In, Rn

def _which_segment(tk: float, edges: np.ndarray) -> int:
    """
    Return segment index k such that edges[k] <= tk < edges[k+1].
    Assumes tk within the edges range.
    """
    # binary search is overkill for small K, but vectorized searchsorted is fine
    k = int(np.searchsorted(edges, tk, side="right") - 1)
    return max(0, min(k, len(edges)-2))

class SIRPiecewiseBeta:
    """
    SIR with piecewise-constant β(t). γ is constant across segments.
    """
    def __init__(self, N: int, gamma: float, edges: Sequence[float], betas: Sequence[float]):
        edges = np.asarray(edges, dtype=float)
        betas = np.asarray(betas, dtype=float)
        assert len(edges) >= 2 and np.all(np.diff(edges) > 0), "edges must be strictly increasing, len>=2"
        assert len(betas) == len(edges) - 1, "betas length must be len(edges)-1"
        self.N = float(N)
        self.gamma = float(gamma)
        self.edges = edges
        self.betas = betas

    def _beta_at(self, tk: float) -> float:
        k = _which_segment(tk, self.edges)
        return float(self.betas[k])

    def simulate(self, t: np.ndarray, I0: int, R0_init: int = 0) -> Dict[str, np.ndarray]:
        S0 = self.N - I0 - R0_init
        S = np.empty_like(t, dtype=float); I = np.empty_like(t, dtype=float); R = np.empty_like(t, dtype=float)
        S[0], I[0], R[0] = S0, float(I0), float(R0_init)
        incidence = np.zeros_like(t, dtype=float)

        for k in range(1, len(t)):
            h = float(t[k] - t[k-1])
            beta = self._beta_at(t[k-1])  # β set by the left edge of the interval
            Sn, In, Rn = _rk4_step(S[k-1], I[k-1], R[k-1], h, beta, self.gamma, self.N)
            S[k] = max(Sn, 0.0); I[k] = max(In, 0.0); R[k] = max(Rn, 0.0)
            incidence[k] = max(S[k-1] - S[k], 0.0)

        return {"t": t, "S": S, "I": I, "R": R, "incidence": incidence}

    @staticmethod
    def summary(outputs: Dict[str, np.ndarray]) -> Dict[str, float]:
        t, I, R = outputs["t"], outputs["I"], outputs["R"]
        N0 = outputs["S"][0] + I[0] + R[0]
        peak_idx = int(np.argmax(I))
        return {
            "peak_day": float(t[peak_idx]),
            "peak_infected": float(I[peak_idx]),
            "peak_prevalence": float(I[peak_idx] / N0),
            "final_size": float(R[-1] / N0),
        }