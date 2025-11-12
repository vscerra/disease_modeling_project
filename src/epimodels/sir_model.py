"""
===========================================================
sir_model.py
Author: Veronica Scerra
Last Updated: 2025-11-12
===========================================================
SIR (Susceptible-Infected-Recovered) Model

A basic compartmental epidemiological model that divides 
a population into three compartments: 
- S: Susceptible individuals
- I: Infected (and infectous) individuals
- R: Recovered (and immune) individuals

This model assumes: 
- Homogenous mixing (everyone has equal contact probability)
- No births, deaths, or migrations (closed population)
- Permanent immunity after recovery
- Fequency-dependent transmission

-----------------------------------------------------------
License: MIT
===========================================================
"""

import numpy as np
from scipy.integrate import odeint
from typing import Tuple, Optional 
import matplotlib.pyplot as plt 

class SIRModel:
    """
    SIR compartmental model for infectious disease dynamics
    
    Parameters:
    -----------
    beta: float
        Transmission rate (contacts per time x probability of transmission per conact)
    gamma: float
        Recover rate (1/gamma = mean infectious period)
    N: float
        Total population size
    """
    def __init__(self, beta:float, gamma:float, N:float):
        self.beta = beta
        self.gamma = gamma
        self.N = N

    @property
    def R0(self) -> float:
        """
        Basic reproduction number: average number of secondary infections 
        caused by a single infected individual in a fully susceptible population
        """
        return self.beta / self.gamma
    
    def deriv(self, y: np.ndarray, t: float) -> np.ndarray:
        """
        Compute derivatives for the SIR model
        
        Parameters:
        -----------
        y: array-like
            current state [S, I, R]
        t: float
            current time (not used in autonomous system, but required by odeint)
        
        Returns:
        --------
        dydt: np.ndarrary
            Derivatives [dS/dt, dI/dt, dR/dt]
        """
        S, I, R = y

        dSdt = -self.beta * S * I / self.N
        dIdt = self.beta * S * I / self.N - self.gamma * I 
        dRdt = self.gamma * I 

        return np.array([dSdt, dIdt, dRdt])
    
    def simulate(self,
                 S0: float,
                 I0: float,
                 R0: float,
                 t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        """ 
        Run simulation of SIR model.
        
        Parameters:
        -----------
        S0 : float
            Initial number of susceptible individuals
        I0 : float
            Initial number of infected individuals
        R0 : float
            Initial number of recovered individuals
        t : np.ndarray
            Time points for simulation
            
        Returns
        -------
        S, I, R : tuple of np.ndarray
            Time series for each compartment
        """
        y0 = [S0, I0, R0]
        solution = odeint(self.deriv, y0, t)

        S, I, R = solution.T
        return S, I, R
    
def compute_epidemic_metrics(t: np.ndarray, 
                            I: np.ndarray, 
                            R: np.ndarray) -> dict:
    """
    Compute key epidemic metrics from simulation results.

    Parameters
    ----------
    t : np.ndarray
        Time points
    I : np.ndarray
        Infected individuals over time
    R : np.ndarray
        Recovered individuals over time

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - peak_infected: maximum number of infected individuals
        - peak_time: time at which peak occurs
        - attack_rate: final proportion of population infected
        - epidemic_duration: approximate duration (when I < 1% of peak)
    """
    peak_infected = np.max(I)
    peak_time = t[np.argmax(I)]
    attack_rate = R[-1]  # Final recovered = total who were infected

    # Find when infection drops below 1% of peak
    threshold = 0.01 * peak_infected
    epidemic_end_idx = np.where(I[np.argmax(I):] < threshold)[0]
    if len(epidemic_end_idx) > 0:
        epidemic_duration = t[np.argmax(I) + epidemic_end_idx[0]] - t[0]
    else:
        epidemic_duration = t[-1] - t[0]

    return {
        'peak_infected': peak_infected,
        'peak_time': peak_time,
        'attack_rate': attack_rate,
        'epidemic_duration': epidemic_duration
    }

