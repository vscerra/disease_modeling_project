"""
===========================================================
yellow_fever_models.py
Author: Veronica Scerra
Last Updated: 2026-01-15
===========================================================
Base SEIR Model with Vaccination for Yellow Fever
==================================================

This module implements a compartmental SEIR model with vaccination
for modeling Yellow Fever transmission dynamics in urban settings.

Model Structure:
    S -> E -> I -> R
         ↓    ↓
         V    D
         
Compartments:
    S: Susceptible
    E: Exposed (infected but not yet infectious)
    I: Infectious
    R: Recovered (natural immunity)
    V: Vaccinated (vaccine-induced immunity)
    D: Deaths (cumulative)

License: MIT
===========================================================
"""

import numpy as np
import warnings 
from scipy.integrate import solve_ivp 
from typing import Dict, Tuple, Callable, Optional 

class YellowFeverModel:
    """
    SEIR model for Yellow Fever with vaccination intervention.
    
    Parameters:
    beta : float. Transmission rate (effective contacts per day)
    sigma : float. Progression rate from Exposed to Infectious (1/incubation_period)
    gamma : float. Recovery rate (1/infectious_period)
    alpha : float. Disease-induced mortality rate
    vaccination_func : callable, optional
        Time-dependent vaccination rate function: ν(t)
        If None, no vaccination is modeled
    
    Attributes:
    N : float
        Total population size
    """

    def __init__(
            self,
            beta: float,
            sigma: float,
            gamma: float,
            alpha: float,
            vaccination_func: Optional[Callable[[float], float]]=None,
            population: float=800000
    ):
        """Initialize the Yellow Fever SEIR model"""
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma 
        self.alpha = alpha 
        self.vaccination_func = vaccination_func if vaccination_func else lambda t: 0.0 
        self.N = population 
        self._validate_parameters() 
    
    def _validate_parameters(self):
        """Validate that all parameters are physically reasonable"""
        if self.beta < 0:
            raise ValueError("transmission rate beta must be non-negative")
        if self.sigma <= 0:
            raise ValueError("progression rate sigma must be positive") 
        if self.gamma <= 0:
            raise ValueError("recovery rate gamma must be positive") 
        if self.alpha < 0:
            raise ValueError("mortality rate alpha must be non-negative")
        if self.N <= 0:
            raise ValueError("population N must be positive")
        
    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """calculate derivatives for the SEIR model
        
        Parameters:
        t: float. Current time 
        y: array-like. Current status [S, E, I, R, V, D]
        
        Returns:
        dydt: ndarray. Derivatives [dS/dt, dE/dt, dI/dt, dR/dt, dV/dt, dD/dt]
        """
        S, E, I, R, V, D = y

        #force of infection (lambda = beta * I / N)
        force_of_infection = self.beta * I / self.N 

        #vaccination rate at time t
        nu = self.vaccination_func(t) 

        #ODE system 
        dS = -force_of_infection * S - nu * S
        dE = force_of_infection * S - self.sigma * E 
        dI = self.sigma * E - self.gamma * I - self.alpha * I 
        dR = self.gamma * I 
        dV = nu * S 
        dD = self.alpha * I 

        return np.array([dS, dE, dI, dR, dV, dD])
    
    def simulate(
            self,
            initial_conditions: Dict[str, float],
            t_span: Tuple[float, float],
            t_eval: Optional[np.ndarray] = None,
            method: str = 'RK45'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Simulate the yellow fever epidemic model 
        
        Parameters:
        initial_conditions: dict. Initial values for compartments: {'S': s0, 'E': e0, 'I': i0, ...}
        t_span: tuple. Time span for integration (t_start, t_end) 
        t_eval: array-like, optional. Times at which to evaluate solution. If none, solver picks times
        method: str, default='RK45'. ODE solver method (RK45, RK23, DOP853, Radau, BDF, LSODA)
        
        Returns: 
        t: ndarray. Time points 
        y: ndarray. Solution array with shape (6, len(t))
        Rows: [S, E, I, R, V, D]
        """
        # set up initial conditions vector
        y0 = np.array([
           initial_conditions.get('S', self.N),
            initial_conditions.get('E', 0),
            initial_conditions.get('I', 0),
            initial_conditions.get('R', 0),
            initial_conditions.get('V', 0),
            initial_conditions.get('D', 0) 
        ])

        # validate initial conditions 
        if not np.isclose(y0[:5].sum(), self.N):
            warnings.warn(
                f"Initial conditions sum to {y0[:5].sum():.0f}, "
                f"but population is {self.N}. Adjusting S."
            )
            y0[0] = self.N - y0[1:5].sum() 
        
        # Solve ODE system
        solution = solve_ivp(
            fun = self.derivatives,
            t_span = t_span,
            y0 = y0,
            method = method, 
            t_eval = t_eval,
            dense_output = True,
            max_step = 1.0 #max step size of 1 day
        )

        if not solution.success:
            raise RuntimeError(f"ODE sovler failed: {solution.message}")
        
        return solution.t, solution.y 
    
    def calculate_r0(self) -> float:
        """Calculate the basic reproduction number R0
        
        For SEIR model: R0 = beta / (gamma + alpha)
        
        Returns:
        r0: float. Basic reproduction number
        """
        r0 = self.beta / (self.gamma + self.alpha)
        return r0 
    
    def calculate_reff(self, S: float) -> float:
        """ Calculate the effective reproduction number r_eff
        r_eff(t) = r0 * (S(t) / N)
        
        Parameters:
        S: float. Current number of susceptibles 

        Returns:
        reff: float. Effective reproduction number
        """
        r0 = self.calculate_r0()
        reff = r0 * (S / self.N)
        return reff 
    
    def get_endemic_equilibrium(self) -> Dict[str, float]:
        """ Calculate endemic equilibrium (if it exists) 
        Endemic equilibrium exists if R0 > 1 
        
        Returns:
        equilibrium: dict. Equilibrium values for each compartment
        """
        r0 = self.calculate_r0()

        if r0 <= 1:
            # disease free equilibrium
            return {
                'S': self.N,
                'E': 0,
                'I': 0,
                'R': 0,
                'V': 0,
                'D': 0
            }
        
        # endemic equilibrium
        I_star = (self.N * self.sigma * (r0 - 1)) / (self.beta * (1 + self.sigma / (self.gamma + self.alpha)))
        S_star = self.N / r0
        E_star = (self.gamma + self.alpha) * I_star / self.sigma
        R_star = self.N - S_star - E_star - I_star
        
        return {
            'S': S_star,
            'E': E_star,
            'I': I_star,
            'R': R_star,
            'V': 0,
            'D': 0
        }
    
    def get_parameter_dict(self) -> Dict[str, float]:
        """Return dictionary of current model parametesr"""
        return {
            'beta': self.beta,
            'sigma': self.sigma,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'N': self.N 
        }
    
    def __repr__(self) -> str:
        """String representation of the model"""
        r0 = self.calculate_r0()
        return (
            f"YellowFeverModel(\n"
            f"  β={self.beta:.4f} (transmission rate)\n"
            f"  σ={self.sigma:.4f} (1/incubation)\n"
            f"  γ={self.gamma:.4f} (recovery rate)\n"
            f"  α={self.alpha:.4f} (mortality rate)\n"
            f"  N={self.N:.0f} (population)\n"
            f"  R₀={r0:.4f}\n"
            f")"
        )
    
def create_vaccination_function(
        start_date: float,
        vaccination_rate: float,
        ramp_duration: float = 7.0,
        vaccine_efficiency: float = 0.95,
) -> Callable[[float], float]:
    """Create a time-dependent vaccination rate function
    
    Models vaccination as a sigmoid ramp=up followed by a constant rate
    Accounts for vaccine efficiency (default 95% per WHO data) 
    
    Parameters:
    start_date: float. Day when vaccine program begins 
    vaccination_rate: float. Daily vaccination rate (proportion of susceptibles vaccinate each day) after full ramp-up
    ramp_duration: float, default = 7.0. Duration of ramp-up period (days)
    vaccine_efficiency: float, default=0.95. Proportion of vaccinated individuals who develop immunity 
    
    Returns: 
    vaccination_func: callable. Function v(t) that returns vaccination rate at time t
    """

    def vaccination_func(t: float) -> float:
        if t < start_date:
            return 0.0 
        elif t < start_date + ramp_duration:
            #sigmoid ramp up
            progress = (t - start_date) / ramp_duration 
            ramp = 1 / (1 + np.exp(-10 * (progress - 0.5))) 
            return vaccination_rate * vaccine_efficiency * ramp 
        else:
            # full vaccination rate
            return vaccination_rate * vaccine_efficiency 
        
    return vaccination_func