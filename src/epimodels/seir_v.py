"""
===============================================================================
seir_v.py
Author: Veronica Scerra
Last Updated: 2026-02-04
===============================================================================
SEIR model with vaccination 

Deterministc SEIR-V compartmental model for measles
transmission with vaccination. The model includes:
- Susceptible (S), Exposed (E), Infected (I), Recovered (R), Vaccinated (V)
- Age-structured vaccination (12 months, 4 years)
- Demographic turnover (births, deaths)
- Vaccine efficiency and waning (if applicable)

This model serves as the foundation for behavioral economics analysis
--------------------------------------------------------------------------------
License: MIT
================================================================================
"""
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import odeint 
from typing import Tuple, Dict, Optional 

class SEIRVModel:
    """SEIR-V compartmental model for measles with vaccination.

    Compartments:
        S - Susceptible
        E - Exposed (infected but not yet infectious)
        I - Infected (infectious)
        R - Recovered (naturally immune)
        V - Vaccinated (vaccine-induced immunity)
    
    Parameters:
    params : MeaslesParameters
        Parameter object containing all model parameters
    """

    def __init__(self, params):
        self.params = params
        self.N = params.population_size

        # initialize compartments
        self.compartments = ['S', 'E', 'I', 'R', 'V']

        # store simulation results
        self.results = None 
        self.time = None 

    def initial_conditions(self) -> np.ndarray:
        """Set initial conditions for the model.
        Returns:
            y0: np.ndarray. Initial values for [S, E, I, R, V]
        """
        # start with mostly susceptible population
        I0 = self.params.initial_infected
        E0 = I0 * 2  # assume 2x exposed as infected initially
        V0 = int(self.N * self.params.initial_vaccinated)
        R0 = 0  # none recovered initially
        S0 = self.N - E0 - I0 - R0 - V0
        
        return np.array([S0, E0, I0, R0, V0], dtype=float)
    
    def vaccination_rate(self, t: float) -> float:
        """Calculate time-dependent vaccination rate.
        
        The implements routine vaccination schedule
        This is the simpler alternative to behavioral decision-making.
        
        Parameters:
        t: float. Current time (days)
        
        Returns:
        vax_rate: float. Vaccination rate (per day)
        """
        # routine vaccination: vaccinate proportion of births
        # simplified: assume vaccination at age 1 year

        annual_births = self.params.birth_rate * self.N * 365
        doses_per_year = annual_births * self.params.baseline_coverage_dose1

        # convert to daily rate
        vax_rate = doses_per_year / 365 / self.N    # per capita per day

        return vax_rate 
    
    def derivatives(self, y: np.ndarray, t: float) -> np.ndarray:
        """ Calculate derivatives for SEIR-V model.
        
        Parameters:
        y : np.ndarray. Current state [S, E, I, R, V]
        t : float. Current time
        
        Returns:
        dydt : np.ndarray. Derivatives [dS/dt, dE/dt, dI/dt, dR/dt, dV/dt]
        """
        S, E, I, R, V = y
        N = S + E + I + R + V
        
        # parameters
        beta = self.params.beta
        sigma = self.params.sigma  # 1/latent_period
        gamma = self.params.gamma  # 1/infectious_period
        mu = self.params.death_rate
        nu = self.params.birth_rate
        
        # vaccination rate (time-dependent)
        vax_rate = self.vaccination_rate(t)
        
        # force of infection
        lambda_t = beta * I / N
        
        # differential equations
        dS = nu * N - lambda_t * S - vax_rate * S - mu * S
        dE = lambda_t * S - sigma * E - mu * E
        dI = sigma * E - gamma * I - mu * I
        dR = gamma * I - mu * R
        dV = vax_rate * S - mu * V
        
        return np.array([dS, dE, dI, dR, dV])
    
    def simulate(
        self, 
        t_max: float = None,
        dt: float = None,
        return_daily: bool = True
    ) -> Dict[str, np.ndarray]:
        """Run the SEIR-V simulation.

        Parameters:
        t_max: float, optional. Maximum simulation time in days
        dt: float, optional. Time step for output
        return_daily: bool. If True, return daily values; if False, return at dt intervals
        
        Returns:
        results: dict. Dictionary with keys: 'time', 'S', 'E', 'I', 'R', 'V', 'cumulative_infections'
        """
        # use default values from parameters if not provided
        if t_max is None:
            t_max = self.params.time_horizon
        if dt is None:
            dt = self.params.dt
         # Time array
        if return_daily:
            t = np.arange(0, t_max, 1.0)  # Daily output
        else:
            t = np.arange(0, t_max, dt)
        
        # Initial conditions
        y0 = self.initial_conditions()
        
        # Solve ODE
        solution = odeint(self.derivatives, y0, t)
        
        # Extract compartments
        S, E, I, R, V = solution.T
        
        # Calculate cumulative infections (new infections over time)
        N = S + E + I + R + V
        lambda_t = self.params.beta * I / N
        new_infections = np.gradient(E + I + R, t)  # Rate of new infections
        cumulative_infections = np.cumsum(np.maximum(0, new_infections))
        
        # Calculate incidence (new cases per day)
        incidence = lambda_t * S
        
        # Store results
        self.results = {
            'time': t,
            'S': S,
            'E': E,
            'I': I,
            'R': R,
            'V': V,
            'N': N,
            'incidence': incidence,
            'cumulative_infections': cumulative_infections,
            'prevalence': I / N,
            'force_of_infection': lambda_t
        }
        self.time = t
        
        return self.results
    
    def calculate_health_outcomes(self) -> Dict[str, float]:
        """Calculate health outcomes from simulation results
        
        Returns:
        outcomes: dict. Total cases, deaths, hospitalizations, etc.
        """
        if self.results is None:
            raise ValueError("Must run simulate() before calculating outcomes")
        
        # total cases (cumulative infections)
        total_cases = self.results['cumulative_infections'][-1]

        # deaths (CFR * cases)
        total_deaths = total_cases * self.params.case_fatality_rate

        # hospitalizations
        total_hospitalizations = total_cases * self.params.hospitalization_rate 

        # complications
        total_complications = total_cases * self.params.complication_rate 

        # peak prevalence
        peak_prevalence = np.max(self.results['prevalence'])
        peak_time = self.time[np.argmax(self.results['prevalence'])]

        # total person-days of infection
        person_days_infected = np.trapz(self.results['I'], self.time)

        outcomes = {
            'total_cases': total_cases,
            'total_deaths': total_deaths,
            'total_hospitalizations': total_hospitalizations,
            'total_complications': total_complications,
            'peak_prevalence': peak_prevalence,
            'peak_time_days': peak_time,
            'person_days_infected': person_days_infected,
            'final_vaccinated': self.results['V'][-1],
            'final_recovered': self.results['R'][-1],
            'attack_rate': total_cases / self.N
        }
        
        return outcomes
    
    def plot_dynamics(self, save_path: Optional[str] = None):
        """Plot epidemic dynamics over time.
        
        Parameters:
        save_path : str, optional. If provided, save figure to this path
        """
        if self.results is None:
            raise ValueError("Must run simulate() before plotting")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: All compartments
        ax = axes[0, 0]
        ax.plot(self.time / 365, self.results['S'], label='Susceptible', color='blue', linewidth=2)
        ax.plot(self.time / 365, self.results['E'], label='Exposed', color='orange', linewidth=2)
        ax.plot(self.time / 365, self.results['I'], label='Infected', color='red', linewidth=2)
        ax.plot(self.time / 365, self.results['R'], label='Recovered', color='green', linewidth=2)
        ax.plot(self.time / 365, self.results['V'], label='Vaccinated', color='purple', linewidth=2)
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Number of individuals', fontsize=12)
        ax.set_title('SEIR-V Compartment Dynamics', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Prevalence
        ax = axes[0, 1]
        ax.plot(self.time / 365, self.results['prevalence'] * 100, color='red', linewidth=2)
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Prevalence (%)', fontsize=12)
        ax.set_title('Disease Prevalence Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Incidence
        ax = axes[1, 0]
        ax.plot(self.time / 365, self.results['incidence'], color='darkred', linewidth=2)
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Daily incidence (new cases/day)', fontsize=12)
        ax.set_title('Daily Incidence', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative cases
        ax = axes[1, 1]
        ax.plot(self.time / 365, self.results['cumulative_infections'], color='darkblue', linewidth=2)
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Cumulative cases', fontsize=12)
        ax.set_title('Cumulative Infections', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print summary of simulation results."""
        if self.results is None:
            raise ValueError("Must run simulate() before printing summary")
        
        outcomes = self.calculate_health_outcomes()
        
        print("SEIR-V SIMULATION RESULTS:")
        print(f"Simulation time: {self.time[-1] / 365:.1f} years")
        print(f"Population size: {self.N:,}")
        print(f"\n--- EPIDEMIC OUTCOMES ---")
        print(f"Total cases: {outcomes['total_cases']:,.0f}")
        print(f"Attack rate: {outcomes['attack_rate'] * 100:.2f}%")
        print(f"Total deaths: {outcomes['total_deaths']:,.0f}")
        print(f"Total hospitalizations: {outcomes['total_hospitalizations']:,.0f}")
        print(f"Peak prevalence: {outcomes['peak_prevalence'] * 100:.3f}%")
        print(f"Peak time: {outcomes['peak_time_days'] / 365:.2f} years")
        print(f"\n--- FINAL STATE ---")
        print(f"Susceptible: {self.results['S'][-1]:,.0f}")
        print(f"Vaccinated: {outcomes['final_vaccinated']:,.0f}")
        print(f"Recovered: {outcomes['final_recovered']:,.0f}")
        print(f"Immune proportion: {(outcomes['final_vaccinated'] + outcomes['final_recovered']) / self.N * 100:.1f}%")
        

if __name__ == "__main__":
    # test the model
    from dataio.seirv_parameters import MeaslesParameters
    
    print("Initializing SEIR-V model...")
    params = MeaslesParameters()
    params.print_summary()
    
    print("\nRunning simulation...")
    model = SEIRVModel(params)
    results = model.simulate(t_max=365*10)  # 10 years
    
    print("\nSimulation complete!")
    model.print_summary()
    
    print("\nGenerating plots...")
    model.plot_dynamics()