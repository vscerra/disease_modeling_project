"""
===============================================================================
coupled_model.py
Author: Veronica Scerra
Last Updated: 2026-02-22
===============================================================================
Coupled SEIR-Behavioral Model

Integration of the SEIR-V epidemic model with behavioral decision-making, 
creating feedback loops between disease prevalence and vaccination behavior. 

Key concepts explored:
- prevalence-dependent vaccination descisions
- Nash equilibrium emergence
- dynamic oscillations (Bauch 2005)
- policy intervention testing
--------------------------------------------------------------------------------
License: MIT
================================================================================
"""

import numpy as np 
from scipy.integrate import solve_ivp 
from typing import Dict, Optional, Tuple 
import matplotlib.pyplot as plt 

from seirv_models import SEIRVModel 
from behavioral_model import BehavioralDecisionModel, BehavioralParameters 

class SEIRBehavioralModel(SEIRVModel):
    """SEIRV model with behavioral feedback
    
    Extension of base SEIR-V model by replacing the static vaccination rate 
    with a behavioral decision model that responds to disease prevalence. 

    This creates the key feedback loop:
    high prevalence -> more vaccination -> lower prevalence -> less vaccination

    Parameters:
        epi_params: MeaslesParameters. Epidemiological parameters
        behavioral_params: BehavioralParameters. Behavioral decision parameters
        enable_feedback: bool. if True, vaccination responds to prevalence (default), 
                                if False, use static baseline rate (Phase 1 model)
    """

    def __init__(self,
                 epi_params,
                 behavioral_params: Optional[BehavioralParameters] = None,
                 enable_feedback: bool = True):
        """Initialize coupled model"""
        # initialize parent SEIR model
        super().__init__(epi_params) 

        # initialize behavioral model
        if behavioral_params is None:
            behavioral_params = BehavioralParameters() 

        self.behavioral_model = BehavioralDecisionModel(behavioral_params) 
        self.behavioral_params = behavioral_params
        self.enable_feedback = enable_feedback 

        # storage for tracking decisions over time
        self.vaccination_probabilities = []
        self.perceived_risks = []
        self.utilities_vax = []
        self.utilities_no_vax = [] 


    def vaccination_rate_behavioral(self,
                                    S: float,
                                    I: float,
                                    V: float,
                                    N: float) -> float:
        """Calculate vaccination rate based on behavioral decisions.
        
        This is the key method that creates prevalence-dependent behavior

        Parameters:
            S: float. Susceptible population 
            I: float. Infected population
            V: float. Vaccinated population
            N: float. Total population 

        Returns:
            vax_rate: float. Per capita vaccination rate (per day)
        """
        # calculate current prevalence and coverage
        prevalence = max(I / N, 1e-10)      # to avoid division by zero
        coverage = V / N 

        # get individual vaccination probability from behavioral model 
        prob_vax = self.behavioral_model.vaccination_probability(
            prevalence,
            coverage,
            self.params.R0,
        )

        # store for tracking
        self.vaccination_probabilities.append(prob_vax)

        # convert probability to population vaccination rate
        # baseline rate: proportion of births that get vaccinated 
        target_coverage = self.params.baseline_coverage_dose1
        
        if S < 1:
            return 0.0
        
        maintenance_rate = (self.params.birth_rate * target_coverage * N) / S

        # behavioral rate: individuals in S choosing to vaccinate
        # mix of routine (births) + responsive (adults/older children)
        # For MMR: mostly routine (school mandates, pediatrician schedule)
        # for flu: mostly responsive (annual decision based on current risk)
        routine_fraction = 1.0 - self.behavioral_params.behavioral_fraction
        responsive_fraction = self.behavioral_params.behavioral_fraction

        routine_rate = routine_fraction * maintenance_rate 
        behavioral_rate = responsive_fraction * maintenance_rate * prob_vax 

        total_rate = routine_rate + behavioral_rate 
        return total_rate 
    

    def derivatives(self, y: np.ndarray, t: float) -> np.ndarray:
        """Calculate derivatives with behavioral vaccination.

        Overrides parent method to incorporate behavioral decision-making 

        Parameters:
            y: np.ndarray. Current state [S, E, I, R, V]
            t: float. Current time

        Returns:
            dydt: np.ndarray. Derivatives [dS/dt, dE/dt, dI/dt, dR/dt, dV/dt]
        """
        # ensure non-negative compartments
        S, E, I, R, V = np.maximum(y, 0)
        N = S + E + I + R + V 

        # parameters
        beta = self.params.beta
        sigma = self.params.sigma 
        gamma = self.params.gamma
        mu = self.params.death_rate 
        nu = self.params.birth_rate 

        # vaccination rate - behavioral or static
        if self.enable_feedback:
            vax_rate = self.vaccination_rate_behavioral(S, I, V, N)
        else:
            # phase 1 mode: static mode
            vax_rate = self.params.baseline_coverage_dose1 * nu 
        
        # force of infection
        lambda_t = beta * I / N 

        # differential equations
        dS = nu * N - lambda_t * S - vax_rate * S - mu * S
        dE = lambda_t * S - sigma * E - mu * E
        dI = sigma * E - gamma * I - mu * I
        dR = gamma * I - mu * R
        dV = vax_rate * S - mu * V
        
        return np.array([dS, dE, dI, dR, dV])
    

    def simulate(self,
                 t_max: float = None,
                 dt: float = None,
                 return_daily: bool = True) -> Dict[str, np.ndarray]:
        """Run simulation with behavioral feedback.
        
        Returns same format as parent but with additional behavioral tracking
        """
        # use parent simulate method
        results = super().simulate(t_max, dt, return_daily)

        # add behavioral metrics if feedback was enabled
        if self.enable_feedback and len(self.vaccination_probabilities) > 0:
            # interpolate to match output times
            n_points = len(results['time'])
            n_decisions = len(self.vaccination_probabilities)

            if n_decisions > 0:
                indices = np.linspace(0, n_decisions-1, n_points, dtype=int)
                results['vaccination_probability'] = np.array(self.vaccination_probabilities)[indices]
        return results 
    

    def run_comparison(self,
                       t_max: float = 3650,
                       scenarios: Optional[Dict] = None) -> Dict:
        """Compare static vs behavioral vaccination strategies.
        
        Parameters:
            t_max: float. Simulation time (days)
            scenarios: dict, optional. Dictionary of scenario means to BehavioralParameters
        
        Returns:
            comparison: dict. Results for each scenario
        """
        if scenarios is None:
            scenarios = {
                'Static (Phase 1)': None,   # no behavioral feedback
                'Behavioral (baseline)': BehavioralParameters(),
                'High Confidence': BehavioralParameters(confidence=0.95),
                'Free Vaccine': BehavioralParameters(vaccine_subsidy=46.0)
            }
        results = {}

        for name, params in scenarios.items():
            print(f"Running scenario: {name}")

            if params is None:
                #static
                self.enable_feedback = False 
                self.behavioral_params = BehavioralParameters()
            else:
                #behavioral mode 
                self.enable_feedback = True 
                self.behavioral_params = params 
                self.behavioral_model = BehavioralDecisionModel(params)
            
            # reset tracking
            self.vaccination_probabilities = []

            # run simulation
            sim_results = self.simulate(t_max=t_max)

            results[name] = sim_results 
        return results
    

class BehavioralDynamicsVisualizer:
    """Visualization tools for behavioral vaccination dynamics"""
    @staticmethod
    def plot_feedback_loop(results: Dict,
                          title: str = "SEIR-Behavioral Dynamics") -> None:
        """Plot coupled epidemic-behavioral dynamics.
        Shows how prevalence and vaccination behavior co-evolve.
        
        Parameters:
            results : dict. Simulation results from SEIRBehavioralModel
            title : str. Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        time_years = results['time'] / 365
        N = results['S'] + results['E'] + results['I'] + results['R'] + results['V']
        
        # Plot 1: Epidemic dynamics
        ax = axes[0, 0]
        ax.plot(time_years, results['I']/N * 100, label='Infected (I)', color='red', linewidth=2)
        ax.plot(time_years, results['E']/N * 100, label='Exposed (E)', color='orange', linewidth=2, alpha=0.7)
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Prevalence (%)', fontsize=11)
        ax.set_title('Disease Dynamics', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Vaccination coverage
        ax = axes[0, 1]
        coverage = results['V'] / N * 100
        ax.plot(time_years, coverage, color='green', linewidth=2)
        ax.axhline(y=93.3, color='red', linestyle='--', linewidth=2, 
                   label='Herd Immunity Threshold', alpha=0.7)
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Vaccination Coverage (%)', fontsize=11)
        ax.set_title('Vaccination Coverage Over Time', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Behavioral response (if available)
        ax = axes[1, 0]
        if 'vaccination_probability' in results:
            ax.plot(time_years, results['vaccination_probability'], 
                   color='blue', linewidth=2, label='Vaccination Probability')
            ax.set_ylabel('Prob(Vaccinate)', fontsize=11)
        else:
            # show static rate
            static_rate = [0.91] * len(time_years)
            ax.plot(time_years, static_rate, color='gray', linestyle='--', 
                   linewidth=2, label='Static coverage')
            ax.set_ylabel('Coverage (static)', fontsize=11)
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_title('Individual Vaccination Decisions', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Phase plane (Prevalence vs Coverage)
        ax = axes[1, 1]
        prevalence = results['I'] / N * 100
        ax.plot(coverage, prevalence, linewidth=2, color='purple', alpha=0.7)
        ax.scatter(coverage[0], prevalence[0], s=100, c='green', 
                  marker='o', label='Start', zorder=5)
        ax.scatter(coverage[-1], prevalence[-1], s=100, c='red', 
                  marker='s', label='End', zorder=5)
        ax.set_xlabel('Vaccination Coverage (%)', fontsize=11)
        ax.set_ylabel('Disease Prevalence (%)', fontsize=11)
        ax.set_title('Phase Plane: Prevalence vs Coverage', fontsize=12, fontweight='bold')
        ax.axvline(x=93.3, color='red', linestyle='--', alpha=0.5, label='HIT')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_scenario_comparison(scenario_results: Dict) -> None:
        """Compare multiple intervention scenarios.
        
        Parameters:
            scenario_results : dict. Dictionary of scenario_name -> simulation results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Cases over time
        ax = axes[0, 0]
        for name, results in scenario_results.items():
            time_years = results['time'] / 365
            N = results['S'] + results['E'] + results['I'] + results['R'] + results['V']
            cases = results['I'] / N * 100
            ax.plot(time_years, cases, label=name, linewidth=2)
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Prevalence (%)', fontsize=11)
        ax.set_title('Disease Prevalence by Scenario', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 2: Coverage over time
        ax = axes[0, 1]
        for name, results in scenario_results.items():
            time_years = results['time'] / 365
            N = results['S'] + results['E'] + results['I'] + results['R'] + results['V']
            coverage = results['V'] / N * 100
            ax.plot(time_years, coverage, label=name, linewidth=2)
        ax.axhline(y=93.3, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Vaccination Coverage (%)', fontsize=11)
        ax.set_title('Coverage by Scenario', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative cases
        ax = axes[1, 0]
        for name, results in scenario_results.items():
            time_years = results['time'] / 365
            cumulative = np.cumsum(results['incidence'])
            ax.plot(time_years, cumulative, label=name, linewidth=2)
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Cumulative Cases', fontsize=11)
        ax.set_title('Total Disease Burden by Scenario', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Final outcomes
        ax = axes[1, 1]
        scenarios = list(scenario_results.keys())
        final_cases = []
        final_coverage = []
        
        for name, results in scenario_results.items():
            N = results['S'][-1] + results['E'][-1] + results['I'][-1] + results['R'][-1] + results['V'][-1]
            final_cases.append(np.sum(results['incidence']))
            final_coverage.append(results['V'][-1] / N * 100)
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, final_cases, width, label='Total Cases', color='red', alpha=0.7)
        bars2 = ax2.bar(x + width/2, final_coverage, width, label='Final Coverage (%)', 
                       color='green', alpha=0.7)
        
        ax.set_xlabel('Scenario', fontsize=11)
        ax.set_ylabel('Total Cases', color='red', fontsize=11)
        ax2.set_ylabel('Final Coverage (%)', color='green', fontsize=11)
        ax.set_title('Final Outcomes by Scenario', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.tick_params(axis='y', labelcolor='red')
        ax2.tick_params(axis='y', labelcolor='green')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.show()


