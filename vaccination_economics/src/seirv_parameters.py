"""
===============================================================================
seirv_parameters.py
Author: Veronica Scerra
Last Updated: 2026-02-05
===============================================================================
Model Parameters for Measles Vaccination Economics

This script contains all epidemiological, demographic, economic, and behavioral 
parameters for the vaccination model. Parameters are sourced from peer-reviewed
literature and public health databases. 

References:
    - CDC Measles Parameters: https://www.cdc.gov/measles/
    - Guerra et al. (2017): Measles epidemiology
    - WHO Cost-Effectiveness Guidelines
    - Galvani et al. (2007): Vaccination economics
--------------------------------------------------------------------------------
License: MIT
================================================================================
"""

import numpy as np
from dataclasses import dataclass 
from typing import Dict, Optional 

@dataclass 
class MeaslesParameters:
    """
    Comprehensive parameter set for measeles SEIR-V model and economic analysis.
    
    All epidemiological parameters are per day unless otherwise noted.
    All costs are in 2024 USD
    """

    # ==================== Population Demographics ================================
    population_size: int = 1_000_000 # total population
    birth_rate: float = 12.0 / 1000 / 365   # births per person per day (~12 per 1000 per year)
    death_rate: float = 8.0 / 1000 / 365    # natural death per person per day 

    # age-structure (for age-stratified vaccination)
    age_groups: Dict[str, tuple] = None    # will be initialized in __post_init__ 

    # ==================== Measles Epidemiology ===================================
    # basic reproduction number (highly contagious)
    R0: float = 15.0    # range 12-18 in literature 
    HIT: float = 1 - (1/R0)

    # disease natural history
    latent_period: float = 8.0     # days (range 8-12)
    infectious_period: float = 10.0  # days (range 6-10) 

    # derived transmission parameters
    sigma: float = None     # 1/latent_period (progression rate from E to I)
    gamma: float = None     # 1/infectious_period (recovery rate)
    beta: float = None      # transmission rate (will be calculated from R0)

    # disease severity (in developed country context)
    hospitalization_rate: float = 0.20  # 20% of cases require hospitalization 
    case_fatality_rate: float = 0.002   # 0.2% (2 per 1000 cases)
    complication_rate: float = 0.30     # 30% experience complications 

    # ==================== Vaccination Parameters==================================
    # vaccine efficacy
    efficacy_1dose: float = 0.93    # 93% after first dose 
    efficacy_2dose: float = 0.97    # 97% after second dose

    # vaccination schedule
    age_first_dose: float = 365.0   # 12 months (in days)
    age_second_dose: float = 4 * 365.0  # 4 years (in days)

    # baseline vaccination coverage (current state)
    baseline_coverage_dose1: float = 0.925   # 92.5% (US national average)
    baseline_coverage_dose2: float = 0.88   # 88% (US national average)

    # vaccine waning (minimal for measles)
    waning_rate: float = 0.0    # per day (measles immunity is lifelong)

    # ==================== Economic Parameters ====================================
    # vaccine costs
    vaccine_dose_cost: float = 21.0     # cost per dose (CDC vaccine price list)
    administration_cost: float = 25.0   # cost to administer vaccine 
    vaccine_delivery_cost: float = None # total = dose + admin (calculated in __post_init__)

    # disease treatment costs
    outpatient_treatment_cost: float = 150.0    # mild case, outpatient
    hospitalization_cost: float = 8_000.0       # severe case, hospitalized
    complication_treatment_cost: float = 3_000.0 # SSPE, encephalitis treatment

    # outbreak response costs
    contact_tracing_cost_per_case: float = 5_000.0  # public health response
    outbreak_investigation_cost: float = 50_000.0   # per outbreak event

    # productivity losses (indirect costs)
    daily_wage: float = 200.0   # average daily wage (US)
    days_lost_mild_case: float = 5.0    # caregiver time for mild case
    days_lost_hospitalization: float = 10.0 # caregiver time for hospitalization

    # =================== Disability Weights (for DALY calculation) ===============
    # from Global Burden of Disease study
    dw_measles_mild: float = 0.051      # acute measles, mild
    dw_measles_moderate: float = 0.133  # acute measles, moderate
    dw_measles_severe: float = 0.280    # acute measles, severe
    dw_complication: float = 0.400      # long-term complications (encephalitis, SSPE)

    # duration of disability (days)
    duration_acute_illness: float = 14.0    # days symptomatic
    duration_complications: float = 90.0    # days for complications 

    # Life expectancy for YLL calculation
    life_expectancy: float = 78.0  # years (US average)

    # =================== Cost-Effectiveness Thresholds ===========================
    gdp_per_capita: float = 70_000.0 # US GDP per capita (2024 USD)

    # WHO thresholds
    ce_threshold_highly: float = None   # <1x GDP per capita (calculated)
    ce_threshold_cost_effective: float = None # 1-3x GDP per capita (calculated)

    # discount rate for future costs/benefits
    discount_rate: float = 0.03     # 3% per year (WHO standard)

    # =================== Behavioral Parameters (for phase 2 BE) ==================
    perceived_vaccine_risk: float = 0.0001  # baseline perceived risk
    perceived_disease_risk: float = None    # dynamically calculated

    risk_perception_accuracy: float = 1.0   # 1.0 = perfect accuracy

    # social influence
    social_influence_strength: float = 0.0  # 0 = no social influence (phase 1)
    network_contacts: int = 50  # average contacts per person

    # =================== Simulation Parameters ===================================
    time_horizon: int = 365 * 30    # 30 years (in days)
    dt: float = 0.1     # time step for simulation (days)

    # initial conditions
    initial_infected: int = 10  # initial number of infected individuals
    initial_vaccinated: float = 0.0 # proportion initially vaccinated (will ramp up)

    def __post_init__(self):
        """Calculate derived parameters after initialization"""
        # transmission rates
        self.sigma = 1.0 / self.latent_period
        self.gamma = 1.0 / self.infectious_period
        self.beta = self.R0 * self.gamma    # beta = R0 * gamma

        # total vaccine delivery cost
        self.vaccine_delivery_cost = self.vaccine_dose_cost + self.administration_cost 

        # cost-effectiveness thresholds
        self.ce_threshold_highly = self.gdp_per_capita  # $70,000 per DALY
        self.ce_threshold_cost_effective = 3 * self.gdp_per_capita  # $ 210,000 per DALY 

        # age groups for age-structured model
        self.age_groups = {
            'infants': (0, 1),  # 0-1 years (eligible for dose 1)
            'preschool': (1, 5),  # 1-5 years (eligible for dose 2)
            'school_age': (5, 18),  # 5-18 years
            'adults': (18, 100)  # 18+ years
        }
    
    def to_dict(self) -> Dict:
        """Convert parameters to dictionary for easy inspection."""
        return {
            'R0': self.R0,
            'HIT': self.HIT,
            'latent_period_days': self.latent_period,
            'infectious_period_days': self.infectious_period,
            'beta': self.beta,
            'sigma': self.sigma,
            'gamma': self.gamma,
            'efficacy_1dose': self.efficacy_1dose,
            'efficacy_2dose': self.efficacy_2dose,
            'vaccine_cost_per_dose': self.vaccine_delivery_cost,
            'hospitalization_cost': self.hospitalization_cost,
            'case_fatality_rate': self.case_fatality_rate,
            'ce_threshold_highly': self.ce_threshold_highly,
            'ce_threshold_cost_effective': self.ce_threshold_cost_effective
        }
    
    def print_summary(self):
        """Print parameter summary for documentation."""
        print("MEASLES VACCINATION MODEL PARAMETERS:")
        print("\n--- EPIDEMIOLOGY ---")
        print(f"R₀: {self.R0:.1f}")
        print(f"Herd Immunity Threshold: {self.HIT*100:.1f}%")
        print(f"Latent period: {self.latent_period:.1f} days")
        print(f"Infectious period: {self.infectious_period:.1f} days")
        print(f"Transmission rate (β): {self.beta:.3f} per day")
        print(f"Case fatality rate: {self.case_fatality_rate * 100:.2f}%")
        
        print("\n--- VACCINATION ---")
        print(f"Vaccine efficacy (1 dose): {self.efficacy_1dose * 100:.0f}%")
        print(f"Vaccine efficacy (2 doses): {self.efficacy_2dose * 100:.0f}%")
        print(f"Current coverage (dose 1): {self.baseline_coverage_dose1 * 100:.0f}%")
        print(f"Current coverage (dose 2): {self.baseline_coverage_dose2 * 100:.0f}%")
        
        print("\n--- ECONOMICS ---")
        print(f"Vaccine delivery cost: ${self.vaccine_delivery_cost:.2f} per dose")
        print(f"Hospitalization cost: ${self.hospitalization_cost:,.0f}")
        print(f"Highly cost-effective threshold: ${self.ce_threshold_highly:,.0f}/DALY")
        print(f"Cost-effective threshold: ${self.ce_threshold_cost_effective:,.0f}/DALY")
       

# Alternative parameter sets for sensitivity analysis
def create_low_transmission_params():
    """Create situation where R0 is lower"""
    return MeaslesParameters(
        R0 = 10.0
    )

def create_high_transmission_params():
    """Situation where R0 is higher"""
    return MeaslesParameters(
        R0 = 18
    )

def create_low_coverage_params():
    """Create low coverage scenario."""
    return MeaslesParameters(
        baseline_coverage_dose1=0.75,
        baseline_coverage_dose2=0.68,
        initial_vaccinated = .75
    )

if __name__ == "__main__":
    # Test parameter initialization
    params = MeaslesParameters()
    params.print_summary()
    
    print("\nParameter dictionary:")
    import pprint
    pprint.pprint(params.to_dict())



