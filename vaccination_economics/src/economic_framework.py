"""
===============================================================================
economic_framework.py
Author: Veronica Scerra
Last Updated: 2026-02-06
===============================================================================
Economic framework for vaccination cost-effectiveness analysis

This module implements cost-effectiveness analysis (CEA) for vaccination
programs, including:
- DALY calculation (disability-adjusted life years)
- ICER calculation (incremental cost-effectiveness ratio)
- Budget impact analysis
- Cost-benefit ratio 
- Net present value with discounting

Follows WHO guidelines for economic evaluation of immunization programs
--------------------------------------------------------------------------------
License: MIT
================================================================================
"""

import numpy as np 
from typing import Dict, Tuple, Optional 
import matplotlib.pyplot as plt 

class EconomicAnalysis:
    """Cost-effectiveness analysis for vaccination programs.
    
    Implements WHO-recommended methods for economic evaluation including
    DALYs, ICERs, and budget impact analysis. 
    
    Parameters:
    params: MeaslesParameters. Parameter object containing economic parameters
    """
    def __init__(self, params):
        self.params = params 
    
    def calculate_dalys(
            self,
            cases: float,
            deaths: float,
            hospitalizations: float,
            complications: float,
            time_horizon_years: float = 30,
    ) -> Dict[str, float]:
        """Calculate disability-adjusted life years (DALYs)
        
        DALYs = YLL (years of life lost) + YLD (years lived with disability)
        
        Parameters:
        cases: float. Total number of disease cases
        deaths: float. Total number of deaths
        hospitalizations: float. Number of hospitalizations
        complications: float. Number of long-term complications
        time_horizon_years: float. Time horizon for analysis (years)
        
        Returns:
        daily_breakdown: dict. YLL, YLD, and total DALYs
        """
        # Years of Life Lost (YLL)
        # assume deaths occur at average age (simplified - could be age-stratified)
        avg_age_at_death = 5.0  # years (measles deaths concentrated in children)
        years_lost_per_death = self.params.life_expectancy - avg_age_at_death 

        # apply discounting if deaths occur over time
        # simplified: assume deaths distributed uniformly over time horizon
        discount_rate = self.params.discount_rate
        discount_factor = 1 / (1 + discount_rate) ** (time_horizon_years / 2)

        YLL = deaths * years_lost_per_death * discount_factor

        # years lived with disability (YLD)
        # acute illness
        yll_acute_mild = (cases - hospitalizations - complications) * (self.params.duration_acute_illness / 365) * self.params.dw_measles_mild 

        yll_acute_severe = hospitalizations * (self.params.duration_acute_illness / 365) * self.params.dw_measles_severe 
        
        # long-term complications
        yll_complications = complications * (self.params.duration_complications / 365) * self.params.dw_complication

        YLD = yll_acute_mild + yll_acute_severe + yll_complications

        # total DALYs
        total_dalys = YLL + YLD 

        return {
            'YLL': YLL,
            'YLD': YLD,
            'total_DALYs': total_dalys,
            'YLL_per_death': years_lost_per_death,
            'YLD_acute': yll_acute_mild + yll_acute_severe,
            'YLD_complications': yll_complications
        }
    
    def calculate_costs(
            self,
            cases: float,
            deaths: float,
            hospitalizations: float,
            complications: float,
            num_vaccinations: float,
            num_outbreaks: int = 0,
            include_productivity: bool = True
    ) -> Dict[str, float]:
        """Calculate total costs of vaccination program and disease treatment.
        
        Parameters:
        cases: float. TOtal number of cases
        deaths: flaot. Total number of deaths
        hospitalizations: float. Number of hospitalizations
        complications: float. Number of complications
        num_vaccinations: float. Number of vaccine doses administered
        num_outbreaks: int. Number of outbreak events
        include_productivity: bool. Whether to include indirect costs (productivity losses)

        Returns:
        cost_breakdown: dict. Direct medical costs, indirect costs, and total costs 
        """
        # direct medical costs - Vaccination 
        vaccination_costs = num_vaccinations * self.params.vaccine_delivery_cost 

        # direct medical costs - Disease treatment
        outpatient_costs = (cases - hospitalizations) * self.params.outpatient_treatment_cost
        hospitalization_costs = hospitalizations * self.params.hospitalization_cost 
        complication_costs = complications * self.params.complication_treatment_cost 

        treatment_costs = outpatient_costs + hospitalization_costs + complication_costs

        # public health costs
        contact_tracing_costs = cases * self.params.contact_tracing_cost_per_case
        outbreak_costs = num_outbreaks * self.params.outbreak_investigation_cost 

        public_health_costs = contact_tracing_costs + outbreak_costs 

        # total direct costs
        direct_costs = vaccination_costs + treatment_costs + public_health_costs 

        # indirect costs (productivity losses)
        if include_productivity: 
            # caregiver time for mild cases
            productivity_mild = (cases - hospitalizations) * self.params.days_lost_mild_case * self.params.daily_wage 

            # caregiver time for hospitalizations
            productivity_hosp = hospitalizations * self.params.days_lost_hospitalization * self.params.daily_wage 

            # productivity loss from deaths (simplified)
            productivity_deaths = deaths * self.params.life_expectancy * 365 * self.params.daily_wage * 0.01 # heavily discounted 

            indirect_costs = productivity_mild + productivity_hosp + productivity_deaths
        else:
            indirect_costs = 0.0 
        
        # total costs
        total_costs = direct_costs + indirect_costs

        return {
            'vaccination_costs': vaccination_costs,
            'treatment_costs': treatment_costs,
            'public_health_costs': public_health_costs,
            'direct_costs': direct_costs,
            'indirect_costs': indirect_costs,
            'total_costs': total_costs,
            'cost_per_case': treatment_costs / max(cases, 1)
        }

    def calculate_icer(
            self,
            intervention_results: Dict,
            baseline_results: Dict,
            intervention_costs: Dict,
            baseline_costs: Dict, 
            perspective: str = 'societal'
    ) -> Dict[str, float]:
        """Calculate incremental cost-effectiveness ratio (ICER)
        ICER = (Cost_intervention - cost_baseline) / (DALY_baseline - DALY_intervention)

        Parameters:
        intervention_results: dict. Health outcomes under intervention
        baseline_results: dict. Health outcomes under baseline/no intervention 
        intervention_costs: dict. Costs under intervention 
        baseline_costs: dict. Costs under baseline 
        perspective: str. 'healthcare' (direct costs only) or 'societal' (direct + indirect)

        Returns:
        cea_results: dict. ICER, incremental costs, incremental DALYs, etc. 
        """
        # Calculate DALYs for both scenarios
        dalys_intervention = self.calculate_dalys(
            cases=intervention_results['total_cases'],
            deaths=intervention_results['total_deaths'],
            hospitalizations=intervention_results['total_hospitalizations'],
            complications=intervention_results['total_complications']
        )

        dalys_baseline = self.calculate_dalys(
            cases=baseline_results['total_cases'],
            deaths=baseline_results['total_deaths'],
            hospitalizations=baseline_results['total_hospitalizations'],
            complications=baseline_results['total_complications']
        )
        # get appropriate costs based on perspective
        if perspective == 'healthcare':
            cost_intervention = intervention_costs['direct_costs']
            cost_baseline = baseline_costs['direct_costs']
        else:   # societal
            cost_intervention = intervention_costs['total_costs']
            cost_baseline = baseline_costs['total_costs']

        # incremental values
        incremental_cost = cost_intervention - cost_baseline 
        incremental_dalys = dalys_baseline['total_DALYs'] - dalys_intervention['total_DALYs']

        # ICER (cost per DALY averted)
        if incremental_dalys > 0:
            icer = incremental_cost / incremental_dalys 
        else:
            icer = float('inf') if incremental_cost > 0 else float('-inf')

        # cost-effectiveness interpretation
        if icer < self.params.ce_threshold_highly:
            ce_category = 'Highly cost-effective'
        elif icer < self.params.ce_threshold_cost_effective:
            ce_category = 'Cost-effective'
        else:
            ce_category = 'Not cost-effective'

        # check if cost-saving (dominant strategy)
        cost_saving = (incremental_cost < 0 and incremental_dalys > 0)

        return {
            'ICER': icer,
            'incremental_cost': incremental_cost,
            'incremental_dalys': incremental_dalys,
            'cost_intervention': cost_intervention,
            'cost_baseline': cost_baseline,
            'dalys_intervention': dalys_intervention['total_DALYs'],
            'dalys_baseline': dalys_baseline['total_DALYs'],
            'cost_effectiveness_category': ce_category,
            'cost_saving': cost_saving,
            'cases_averted': baseline_results['total_cases'] - intervention_results['total_cases'],
            'deaths_averted': baseline_results['total_deaths'] - intervention_results['total_deaths']
        }
    
    def budget_impact_analysis(
            self, 
            intervention_costs: Dict,
            baseline_costs: Dict,
            time_horizon_years: int = 5
    ) -> Dict[str, float]:
        """Calculate budget impact over time horizon
        Parameters:
        intervention_costs: dict. Annual costs under intervention
        baseline_costs: dict. Annual costs under baseline
        time_horizon_years: int. Time horizon for budget impact (years)

        Returns:
        budget_impact: dict. Total budget impact and annual breakdown
        """
        # calculate annual incremental costs
        annual_incremental = intervention_costs['total_costs'] - baseline_costs['total_costs']

        # calculate NPV with discounting
        discount_rate = self.params.discount_rate 
        npv = 0 
        annual_costs = [] 

        for year in range(time_horizon_years):
            discounted_cost = annual_incremental / (1 + discount_rate) ** year 
            npv += discounted_cost 
            annual_costs.append(discounted_cost)

        return {
            'total_budget_impact_npv': npv, 
            'annual_undiscounted': annual_incremental,
            'annual_costs_discounted': annual_costs,
            'time_horizon_years': time_horizon_years
        }
    
    def cost_benefit_ratio(
            self,
            intervention_costs: Dict,
            baseline_costs: Dict,
            value_of_statistical_life: float = 10_000_000
    ) -> float:
        """Calculate benefit-cost ratio
        Parameters: 
        intervention_costs: dict
        baseline_costs: dict
        value_of_statistical_life: float. Monetary value of a statistical life (vSL)

        Returns:
        bcr: float. Benefit-cost ratio (>1 means benefits exceed costs)
        """
        # costs
        total_cost = intervention_costs['vaccination_costs']

        # benefits (avoided treatment costs + value of lives saved)
        avoided_treatment = baseline_costs['treatment_costs'] - intervention_costs['treatment_costs'] 

        # deaths averted
        deaths_averted = (baseline_costs['treatment_costs'] / self.params.hospitalization_cost * self.params.case_fatality_rate)
        value_lives_saved = deaths_averted * value_of_statistical_life 
        total_benefit = avoided_treatment + value_lives_saved
        bcr = total_benefit / max(total_cost, 1)

        return bcr 
    
    def print_cea_summary(self, cea_results: Dict):
        """Print formatted CEA results."""
        print("COST-EFFECTIVENESS ANALYSIS RESULTS:")
        print(f"\n--- HEALTH OUTCOMES ---")
        print(f"Cases averted: {cea_results['cases_averted']:,.0f}")
        print(f"Deaths averted: {cea_results['deaths_averted']:,.0f}")
        print(f"DALYs averted: {cea_results['incremental_dalys']:,.1f}")
        
        print(f"\n--- COSTS ---")
        print(f"Baseline costs: ${cea_results['cost_baseline']:,.0f}")
        print(f"Intervention costs: ${cea_results['cost_intervention']:,.0f}")
        print(f"Incremental cost: ${cea_results['incremental_cost']:,.0f}")
        
        print(f"\n--- COST-EFFECTIVENESS ---")
        if cea_results['cost_saving']:
            print("Status: DOMINANT STRATEGY (Cost-saving and more effective)")
        else:
            print(f"ICER: ${cea_results['ICER']:,.0f} per DALY averted")
            print(f"Category: {cea_results['cost_effectiveness_category']}")
            print(f"  Highly CE threshold: ${self.params.ce_threshold_highly:,.0f}/DALY")
            print(f"  Cost-effective threshold: ${self.params.ce_threshold_cost_effective:,.0f}/DALY")

if __name__ == "__main__":
    # Test economic analysis
    from dataio.seirv_parameters import MeaslesParameters
    
    params = MeaslesParameters()
    econ = EconomicAnalysis(params)
    
    # Simulate scenarios
    print("Testing DALY calculation...")
    dalys = econ.calculate_dalys(
        cases=10000,
        deaths=20,
        hospitalizations=2000,
        complications=3000
    )
    print(f"Total DALYs: {dalys['total_DALYs']:,.1f}")
    print(f"  YLL: {dalys['YLL']:,.1f}")
    print(f"  YLD: {dalys['YLD']:,.1f}")
    
    print("\nTesting cost calculation...")
    costs = econ.calculate_costs(
        cases=10000,
        deaths=20,
        hospitalizations=2000,
        complications=3000,
        num_vaccinations=900000
    )
    print(f"Total costs: ${costs['total_costs']:,.0f}")
    print(f"  Vaccination: ${costs['vaccination_costs']:,.0f}")
    print(f"  Treatment: ${costs['treatment_costs']:,.0f}")