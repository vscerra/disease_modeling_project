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
