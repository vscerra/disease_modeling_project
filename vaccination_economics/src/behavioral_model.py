"""
===============================================================================
behavioral_model.py
Author: Veronica Scerra
Last Updated: 2026-02-17
===============================================================================
Behavioral Economics Model for Vaccination Decisions

This module implements game-theoretic decision-making for childhood vaccination,
following Bauch & Earn (2004), Galvani et al. (2007), WHO SAGE (2015), MacDonald
et al. (2015).

Key concepts:
- WHO vaccine hesitancy framework (Confidence, Complacency, Convenience)
- Behavioral economics utility theory (risk aversion, biased perception)
- Game-theoretic equilibrium analysis (Nash vs. Utilitarian optimum)
--------------------------------------------------------------------------------
License: MIT
================================================================================
"""
import numpy as np 
from typing import Dict, Tuple, Optional
from dataclasses import dataclass 
from scipy.optimize import fsolve, minimize_scalar

@dataclass 
class BehavioralParameters:
    """Hybrid behavioral parameters combining WHO framework with behavioral economics
    Designed to be:
    - Evidence-based: WHO 3 C's have measurement tools
    - Parsimonious: avoids redundancy with economic_framework.py
    - Defensible: standard behavioral economics constructs
    - Calibratable: literature values available
    """

    # PERCEIVED risks/costs
    # note: actual costs tracked in economic_framework.py
    vaccine_perceived_cost: float = 46.0        # perceived out of pocket cost ($)
    vaccine_perceived_risk: float = 0.0001      # perceived risk of vaccine adverse event
    vaccine_perceived_severity: float = 1000.0  # perceived cost of adverse event ($)

    disease_perceived_severity: float = 50000.0 # perceived cost of infection ($)
    
    # WHO 3 C's Framework
    # evidence-based determinants from WHO SAGE (2015)
    # confidence: trust in vaccine safety, efficacy, and health system
    confidence: float = 0.9                     # scale 0-1 (0.9 = high trust)
                                                # reduces perceived vaccine risk: actual risk x (1 - confidence)

    # complacency: risk perception when disease is rare
    complacency_threshold: float = 0.001        # prevalence below which complacency occurs
    complacency_strength: float = 0.5           # how much complacency reduces perceived risk (0-1)
                                                # when prevalence < threshold: perceived_risk x (1 - complacency_strength)

    # convenience: accessibility and ease of vaccination
    convenience_factor: float = 1.0             # multiplier on perceived costs (access barriers)
                                                # <1 = easier access, >1 = harder access  

    # Behavioral Economics
    # risk aversion: curvature of utility function
    risk_aversion: float = 1.0                  # standard value
                                                # >1 = risk-averse (concave utility), <1 = risk-seeking (convex)
                                                # literature range: 0.5-4.0 (Galvani et al. 2007)
    # risk perception bias: multiplier on objective disease risk
    risk_perception_bias: float = 1.0           # 1.0 = accurate perception, >1 = overestimate disease risk, <1 = underestimate 
                                                # influenced by media effects, information quality, etc. 
    
    # Social Dynamics - for future extensions
    social_influence_weight: float = 0.1        # weight on others' behavior vs. own utility
                                                # 0 = totally individual decision, 1 = purely imitative 
    
    # Policy Parameters
    vaccine_subsidy: float = 0.0                # $ reduction in perceived cost, ex. 46.0 = fully free, 0.0 = full out-of-pocket
    information_campaign: bool = False          # whether campaign is active 
    campaign_effectiveness: float = 0.5         # how much campaign corrects bias. If active: bias -> bias x (1 - effectiveness)
                   

class BehavioralDecisionModel:
    """Calculate individual utility for vaccination decision
    
    Implements the game-theoretic framework from from Bauch & Earn (2004):
    - U(vaccinate) = -(vaccine costs) - (perceived vaccine risk)
    - U(no vaccine) = -(expected disease costs) x P(infection)

    Decision rule: vaccinate if U(vaccinate) > U(no vaccine)

    Utilities incorporated:
    - Risk aversion (curvature)
    - Perceived costs and risks 
    - WHO 3 C's 
    - Behavioral biases
    """

    def __init__(self, behavioral_params: BehavioralParameters):
        """Initialize behavioral decision model

        Parameters: 
            behavioral_params: behavioral parameters governing decision making
        """
        self.params = behavioral_params 
        self.prevalence_history = []        # store for tracking (opt)
        self.decision_history = []

    def _calculate_infection_probability(self,
                                         prevalence: float,
                                         vaccination_coverage: float,
                                         R0: float) -> float:
        """Calculate individual's probability of infection (pbjective)

        Combines current prevalence with herd immunity effects 
        Based on force of infection: lambda = Beta x I/N
        
        Parameters:
            prevalence: float. current disease prevalence (I/N)
            vaccination_coverage: float. proportion vaccinated
            R0: float. basic reproduction number 

        Returns:
            infection_prob: float. Objective probability of infection over a time period (typically 1 year)
        """
        # Approximate infection risk based on:
        # 1. current prevalence (more disease = more exposure)
        # 2. contact rate (derived from R0)
        # 3. herd immunity effect (coverage reduces individual risk)
        
        # contact rate approximation
        # R0 = β/γ where β = contact_rate × transmission_prob, for measles: infectious period ≈ 8 days, so γ = 1/8
        contacts_per_day = R0 / 8.0         # rough approximation, assumes transmission_prob ≈ 0.9 (measles is highly transmissible)
        
        # base infection risk: proportional to prevalence and contacts
        base_risk = prevalence * contacts_per_day * 0.1  # scale factor
        
        # herd immunity reduces individual risk. Coverage = 0: full risk; Coverage = HIT: minimal risk
        herd_immunity_threshold = 1 - 1/R0
        
        if vaccination_coverage >= herd_immunity_threshold:
            # Above HIT: exponential decay toward zero
            herd_factor = np.exp(-10 * (vaccination_coverage - herd_immunity_threshold))
        else:
            # Below HIT: linear interpolation
            herd_factor = 1.0 - (vaccination_coverage / herd_immunity_threshold) * 0.9
        
        objective_risk = base_risk * herd_factor
        
        return min(objective_risk, 1.0)  # Cap at 100%
    
    def perceived_infection_risk(self, 
                                prevalence: float,
                                vaccination_coverage: float,
                                R0: float) -> float:
        """Calculate PERCEIVED probability of infection (includes biases)
        
        This combines:
        1. Risk perception bias
        2. Complacency effects
        3. Information campaign corrections
        
        Parameters:
            prevalence : float. Current infection prevalence (I/N)
            R0 : float. Basic reproduction number
            vaccination_coverage : float. Current vaccination coverage (proportion)
        
        Returns:
            perceived_risk : float. Perceived probability of getting infected
        """
        # start with objective risk
        objective_risk = self._calculate_infection_probability(
            prevalence, vaccination_coverage, R0
        )
        
        # apply risk perception bias
        bias = self.params.risk_perception_bias
        
        # information campaign correction
        if self.params.information_campaign:
            # campaign moves bias toward 1.0 (accurate)
            effectiveness = self.params.campaign_effectiveness
            bias = bias + (1.0 - bias) * effectiveness
        
        perceived_risk = objective_risk * bias
        
        # complacency: when disease is rare, people underestimate risk
        if prevalence < self.params.complacency_threshold:
            complacency_reduction = self.params.complacency_strength
            perceived_risk *= (1.0 - complacency_reduction)
        
        return min(perceived_risk, 1.0)
    
    def utility_vaccinate(self) -> float:
        """Calculate utility of getting vaccinated
        U(vaccinate) = -(effective_cost^r + adverse_event_cost^r)
        
        Where r = risk_aversion parameter

        Returns:
            utility value (negative = cost, higher is better)
        """
        # effective cost after subsidies and convenience
        cost_after_subsidy = max(0, self.params.vaccine_perceived_cost - self.params.vaccine_subsidy)
        effective_cost = cost_after_subsidy * self.params.convenience_factor
        
        # Perceived adverse event risk (modulated by confidence): High confidence → low perceived risk
        adverse_event_prob = (self.params.vaccine_perceived_risk * (1.0 - self.params.confidence))
        adverse_event_cost = (adverse_event_prob * self.params.vaccine_perceived_severity)
        
        # Risk-averse utility function: U = -cost^r
        r = self.params.risk_aversion
        
        # Handle edge cases
        if effective_cost == 0 and adverse_event_cost == 0:
            utility = 0.0
        else:
            # Apply risk aversion to both cost components
            utility = -(effective_cost ** r + adverse_event_cost ** r)
        
        return utility 
    
    def utility_no_vaccinate(self,
                           prevalence: float,
                           vaccination_coverage: float,
                           R0: float) -> float:
        """Calculate utility of not getting vaccinated.
        Parameters:
            prevelance: float. current disease prevalence in population
            vaccination_coverage: float. current vaccination coverage
            R0: float. basic reproduction number
        Returns:
            utility: float. Utility of not vaccinating (negative = expected cost, higher is better)
        """
        # perceived infection risk
        infection_risk = self.perceived_infection_risk(prevalence, vaccination_coverage, R0)
        
        # expected disease cost
        expected_cost = infection_risk * self.params.disease_perceived_severity
        
        # risk-averse utility: U = -(expected_cost)^r
        r = self.params.risk_aversion
        
        if expected_cost == 0:
            utility = 0.0
        else:
            utility = -(expected_cost ** r)
        
        return utility 
    
    
    
    def vaccination_rate(self,
                        prevalence: float,
                        vaccination_coverage: float,
                        R0: float,
                        baseline_rate: float,
                        susceptible_pop: float) -> float:
        """Calculate population-level vaccination rate based on individual decisions.
        This is what gets plugged into the SEIR model as the vaccination rate.
        
        Parameters:
            prevalence : float. Current infection prevalence (I/N)
            R0 : float. Basic reproduction number
            vaccination_coverage : float. Current proportion vaccinated (V/N)
            baseline_rate : float. Baseline vaccination rate (e.g., from routine schedule)
            susceptible_pop : float. Number of susceptible individuals
        
        Returns:
            vax_rate : float. Vaccination rate (people per day)
        """
        # Get probability individual chooses vaccination
        prob_vax = self.vaccination_probability(prevalence, vaccination_coverage, R0)
        
        # Convert to population rate: mix behavioral decision with baseline routine vaccination
        behavioral_weight = 0.8                         # How much behavior matters (80%) vs routine (20%)
        effective_prob = (behavioral_weight * prob_vax + (1 - behavioral_weight) * vaccination_coverage)
        
        # Rate is proportion of susceptibles choosing vaccination. Scale by baseline rate to maintain realistic timescales
        rate = effective_prob * baseline_rate
        
        return rate
    
    def vaccination_probability(self,
                               prevalence: float,
                               vaccination_coverage: float,
                               R0: float) -> float:
        """Calculate the probability of vaccination (for population-level modeling)
        Uses logistic choice function (quantal response equilibrium) rather than binary decision:
        P(vax) = 1 / (1 + exp(-ΔU/λ))
        
        Where:
        - ΔU = U(vax) - U(no vax)
        - λ = temperature parameter (decision noise)
        This represents heterogeneity in the population

        Parameters:
            prevalence: float. Current disease prevalence
            vaccination_coverage: float. Current vaccination coverage
            R0: float. Basic reproduction number
            
        Returns:
            prob_vaccinate: float. Probability of vaccination (0-1)
        """
        u_vax = self.utility_vaccinate()
        u_no_vax = self.utility_no_vaccinate(prevalence, vaccination_coverage, R0)

        # utility difference
        delta_u = u_vax - u_no_vax 

        # logistic choice (quantal response) Temperature parameter: higher = more noise/randomness
        temperature = 100.0
        prob = 1.0 / (1.0 + np.exp(-delta_u / temperature))

        # apply social influence: people are influenced by what others do
        if self.params.social_influence_weight > 0:
            weight = self.params.social_influence_weight
            prob = weight * vaccination_coverage + (1 - weight) * prob
        
        return np.clip(prob, 0.0, 1.0)
 
 
