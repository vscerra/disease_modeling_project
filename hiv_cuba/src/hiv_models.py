"""
===========================================================
hiv_models.py
Author: Veronica Scerra
Last Updated: 2026-01-08
===========================================================
Compartmental model (SEIR-type with contact tracing) with 
compartments for:
S: susceptible
H: HIV-positive (untreated)
T: HIV-positive (under treatment/contact tracing)
A: AIDS patients
D: Deaths from AIDS

Control parameters: 

tau: Contact tracing rate control parameter. Determines
how quickly HIV+ individuals are identified and treated

beta: Transmission rate. Controls epidemic spread velocity

alpha and alpha_T: Disease progression rates. Determine time
from HIV to AIDS for untreated vs. treated individuals

delta: AIDS mortality rate. Influences death outcomes

mu: Natural birth/death rate (per year)

-----------------------------------------------------------
License: MIT
===========================================================
"""

import numpy as np 
from scipy.integrate import odeint

def hiv_model(y, t, beta, alpha, alpha_T, tau, delta, mu, N):
    """
    HIV/AIDS epidemiological model with contact tracing
    
    Parameters:
    y: array-like state vector [S, H, T, A, D]
    t: float. Time point
    beta: float. transmission rate (per year)
    alpha: float. HIV to AIDS progression rate (untreated, per year)
    alpha_T: float. HIV to AIDS progression rate (treated, per year)
    tau: float. Contact tracing/treatment initiation rate (per year)
    delta: float. AIDS mortality rate (per year)
    mu: float. Natural birth/death rate (per year)
    N: float. total population

    Returns:
    list: derivatives [dS/dt, dH/dt, dT/dt, dA/dt, dD/dt]

    Notes:
    The model assumes:
    - homogenous mixing in the population
    - only treated HIV-positive individuals (H) contribute to transmission
    - Treated individuals (T) have reduced progression to AIDS
    - Natural mortality affects all compartments
    """
    S, H, T, A, D = y
    
    # force of infection (only untreated H contribute to transmission)
    lambda_t = beta * H / N
    
    # differential equations
    dS = -lambda_t * S + mu * N - mu * S
    dH = lambda_t * S - alpha * H - tau * H - mu * H 
    dT = tau * H - alpha_T * T - mu * T 
    dA = alpha * H + alpha_T * T - delta * A - mu * A
    dD = delta * A
    
    return [dS, dH, dT, dA, dD]

def hiv_model_with_control(y, t, beta, alpha, alpha_T, tau, delta, mu, N, 
                           control_start_time, enhanced_tau):
    """
    HIV/AIDS model with time-dependent control mechanism.
    
    This variant implements enhanced contact tracing after a specified time,
    allowing simulation of intervention scenarios.
    
    Parameters:
    y : array-like. State vector [S, H, T, A, D]
    t : float. Time point
    beta : float. Transmission rate (per year)
    alpha : float. HIV to AIDS progression rate for untreated individuals (per year)
    alpha_T : float. HIV to AIDS progression rate for treated individuals (per year)
    tau : float. Baseline contact tracing rate (per year)
    delta : float. AIDS mortality rate (per year)
    mu : float. Natural birth/death rate (per year)
    N : float. Total population size
    control_start_time : float. Time point when enhanced control begins
    enhanced_tau : float. Enhanced contact tracing rate after control_start_time (per year)
    
    Returns: 
    list. Derivatives [dS/dt, dH/dt, dT/dt, dA/dt, dD/dt]
    
    Examples
    --------
    >>> # Simulate with 2x enhanced tracing starting at year 11
    >>> y0 = [1e7, 100, 0, 5, 2]
    >>> t = np.linspace(0, 20, 100)
    >>> sol = odeint(hiv_model_with_control, y0, t,
    ...              args=(0.5, 0.1, 0.05, 0.1, 0.3, 0.01, 1e7, 11, 0.2))
    """
    S, H, T, A, D = y
    
    # apply enhanced contact tracing after control period
    current_tau = enhanced_tau if t >= control_start_time else tau
    
    # force of infection 
    lambda_t = beta * H / N

    # differential equations
    dS = -lambda_t * S + mu * N - mu * S
    dH = lambda_t * S - alpha * H - current_tau * H - mu * H
    dT = current_tau * H - alpha_T * T - mu * T
    dA = alpha * H + alpha_T * T - delta * A - mu * A
    dD = delta * A
    
    return [dS, dH, dT, dA, dD]


def simulate_model(params, t, y0, N):
    """
    Simulate the HIV/AIDS model with given parameters 
    
    Parameters:
    params: array-like. Model parameters [beta, alpha, alpha_T, tau, delta, mu]
    t: array-like. Time points for simulation
    y0: array-like. Initial conditions [S0, H0, T0, A0, D0]
    N: float. Total population size
    
    Returns: 
    ndarray: solution array with shape (len(t), 5) containing [S, H, T, A, D] 
    for each timepoint
    """
    beta, alpha, alpha_T, tau, delta, mu = params
    solution = odeint(hiv_model, y0, t, args=(beta, alpha, alpha_T, tau, delta, mu, N))
    return solution


def calculate_r0(beta, alpha, tau, mu):
    """
    Calculate the basic reproduction number (R0).

    R0 represents the average number of secondary infections caused by 
    one infected individual in a competely susceptible population

    Parameters:
    beta: float. Transmission rate
    alpha: float. Disease progression rate
    tau: float. Contact tracing rate 
    mu: float. Natural mortality rate

    Returns:
    float. Basic reproduction number 

    Notes:
    R0 = beta / (alpha + tau + mu)
    if R0 > 1, the infection will spread in the population
    if R0 < 1, the infection will die out 
    """
    return beta / (alpha + tau + mu) 


def calculate_derived_quantities(params):
    """
    Calculate epidemiologically meaningful derived quantities from parameters. 

    Parameters: 
    params: dict. Dictionary with keys: 'beta', 'alpha', 'alpha_T', 'tau', 'delta', 'mu'

    Returns:
    dict containing:
        -'R0': basic reproduction number 
        -'mean_incubation_untreated': mean time from HIV to AIDS (untreated)
        -'mean_incubation_treated': meain time from HIV to AIDS (treated)
        -'mean_survival_aids': mean survival time with AIDS
        -'tracing_probability': probability of being contact traced
        -'treatment_benefit': ratio of treated to untreated incubation times
    """
    beta = params['beta']
    alpha = params['alpha']
    alpha_T = params['alpha_T']
    tau = params['tau']
    delta = params['delta']
    mu = params['mu']

    results = {
        'R0': calculate_r0(beta, alpha, tau, mu),
        'mean_incubation_untreated': 1 / alpha,
        'mean_incubation_treated': 1 / alpha_T,
        'mean_survival_aids': 1/ delta,
        'tracing_probability': tau / (tau + alpha + mu) if (tau + alpha + mu) > 0 else 0,
    }

    results['treatment_benefit'] = (results['mean_incubation_treated'] / results['mean_incubation_untreated'])
    return results 


def objective_function(params, t_data, y0, N, df):
    """
    Objective function for parameter optimization. 

    Calculates the weighted sum of squared errors between model predictions
    and observed data for HIV cases, AIDS cases, and deaths

    Parameters:
    params: array-like. Model parameters [beta, alpha, alpha_T, tau, delta, mu]
    t_data: array-like. Time points corresponding to observed data 
    y0: array-like. Initial conditions
    N: float. Total population size 
    df: pandas.DataFrame. Dataframe with columns 'HIV_cases', 'AIDS_cases', 'Deaths'

    Returns: 
    float Total weighted error 

    Notes: 
    Returns a very large error if simulation fails, ensuring the optimizer avoids invalid 
    parameter combinations
    """
    try:
        beta, alpha, alpha_T, tau, delta, mu = params 

        # simulate model
        solution = odeint(hiv_model, y0, t_data, args=(beta, alpha, alpha_T, tau, delta, mu, N))

        S, H, T, A, D = solution.T 

        # calculate new cases per year (annual incidence)
        new_HIV = np.diff(H + T, prepend=H[0] + T[0])
        new_AIDS = np.diff(A, prepend=A[0])
        new_Deaths = np.diff(D, prepend=D[0])

        # observed data
        obs_HIV = df['HIV_cases'].values
        obs_AIDS = df['AIDS_cases'].values
        obs_Deaths = df['Deaths'].values 

        # weighted sum of squared errors (normalized by mean)
        # this prevents any single variable from dominating the fit
        error_HIV = np.sum((new_HIV - obs_HIV)**2) / np.mean(obs_HIV**2)
        error_AIDS = np.sum((new_AIDS - obs_AIDS)**2) / np.mean(obs_AIDS**2)
        error_Deaths = np.sum((new_Deaths - obs_Deaths)**2) / np.mean(obs_Deaths**2)

        total_error = error_HIV + error_AIDS + error_Deaths 
        return total_error 
    except: 
        # return large error if simulation fails
        return 1e10 
    

def calculate_new_cases(solution, prepend_first=True):
    """
    Calculate annual new cases from cumulative compartment values 

    Parameters:
    solution: ndarray. Solution array from odeint with shape (n_time, 5)
    prepend_first: bool, optional. If true, prepends the first value to differences (default: True)

    Returns:
    dict: Dictionary with keys 'HIV', 'AIDS', 'Deaths' containing annual new cases
    """ 
    S, H, T, A, D = solution.T

    # calculate new cases (differences between time points)
    if prepend_first:
        new_HIV = np.diff(H + T, prepend=H[0] + T[0])
        new_AIDS = np.diff(A, prepend=A[0])
        new_Deaths = np.diff(D, prepend=D[0])
    else:
        new_HIV = np.diff(H + T)
        new_AIDS = np.diff(A)
        new_Deaths = np.diff(D)

    return {
        'HIV': new_HIV,
        'AIDS': new_AIDS,
        'Deaths': new_Deaths
    }


def calculate_r_squared(observed, predicted):
    """
    Calculate the coefficient of determination (R**2)

    Parameters:
    observed: array-like. Observed data values
    predicted: array-like. Prediced/fitted values

    Returns:
    float: R**2 value (1.0 is perfect fit, 0.0 is no better than mean)

    Notes:
    R**2 = 1 - (SS_res / SS_tot)
    where SS_res is sum of squared residuals and SS_tot is total sum of squares
    """
    ss_res = np.sum((observed - predicted)**2)
    ss_tot = np.sum((observed - np.mean(observed))**2)
    return 1 - (ss_res / ss_tot)


def project_scenarios(y0, t, params, N, control_start_time, enhancement_factors):
    """
    Project multiple control scenarios with different enhancement levels.

    Parameters:
    y0: array-like. Initial conditions
    t: array-like. Time points for projection
    params: dict. Model parameters with keys: beta, alpha, alpha_T, tau, delta, mu
    N: float. Total population
    control_start_time: float. When enhancement control begins
    enhancement_factors: list. List of enhancement multipliers

    Returns:
    dict: Map of enhancement factors to solution array
    """

    beta = params['beta']
    alpha = params['alpha']
    alpha_T = params['alpha_T']
    tau = params['tau']
    delta = params['delta']
    mu = params['mu']

    scenarios = {}

    for factor in enhancement_factors:
        if factor == 1:
            # baseline scenario (no enhancement)
            sol = odeint(hiv_model, y0, t, args=(beta, alpha, alpha_T, tau, delta, mu, N))
            scenarios['baseline'] = sol 
        else:
            # enhanced scenarios
            enhanced_tau = tau * factor
            sol = odeint(hiv_model_with_control, y0, t, args=(beta, alpha, alpha_T, tau, delta, mu, N, control_start_time, enhanced_tau))
            scenarios[f'{factor}x'] = sol 
    return scenarios 


def calculate_impact_metrics(baseline_solution, intervention_solution, compartment_index, time_index):
    """
    Calculate impact metrics comparing intervention to baseline.
    
    Parameters:
    baseline_solution : ndarray. Baseline scenario solution
    intervention_solution : ndarray. Intervention scenario solution
    compartment_index : int. Index of compartment to analyze (3 for AIDS, 4 for Deaths)
    time_index : int or slice. Time point(s) to evaluate
    
    Returns:
    dict: Dictionary with 'absolute_reduction', 'percent_reduction', 
        'baseline_value', 'intervention_value'
    """
    baseline_val =  baseline_solution[time_index, compartment_index]
    intervention_val = intervention_solution[time_index, compartment_index]

    absolute_reduction = baseline_val - intervention_val 
    percent_reduction = (absolute_reduction / baseline_val * 100) if baseline_val > 0 else 0

    return {
        'absolute_reduction': absolute_reduction,
        'percent_reduction': percent_reduction,
        'baseline_value': baseline_val,
        'intervention_val': intervention_val
    }
