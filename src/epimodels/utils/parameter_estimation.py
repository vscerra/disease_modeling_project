"""
===========================================================
parameter_estimation.py
Author: Veronica Scerra
Last Updated: 2026-01-19
===========================================================
Parameter estimation for yellow fever model
============================================

This module provides tools for fitting the Yellow Fever SEIR
model to observed outbreak data using Maximum Likelihood 
Estimation (MLE) and Bayesian methods

License: MIT
===========================================================
"""
import numpy as np 
import pandas as pd 
from scipy.optimize import minimize, differential_evolution 
from scipy.stats import poisson, norm 
from typing import Dict, Tuple, List, Callable, Optional 
import warnings 
import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from yellow_fever_models import YellowFeverModel, create_vaccination_function

class ParameterEstimator:
    """
    Parameter estimation for Yellow Fever SEIR model

    Supports:
    - Maximum Likelihood Estimation (MLE)
    - Least Squares fitting 
    - Bayesian inference with MCMC (optional)
    """
    def __init__(
            self, 
            data: pd.DataFrame,
            population: float = 800000,
            fixed_params: Optional[Dict[str, float]] = None 
    ):
        """Initialize parameter estimator:
        Parameters:
        data: DataFrame. Observed data with columns ['date', 'days_since_start', 'cases_cumulative', 'deaths_cumulative'] 
        population: float. Total population size 
        fixed_params: dict, optional. Parameters to hold fixed during optimization
        """
        self.data = data 
        self.N = population 
        self.fixed_params = fixed_params or {} 

        # extracted time series
        self.t_data = data['days_since_start'].values 
        self.cases_data = data['cases_cumulative'].values 
        self.deaths_data = data['deaths_cumulative'].values 

    def _create_model(self, params: Dict[str, float]) -> YellowFeverModel:
        """ Create model instance from parameter dictionary
        Parameters:
        params: dict. Parameter values 
        
        Returns:
        model: YellowFeverModel. Configured model instance
        """
        # merge with fixed parameters
        all_params = {**self.fixed_params, **params}

        # create vaccination function if parameters provided
        if 'vaccination_start' in all_params and 'vaccination_rate' in all_params:
            vaccination_func = create_vaccination_function(
                start_date=all_params['vaccination_start'],
                vaccination_rate=all_params['vaccination_rate'],
                ramp_duration=7.0
            )
        else:
            vaccination_func = None 
        
        model = YellowFeverModel(
            beta=all_params['beta'],
            sigma=all_params['sigma'],
            gamma=all_params['gamma'],
            alpha=all_params['alpha'],
            vaccination_func=vaccination_func,
            population=self.N
        )
        
        return model

    def simulate_model(
            self,
            params: Dict[str, float],
            initial_infected: float = 10
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Simulate model and extract cumulative cases and deaths.
            Parameters:
            params : dict. Model parameters
            initial_infected : float. Initial number of infectious individuals
                
            Returns:
            cases_cumulative : ndarray. Cumulative cases at data time points
            deaths_cumulative : ndarray. Cumulative deaths at data time points
            """
            model = self._create_model(params)
            
            # initial conditions
            initial_conditions = {
                'S': self.N - initial_infected,
                'E': 0,
                'I': initial_infected,
                'R': 0,
                'V': 0,
                'D': 0
            }
            
            # simulate
            t_max = self.t_data.max()
            t, y = model.simulate(
                initial_conditions=initial_conditions,
                t_span=(0, t_max),
                t_eval=self.t_data
            )
            
            # extract compartments
            S, E, I, R, V, D = y
            
            # cumulative cases = E + I + R + D (everyone who has been infected)
            cases_cumulative = E + I + R + D
            
            # cumulative deaths
            deaths_cumulative = D
            
            return cases_cumulative, deaths_cumulative

    def negative_log_likelihood_poisson(
            self,
            param_vector: np.ndarray,
            param_names: List[str],
            initial_infected: float = 10
    ) -> float:
        """ Calculate negative log-likelihood assuming Poisson distributed observations.
        Parameters:
        param_vector : ndarray. Parameter values as array
        param_names : list. Names of parameters in param_vector
        initial_infected : float. Initial infectious individuals
                
        Returns:
        neg_log_likelihood : float. Negative log-likelihood value
        """
        # convert vector to dictionary
        params = dict(zip(param_names, param_vector))
            
        # check parameter bounds
        if not self._check_bounds(params):
            return 1e10  # Large penalty for out-of-bounds
            
        try:
            # simulate model
            cases_pred, deaths_pred = self.simulate_model(params, initial_infected)
                
            # avoid zeros (add small constant)
            cases_pred = np.maximum(cases_pred, 0.1)
            deaths_pred = np.maximum(deaths_pred, 0.1)
                
            # Poisson log-likelihood for cases
            ll_cases = poisson.logpmf(self.cases_data, cases_pred).sum()
                
            # Poisson log-likelihood for deaths
            ll_deaths = poisson.logpmf(self.deaths_data, deaths_pred).sum()
                
            # Total log-likelihood
            log_likelihood = ll_cases + ll_deaths
                
            # Return negative (for minimization)
            return -log_likelihood
                
        except Exception as e:
                warnings.warn(f"Error in likelihood calculation: {e}")
        return 1e10

    def sum_of_squares(
            self, 
            param_vector: np.ndarray,
            param_names: List[str],
            initial_infected: float=10,
            weights: Optional[Dict[str, float]] = None 
    )-> float:
        """ Calculate the weighted sum of squared errors 
        Parameters: 
        param_vector: ndarray. Parameter values 
        param_names: list. Parameter names 
        initial_infected: float. Initial infectious 
        weights: dict, optional. Weights for cases and deaths {'cases': w1, 'deaths': w2}
        
        Returns:
        sse: float. Weighted sum of squared errors
        """
        params = dict(zip(param_names, param_vector)) 
        if not self._check_bounds(params):
            return 1e10 
        try:
            cases_pred, deaths_pred = self.simulate_model(params, initial_infected)
            
            # default equal weights
            if weights is None:
                weights = {'cases': 1.0, 'deaths': 1.0}
            
            # calculate SSE
            sse_cases = weights['cases'] * np.sum((self.cases_data - cases_pred)**2)
            sse_deaths = weights['deaths'] * np.sum((self.deaths_data - deaths_pred)**2) 
            return sse_cases + sse_deaths 
        
        except Exception:
            return 1e10 

    def _check_bounds(self, params: Dict[str, float]) -> bool:
        """Check if parameters are within reasonable bounds."""
        bounds = {
            'beta': (0.001, 10.0),
            'sigma': (0.01, 1.0),
            'gamma': (0.01, 1.0),
            'alpha': (0.0, 0.5),
            'vaccination_rate': (0.0, 0.1),
            'vaccination_start': (0, 400)
        }
        
        for name, value in params.items():
            if name in bounds:
                lower, upper = bounds[name]
                if not (lower <= value <= upper):
                    return False
        return True
    
    def fit_mle(
            self,
            initial_guess: Dict[str, float],
            method: str='L-BFGS-B',
            use_global: bool=False
    )-> Tuple[Dict[str, float], Dict]:
        """Fit model paramters using Maximum Likelihood Estimation
        Parameters:
        initial_guess: dict. INitial parameter values 
        method: str. Optimization method ('L-BFGS-B', 'Nelder-Mead', 'Powell')
        use_global: bool. Whether to use global optimization (differential_evolution)
        
        Returns:
        fitted_params: dict. Fitted parameter values
        result: dict. Optimization result information
        """

        param_names = list(initial_guess.keys())
        initial_vector = np.array([initial_guess[name] for name in param_names])

        # define bounds
        bounds_dict = {
            'beta': (0.1, 5.0),
            'sigma': (1/14, 1/3),  # incubation 3-14 days
            'gamma': (1/14, 1/3),  # recovery 3-14 days
            'alpha': (0.0, 0.3),   # CFR up to 30%
            'vaccination_rate': (0.0, 0.05),
            'vaccination_start': (250, 270)  # Around Oct 1
        }
        bounds = [bounds_dict.get(name, (0, 10)) for name in param_names]
        
        if use_global:
            # global optimization
            result = differential_evolution(
                func=lambda x: self.negative_log_likelihood_poisson(x, param_names),
                bounds=bounds,
                seed=42,
                maxiter=1000,
                atol=1e-6,
                tol=1e-6,
                disp=True
            )
        else:
            # local optimization
            result = minimize(
                fun=lambda x: self.negative_log_likelihood_poisson(x, param_names),
                x0=initial_vector,
                method=method,
                bounds=bounds,
                options={'disp': True, 'maxiter': 10000}
            )
        
        # extract fitted parameters
        fitted_params = dict(zip(param_names, result.x))
        
        # calculate AIC and BIC
        n_data = len(self.cases_data) + len(self.deaths_data)
        k = len(param_names)
        neg_ll = result.fun
        
        aic = 2 * k + 2 * neg_ll
        bic = k * np.log(n_data) + 2 * neg_ll
        
        result_info = {
            'success': result.success,
            'neg_log_likelihood': neg_ll,
            'AIC': aic,
            'BIC': bic,
            'n_iterations': result.nfev if hasattr(result, 'nfev') else None,
            'message': result.message if hasattr(result, 'message') else None
        }
        
        return fitted_params, result_info
    
    def fit_least_squares(
        self,
        initial_guess: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], Dict]:
        """ Fit model parameters using weighted least squares.
        Parameters:
        initial_guess : dict. Initial parameter values
        weights : dict, optional. Weights for cases and deaths
            
        Returns:
        fitted_params : dict. Fitted parameter values
        result : dict. Optimization result
        """
        param_names = list(initial_guess.keys())
        initial_vector = np.array([initial_guess[name] for name in param_names])
        
        bounds_dict = {
            'beta': (0.1, 5.0),
            'sigma': (1/14, 1/3),
            'gamma': (1/14, 1/3),
            'alpha': (0.0, 0.3),
            'vaccination_rate': (0.0, 0.05),
            'vaccination_start': (250, 270)
        }
        bounds = [bounds_dict.get(name, (0, 10)) for name in param_names]
        
        result = minimize(
            fun=lambda x: self.sum_of_squares(x, param_names, weights=weights),
            x0=initial_vector,
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True}
        )
        
        fitted_params = dict(zip(param_names, result.x))
        
        result_info = {
            'success': result.success,
            'sse': result.fun,
            'rmse': np.sqrt(result.fun / (len(self.cases_data) + len(self.deaths_data)))
        }
        
        return fitted_params, result_info
    
    def calculate_confidence_intervals(
        self,
        fitted_params: Dict[str, float],
        alpha: float = 0.05
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals using Fisher Information (approximate).
        Parameters:
        fitted_params : dict. Fitted parameter values
        alpha : float. Significance level (default: 0.05 for 95% CI)
            
        Returns:
        confidence_intervals : dict. Confidence intervals for each parameter
        """
        # this is a placeholder - proper implementation should use
        # numerical Hessian or bootstrap
        warnings.warn("Confidence intervals are approximate")
        
        ci = {}
        for name, value in fitted_params.items():
            # rough estimate: ±20% of parameter value
            margin = 1.96 * 0.2 * abs(value)
            ci[name] = (value - margin, value + margin)
        
        return ci
    
def calculate_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate goodness-of-fit metrics.
    Parameters:
    y_true : ndarray. Observed data
    y_pred : ndarray. Model predictions
        
    Returns:
    metrics : dict. Dictionary of metric values
    """
    residuals = y_true - y_pred
    
    # Mean Absolute Error
    mae = np.mean(np.abs(residuals))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean(residuals**2))
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs(residuals / (y_true + 1e-10))) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r_squared,
        'MAPE': mape
    }