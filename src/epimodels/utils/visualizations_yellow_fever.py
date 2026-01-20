"""
===========================================================
visualizations_yellow_fever.py
Author: Veronica Scerra
Last Updated: 2026-01-20
===========================================================
Visualization module for yellow fever model
============================================

Plots for model results, data fitting, and scenario 
comparisons

License: MIT
===========================================================
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from typing import Dict, List, Optional, Tuple 
from datetime import datetime, timedelta 

# set style 
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 150 
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['font.sans-serif'] = ['Arial'] 

class YellowFeverVisualizer:
    """Visualization tools for Yellow Fever epidemic modeling"""

    def __init__(self, start_date: str = '2002-10-04'):
        """Initialize visualizer
        Parameters:
        start_date: str. start date for converting days into dates (YYYY-MM-DD)
        """
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.colors = {
            'S': '#1f77b4',  # Blue
            'E': '#ff7f0e',  # Orange
            'I': '#d62728',  # Red
            'R': '#2ca02c',  # Green
            'V': '#9467bd',  # Purple
            'D': '#8c564b',  # Brown
            'observed': '#000000',  # Black
            'fitted': '#e377c2'  # Pink
        }
    
    def days_to_dates(self, days: np.ndarray) -> List[datetime]:
        """convert days to datetime objects"""
        return [self.start_date + timedelta(days=int(d)) for d in days] 
    
    def plot_data_overview(
        self,
        data: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot overview of observed data.
        Parameters:
        data : DataFrame. Outbreak data
        save_path : str, optional. Path to save figure
            
        Returns:
        fig : Figure. Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # convert days to dates
        dates = self.days_to_dates(data['days_since_start'])
        
        # cumulative cases
        ax = axes[0, 0]
        ax.plot(dates, data['cases_cumulative'], 'o-', 
                color=self.colors['I'], linewidth=2, markersize=8, label='Cumulative Cases')
        ax.set_ylabel('Cumulative Cases')
        ax.set_title('Yellow Fever Cases - Senegal 2002')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # cumulative deaths
        ax = axes[0, 1]
        ax.plot(dates, data['deaths_cumulative'], 's-',
                color=self.colors['D'], linewidth=2, markersize=8, label='Cumulative Deaths')
        ax.set_ylabel('Cumulative Deaths')
        ax.set_title('Yellow Fever Deaths - Senegal 2002')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # daily incidence (if available)
        if 'cases_daily' in data.columns:
            ax = axes[1, 0]
            ax.bar(dates, data['cases_daily'], color=self.colors['I'], 
                   alpha=0.7, label='Daily Cases')
            ax.set_ylabel('Daily Cases')
            ax.set_xlabel('Date')
            ax.set_title('Daily Incidence')
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # case fatality rate (CFR)
        ax = axes[1, 1]
        cfr = (data['deaths_cumulative'] / data['cases_cumulative'] * 100).fillna(0)
        ax.plot(dates, cfr, 'd-', color='darkred', 
                linewidth=2, markersize=8, label='CFR')
        ax.set_ylabel('Case Fatality Rate (%)')
        ax.set_xlabel('Date')
        ax.set_title('Case Fatality Rate Over Time')
        ax.axhline(y=cfr.iloc[-1], linestyle='--', color='gray', 
                   label=f'Final CFR: {cfr.iloc[-1]:.1f}%')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_model_fit(
        self,
        data: pd.DataFrame,
        t_model: np.ndarray,
        y_model: np.ndarray,
        show_compartments: bool = False,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot model fit to data.
        Parameters:
        data : DataFrame. Observed data
        t_model : ndarray. Model time points
        y_model : ndarray. Model solution [S, E, I, R, V, D]
        show_compartments : bool. Whether to show all compartments
        save_path : str, optional. Path to save figure
            
        Returns:
        fig : Figure
        """
        if show_compartments:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        dates_data = self.days_to_dates(data['days_since_start'])
        dates_model = self.days_to_dates(t_model)
        
        S, E, I, R, V, D = y_model
        cases_model = E + I + R + D
        
        # cases fit
        ax = axes[0]
        ax.plot(dates_data, data['cases_cumulative'], 'o', 
                color=self.colors['observed'], markersize=10, 
                label='Observed Cases', zorder=10)
        ax.plot(dates_model, cases_model, '-', 
                color=self.colors['fitted'], linewidth=2.5, 
                label='Model Fit', alpha=0.8)
        ax.set_ylabel('Cumulative Cases', fontsize=12)
        ax.set_title('Model Fit: Cumulative Cases', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # deaths fit
        ax = axes[1]
        ax.plot(dates_data, data['deaths_cumulative'], 's', 
                color=self.colors['observed'], markersize=10, 
                label='Observed Deaths', zorder=10)
        ax.plot(dates_model, D, '-', 
                color=self.colors['D'], linewidth=2.5, 
                label='Model Fit', alpha=0.8)
        ax.set_ylabel('Cumulative Deaths', fontsize=12)
        ax.set_title('Model Fit: Cumulative Deaths', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        if show_compartments:
            # all compartments
            ax = axes[2]
            ax.plot(dates_model, S, label='Susceptible', linewidth=2, color=self.colors['S'])
            ax.plot(dates_model, E, label='Exposed', linewidth=2, color=self.colors['E'])
            ax.plot(dates_model, I, label='Infectious', linewidth=2, color=self.colors['I'])
            ax.plot(dates_model, R, label='Recovered', linewidth=2, color=self.colors['R'])
            ax.plot(dates_model, V, label='Vaccinated', linewidth=2, color=self.colors['V'])
            ax.set_ylabel('Number of Individuals', fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_title('All Compartments', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # infectious and exposed (zoomed)
            ax = axes[3]
            ax.plot(dates_model, E, label='Exposed', linewidth=2.5, color=self.colors['E'])
            ax.plot(dates_model, I, label='Infectious', linewidth=2.5, color=self.colors['I'])
            ax.set_ylabel('Number of Individuals', fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_title('Disease Progression', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig 
    
    def plot_scenario_comparison(
        self,
        scenarios: Dict[str, Tuple[np.ndarray, np.ndarray]],
        metric: str = 'cases',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Compare multiple scenarios.
        Parameters:
        scenarios : dict. Dictionary mapping scenario names to (t, y) tuples
        metric : str. Which metric to plot ('cases', 'deaths', 'both')
        save_path : str, optional. Path to save figure
            
        Returns:
        fig : Figure
        """
        if metric == 'both':
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            axes = [ax]
        
        colors_scenario = plt.cm.Set2(np.linspace(0, 1, len(scenarios)))
        
        for idx, (scenario_name, (t, y)) in enumerate(scenarios.items()):
            dates = self.days_to_dates(t)
            S, E, I, R, V, D = y
            cases = E + I + R + D
            
            if metric in ['cases', 'both']:
                ax = axes[0] if metric == 'both' else axes[0]
                ax.plot(dates, cases, linewidth=2.5, 
                       label=scenario_name, color=colors_scenario[idx])
                ax.set_ylabel('Cumulative Cases', fontsize=12)
                ax.set_title('Scenario Comparison: Cases', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            if metric in ['deaths', 'both']:
                ax = axes[1] if metric == 'both' else axes[0]
                ax.plot(dates, D, linewidth=2.5,
                       label=scenario_name, color=colors_scenario[idx])
                ax.set_ylabel('Cumulative Deaths', fontsize=12)
                ax.set_title('Scenario Comparison: Deaths', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        if metric == 'both':
            for ax in axes:
                ax.set_xlabel('Date', fontsize=12)
        else:
            axes[0].set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
        
    def plot_r_effective(
        self,
        t: np.ndarray,
        S: np.ndarray,
        r0: float,
        N: float,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot effective reproduction number over time.
        Parameters:
        t : ndarray. Time points
        S : ndarray. Susceptible population over time
        r0 : float. Basic reproduction number
        N : float. Total population
        save_path : str, optional. Path to save figure
            
        Returns:
        fig : Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        dates = self.days_to_dates(t)
        r_eff = r0 * (S / N)
        
        ax.plot(dates, r_eff, linewidth=2.5, color='darkblue', label='R_eff(t)')
        ax.axhline(y=1, linestyle='--', color='red', linewidth=2, 
                   label='Epidemic Threshold (R=1)', alpha=0.7)
        ax.fill_between(dates, 0, r_eff, where=(r_eff > 1), 
                        alpha=0.2, color='red', label='Epidemic Growth')
        ax.fill_between(dates, 0, r_eff, where=(r_eff <= 1), 
                        alpha=0.2, color='green', label='Epidemic Decline')
        
        ax.set_ylabel('Effective Reproduction Number R_eff', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title(f'Effective Reproduction Number Over Time (R0 = {r0:.2f})', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_residuals(
        self,
        data: pd.DataFrame,
        t_model: np.ndarray,
        y_model: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot residuals for model fit assessment.
        
        Parameters:
        data : DataFrame. Observed data
        t_model : ndarray. Model time points
        y_model : ndarray. Model solution
        save_path : str, optional. Path to save figure
            
        Returns:
        fig : Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # interpolate model to data time points
        from scipy.interpolate import interp1d
        
        S, E, I, R, V, D = y_model
        cases_model = E + I + R + D
        
        f_cases = interp1d(t_model, cases_model, kind='cubic')
        f_deaths = interp1d(t_model, D, kind='cubic')
        
        t_data = data['days_since_start'].values
        cases_pred = f_cases(t_data)
        deaths_pred = f_deaths(t_data)
        
        # calculate residuals
        residuals_cases = data['cases_cumulative'].values - cases_pred
        residuals_deaths = data['deaths_cumulative'].values - deaths_pred
        
        # residuals vs time - cases
        ax = axes[0, 0]
        dates_data = self.days_to_dates(t_data)
        ax.plot(dates_data, residuals_cases, 'o-', markersize=8, linewidth=2)
        ax.axhline(y=0, linestyle='--', color='red', alpha=0.7)
        ax.set_ylabel('Residual (Observed - Predicted)', fontsize=11)
        ax.set_title('Residuals: Cases', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # residuals vs time - deaths
        ax = axes[0, 1]
        ax.plot(dates_data, residuals_deaths, 's-', markersize=8, linewidth=2)
        ax.axhline(y=0, linestyle='--', color='red', alpha=0.7)
        ax.set_ylabel('Residual (Observed - Predicted)', fontsize=11)
        ax.set_title('Residuals: Deaths', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # residual histogram - cases
        ax = axes[1, 0]
        ax.hist(residuals_cases, bins=5, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, linestyle='--', color='red', linewidth=2)
        ax.set_xlabel('Residual', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Residual Distribution: Cases', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # residual histogram - deaths
        ax = axes[1, 1]
        ax.hist(residuals_deaths, bins=5, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, linestyle='--', color='red', linewidth=2)
        ax.set_xlabel('Residual', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Residual Distribution: Deaths', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig