"""
===========================================================
sir_figures.py
Author: Veronica Scerra
Last Updated: 2025-11-12
===========================================================
Visualization functions for SIR model analysis.

This module provides plotting utilities for exploring SIR model dynamics,
including time series plots, phase portraits, and parameter sensitivity analyses.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_simulation(t: np.ndarray,
                   S: np.ndarray,
                   I: np.ndarray,
                   R: np.ndarray,
                   R0: Optional[float] = None,
                   ax: Optional[Axes] = None,
                   show: bool = True,
                   title: Optional[str] = None) -> Axes:
    """
    Plot SIR simulation results as time series.
    
    Parameters
    ----------
    t : np.ndarray
        Time points
    S, I, R : np.ndarray
        Compartment values over time
    R0 : float, optional
        Basic reproduction number to display in title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    show : bool
        Whether to display the plot immediately
    title : str, optional
        Custom title. If None and R0 provided, uses default format
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, S, 'b-', linewidth=2, label='Susceptible')
    ax.plot(t, I, 'r-', linewidth=2, label='Infected')
    ax.plot(t, R, 'g-', linewidth=2, label='Recovered')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Number of individuals', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    elif R0 is not None:
        ax.set_title(f'SIR Model ($R_0$ = {R0:.2f})', fontsize=14)
    else:
        ax.set_title('SIR Model Dynamics', fontsize=14)
    
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.tight_layout()
        plt.show()
        
    return ax


def plot_phase_portrait(S: np.ndarray,
                       I: np.ndarray,
                       R0: Optional[float] = None,
                       N: Optional[float] = None,
                       ax: Optional[Axes] = None,
                       show: bool = True,
                       label: Optional[str] = None,
                       **plot_kwargs) -> Axes:
    """
    Plot phase portrait (S vs I) for SIR model.
    
    Parameters
    ----------
    S : np.ndarray
        Susceptible individuals over time
    I : np.ndarray
        Infected individuals over time
    R0 : float, optional
        Basic reproduction number
    N : float, optional
        Total population size (for adding threshold line)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    show : bool
        Whether to display the plot immediately
    label : str, optional
        Label for this trajectory
    **plot_kwargs
        Additional keyword arguments passed to plot()
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Default plot settings
    default_kwargs = {'linewidth': 2, 'alpha': 0.8}
    default_kwargs.update(plot_kwargs)
    
    if label and R0 is not None:
        label = f'{label} ($R_0$={R0:.2f})'
    elif R0 is not None:
        label = f'$R_0$ = {R0:.2f}'
    
    ax.plot(S, I, label=label, **default_kwargs)
    
    # Add epidemic threshold line if N is provided
    if N is not None and R0 is not None:
        threshold = N / R0
        ax.axvline(x=threshold, color='gray', linestyle='--', 
                   linewidth=1.5, alpha=0.6, 
                   label=f'Threshold (S = N/$R_0$)')
    
    ax.set_xlabel('Susceptible (S)', fontsize=12)
    ax.set_ylabel('Infected (I)', fontsize=12)
    ax.set_title('SIR Phase Portrait (S-I plane)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if label or (N is not None and R0 is not None):
        ax.legend(fontsize=10)
    
    if show:
        plt.tight_layout()
        plt.show()
        
    return ax


def plot_multiple_phase_portraits(simulations: List[Tuple],
                                  N: float,
                                  colors: Optional[List[str]] = None,
                                  figsize: Tuple[float, float] = (10, 8),
                                  show: bool=True) -> Tuple[Figure, Axes]:
    """
    Plot multiple phase portraits on the same axes for comparison.
    
    Parameters
    ----------
    simulations : list of tuples
        Each tuple contains (S, I, R0, label) for one simulation
    N : float
        Total population size
    colors : list of str, optional
        Colors for each trajectory
    figsize : tuple
        Figure size (width, height)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(simulations)))
    
    threshold_added = False
    for i, (S, I, R0, label) in enumerate(simulations):
        color = colors[i] if i < len(colors) else None
        plot_phase_portrait(S, I, R0=R0, N=N if not threshold_added else None,
                          ax=ax, show=False, label=label, color=color)
        threshold_added = True
    
    if show: 
        plt.tight_layout()
        plt.show()
    
    return fig, ax


def plot_infected_comparison(t: np.ndarray,
                            infected_series: List[Tuple[np.ndarray, str]],
                            title: str = "Infected Dynamics Comparison",
                            xlabel: str = "Time (days)",
                            ylabel: str = "Infected individuals",
                            figsize: Tuple[float, float] = (12, 7),
                            show_threshold: bool = False) -> Figure:
    """
    Plot multiple infected time series for comparison.
    
    Parameters
    ----------
    t : np.ndarray
        Time points
    infected_series : list of tuples
        Each tuple contains (I, label) where I is infected array and label is str
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    show_threshold : bool
        Whether to show a threshold line at I=1
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for I, label in infected_series:
        ax.plot(t, I, linewidth=2, label=label)
    
    if show_threshold:
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_parameter_grid(t: np.ndarray,
                       simulations: List[Tuple],
                       param_name: str,
                       nrows: int = 2,
                       ncols: int = 2,
                       figsize: Tuple[float, float] = (14, 10),
                       show_metrics: bool = True,
                       show: bool=True) -> Tuple[Figure, Axes]:
    """
    Create a grid of SIR simulation plots for different parameter values.
    
    Parameters
    ----------
    t : np.ndarray
        Time points
    simulations : list of tuples
        Each tuple contains (S, I, R, param_value, R0, metrics_dict)
        where metrics_dict has keys: 'peak_infected', 'attack_rate'
    param_name : str
        Name of the parameter being varied (for titles)
    nrows, ncols : int
        Grid dimensions
    figsize : tuple
        Figure size
    show_metrics : bool
        Whether to display metric text boxes
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (S, I, R, param_val, R0, metrics) in enumerate(simulations[:len(axes)]):
        ax = axes[i]
        
        ax.plot(t, S, 'b-', linewidth=2, label='Susceptible')
        ax.plot(t, I, 'r-', linewidth=2, label='Infected')
        ax.plot(t, R, 'g-', linewidth=2, label='Recovered')
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Individuals', fontsize=11)
        ax.set_title(f'{param_name} = {param_val:.2f}, R\u2080 = {R0:.2f}', fontsize=12)
        ax.legend(fontsize=9, loc='right')
        ax.grid(True, alpha=0.3)
        
        if show_metrics and metrics:
            N = S[0] + I[0] + R[0]
            text = (f"Peak: {metrics['peak_infected']:.0f}\n"
                   f"Attack: {metrics['attack_rate']/N*100:.1f}%")
            ax.text(0.98, 0.97, text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
    
    # Hide unused subplots
    for j in range(len(simulations), len(axes)):
        axes[j].set_visible(False)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return fig, ax


def plot_final_size_relation(R0_values: np.ndarray,
                             attack_rates: np.ndarray,
                             N: float,
                             theoretical: bool = True,
                             ax: Optional[Axes] = None,
                             show: bool = True) -> Axes:
    """
    Plot the relationship between R0 and final epidemic size (attack rate).
    
    Parameters
    ----------
    R0_values : np.ndarray
        Array of R0 values
    attack_rates : np.ndarray
        Corresponding final attack rates (total recovered)
    N : float
        Population size
    theoretical : bool
        Whether to overlay theoretical curve
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    show : bool
        Whether to display immediately
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot simulation results
    attack_rate_fraction = attack_rates / N
    ax.plot(R0_values, attack_rate_fraction * 100, 'o-', 
            linewidth=2, markersize=8, label='Simulation')
    
    # Add theoretical curve if requested (implicit solution)
    if theoretical:
        from scipy.optimize import fsolve
        
        R0_theory = np.linspace(R0_values.min(), R0_values.max(), 100)
        attack_theory = []
        
        for R0 in R0_theory:
            # Solve: R_inf = 1 - exp(-R0 * R_inf)
            if R0 <= 1:
                attack_theory.append(0)
            else:
                sol = fsolve(lambda x: x - (1 - np.exp(-R0 * x)), 0.5)[0]
                attack_theory.append(sol)
        
        ax.plot(R0_theory, np.array(attack_theory) * 100, '--', 
                linewidth=2, color='red', alpha=0.7, label='Theoretical')
    
    ax.axvline(x=1, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    
    ax.set_xlabel('Basic Reproduction Number ($R_0$)', fontsize=12)
    ax.set_ylabel('Final Attack Rate (%)', fontsize=12)
    ax.set_title('Epidemic Final Size vs $R_0$', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax