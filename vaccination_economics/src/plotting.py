"""
===============================================================================
plotting.py
Author: Veronica Scerra
Last Updated: 2026-02-25
===============================================================================
Plotting functions for phase2b_coupled_dynamics.ipynb notebook (to save space
in cells)

functions:
- plot_static_vs_behavioral

--------------------------------------------------------------------------------
License: MIT
================================================================================
"""
import numpy as np 
import matplotlib.pyplot as plt 

def plot_static_vs_behavioral(results_static, results_behavioral):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # calculate metrics for both
    for idx, (name, results) in enumerate([('Static', results_static), ('Behavioral', results_behavioral)]):
        time_years = results['time'] / 365
        N_t = results['S'] + results['E'] + results['I'] + results['R'] + results['V']
        prevalence = results['I'] / N_t * 100
        coverage = results['V'] / N_t * 100
        
        color = 'blue' if name == 'Static' else 'red'
        
        # prevalence
        axes[0, 0].plot(time_years, prevalence, label=name, linewidth=2, color=color)
        
        # poverage
        axes[0, 1].plot(time_years, coverage, label=name, linewidth=2, color=color)
        
        # cumulative cases
        cumulative = np.cumsum(results['incidence'])
        axes[1, 0].plot(time_years, cumulative, label=name, linewidth=2, color=color)
        
        # phase plane
        axes[1, 1].plot(coverage, prevalence, label=name, linewidth=2, color=color, alpha=0.7)

    # customize plots
    axes[0, 0].set_xlabel('Time (years)')
    axes[0, 0].set_ylabel('Prevalence (%)')
    axes[0, 0].set_title('Disease Prevalence', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    axes[0, 1].set_xlabel('Time (years)')
    axes[0, 1].set_ylabel('Coverage (%)')
    axes[0, 1].set_title('Vaccination Coverage', fontweight='bold')
    axes[0, 1].axhline(y=93.3, color='green', linestyle='--', label='HIT', alpha=0.5)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Time (years)')
    axes[1, 0].set_ylabel('Cumulative Cases')
    axes[1, 0].set_title('Total Disease Burden', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Coverage (%)')
    axes[1, 1].set_ylabel('Prevalence (%)')
    axes[1, 1].set_title('Phase Plane: Coverage vs Prevalence', fontweight='bold')
    axes[1, 1].axvline(x=93.3, color='green', linestyle='--', alpha=0.5, label='HIT')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.suptitle('Static vs Behavioral Vaccination Strategies', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()