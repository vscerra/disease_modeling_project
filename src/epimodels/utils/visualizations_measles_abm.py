"""
===========================================================
visualizations_measles_abm.py
Author: Veronica Scerra
Last Updated: 2026-01-28
===========================================================
Visualization functions for measles ABM simulations
======================================================
Provides functions for:
- Time series plotting (SEIR curves)
- Classroom-level spatial heatmaps
- Animated outbreak spread
- Comparative scenario analysis
- Statistical summaries 

License: MIT
===========================================================
"""
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
from matplotlib.patches import Rectangle 
from matplotlib.colors import ListedColormap 
import seaborn as sns 
from typing import List, Dict, Tuple, Optional 
import pandas as pd 

def plot_seir_timeseries(
    history: List[Dict], 
    title: str = 'SEIR Dynamics', 
    figsize: Tuple[int, int] = (12, 6), 
    save_path: Optional[str] = None
    ):
    """Plot SEIR time series from simulation history.
    Parameters:
    history: List[Dict]. Simulation history from MeaslesABM.run()
    title: str. Plot title 
    figsize: Tuple[int, int]. Figure size in inches
    save_path: str, optional. Path to save figure
    """
    times = [record['time'] for record in history]
    S = [record['S'] for record in history] 
    E = [record['E'] for record in history]
    I = [record['I'] for record in history]
    R = [record['R'] for record in history]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, S, 'b-', linewidth=2, label='Susceptible', alpha=0.8)
    ax.plot(times, E, 'y-', linewidth=2, label='Exposed', alpha=0.8)
    ax.plot(times, I, 'r-', linewidth=2, label='Infectious', alpha=0.8)
    ax.plot(times, R, 'g-', linewidth=2, label='Recovered', alpha=0.8)
    
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Number of students', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    # add key statistics as text
    total_infected = R[-1] - R[0]
    attack_rate = total_infected / (S[0] + E[0]) * 100 if (S[0] + E[0]) > 0 else 0
    peak_infectious = max(I)
    peak_day = times[I.index(peak_infectious)]
    
    stats_text = (f"Attack Rate: {attack_rate:.1f}%\n"
                 f"Peak Infectious: {peak_infectious} (Day {peak_day:.0f})")
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_classroom_heatmap(
        classroom_data: Dict[Tuple[int, int], Dict[str, int]],
        state: str = 'I',
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None):
    """Create a heatmap showing disease prevalence by classroom
    Parameters:
    classroom_data: Dict. Dictionary mapping (grade, classroom) to state counts
    state: str. Which disease state to visualize ('S', 'E', 'I', 'R')
    title: str, optional. Plot title
    figsize: Tuple[int, int]. Figure size in inches
    save_path: str, optional. Path to save figure
    """
    # extract dimensions
    grades = sorted(set(k[0] for k in classroom_data.keys()))
    classrooms = sorted(set(k[1] for k in classroom_data.keys()))

    # create matrix
    matrix = np.zeros((len(grades), len(classrooms)))

    for (grade, classroom), counts in classroom_data.items():
        grade_idx = grades.index(grade)
        classroom_idx = classrooms.index(classroom)
        matrix[grade_idx, classroom_idx] = counts[state] 
    
    # create plot 
    fig, ax = plt.subplots(figsize=figsize)

    # choose colormap based on state
    if state == 'S':
        cmap = 'Blues'
        state_name = 'Susceptible'
    elif state == 'E':
        cmap = 'YlOrBr'
        state_name = 'Exposed'
    elif state == 'I':
        cmap = 'Reds'
        state_name = 'Infected'
    else: #R
        cmap = 'Greens'
        state_name = 'Recovered' 

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # set ticks and labels
    ax.set_xticks(range(len(classrooms)))
    ax.set_yticks(range(len(grades)))
    ax.set_xticklabels([f'Class {c}' for c in classrooms])
    ax.set_yticklabels([f'Grade {g}' for g in grades])
    
    # add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'Number of {state_name} Students', fontsize=11)
    
    # add values as text
    for i in range(len(grades)):
        for j in range(len(classrooms)):
            text = ax.text(j, i, f'{int(matrix[i, j])}',
                         ha="center", va="center", color="black", fontsize=9)
    
    if title is None:
        title = f'{state_name} Students by Classroom'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_outbreak_animation(
        history: List[Dict],
        n_grades: int, 
        n_classrooms: int, 
        interval: int = 200,
        save_path: Optional[str] = None
):
    """Create an animated visualization of outbreak spread through classrooms
    Parameters:
    history: List[Dict]. Simulation history from MeaslesABM.run()
    n_grade: int. Number of grades in school 
    n_classrooms: int. Number of classrooms per grade
    interval: int. Delay between frames in milliseconds 
    save_path: str, optional. Path to save animation (as .gif or .mp4)

    Returns: 
    anim: matplotlib.animation.FuncAnimation. Animation object 
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # setup for classroom heatmap
    matrix = np.zeros((n_grades, n_classrooms))
    im = ax1.imshow(matrix, cmap='Reds', aspect='auto', vmin=0, vmax=25,
                   interpolation='nearest')
    ax1.set_xticks(range(n_classrooms))
    ax1.set_yticks(range(n_grades))
    ax1.set_xticklabels([f'Class {c}' for c in range(n_classrooms)])
    ax1.set_yticklabels([f'Grade {g+1}' for g in range(n_grades)])
    ax1.set_title('Infectious Students by Classroom', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Number Infectious')

    # setup for timeseries 
    times = [record['time'] for record in history]
    line_s, = ax2.plot([], [], 'b-', linewidth=2, label='Susceptible', alpha=0.8)
    line_e, = ax2.plot([], [], 'y-', linewidth=2, label='Exposed', alpha=0.8)
    line_i, = ax2.plot([], [], 'r-', linewidth=2, label='Infectious', alpha=0.8)
    line_r, = ax2.plot([], [], 'g-', linewidth=2, label='Recovered', alpha=0.8)
    
    ax2.set_xlim(0, max(times))
    ax2.set_ylim(0, max([record['S'] + record['R'] for record in history]) * 1.1)
    ax2.set_xlabel('Time (days)', fontsize=12)
    ax2.set_ylabel('Number of students', fontsize=12)
    ax2.set_title('Epidemic Dynamics', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    time_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        im.set_data(matrix)
        line_s.set_data([], [])
        line_e.set_data([], [])
        line_i.set_data([], [])
        line_r.set_data([], [])
        time_text.set_text('')
        return im, line_s, line_e, line_i, line_r, time_text
    
    def animate(frame):
        record = history[frame]
        # Update classroom heatmap
        classroom_data = record['classroom_data']
        for (grade, classroom), counts in classroom_data.items():
            matrix[grade-1, classroom] = counts['I']
        im.set_data(matrix)
        
        # Update time series
        current_times = times[:frame+1]
        S = [history[i]['S'] for i in range(frame+1)]
        E = [history[i]['E'] for i in range(frame+1)]
        I = [history[i]['I'] for i in range(frame+1)]
        R = [history[i]['R'] for i in range(frame+1)]
        
        line_s.set_data(current_times, S)
        line_e.set_data(current_times, E)
        line_i.set_data(current_times, I)
        line_r.set_data(current_times, R)
        
        time_text.set_text(f'Day {record["time"]:.0f}')
        
        return im, line_s, line_e, line_i, line_r, time_text
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(history), interval=interval,
                                  blit=True, repeat=True)
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000/interval)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=1000/interval)
    
    plt.tight_layout()
    return anim

def compare_scenarios(
        scenario_results: Dict[str, List[Dict]],
        scenario_labels: Dict[str, str],
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
):
    """Compare multiple simulation scenarios side-by-side
    Parameters:
    scenario_results: Dict[str, List[Dict]]. Dictionary mapping scenario names to simulation histories
    scenario_labels: Dict[str, str]. Dictionary mapping scenario names to display labels
    figsize: Tuple[int, int]. Figure size in inches
    save_path: str, optional. Path to save figure
    """
    n_scenarios = len(scenario_results)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten() 

    # define colors for each scenario
    colors = plt.cm.tab10(np.linspace(0, 1, n_scenarios)) 

    # plot 1: infectious over time 
    for idx, (scenario, history) in enumerate(scenario_results.items()):
        times = [record['time'] for record in history] 
        I = [record['I'] for record in history]
        axes[0].plot(times, I, linewidth=2.5, label=scenario_labels[scenario], color=colors[idx], alpha=0.8)
    
    axes[0].set_xlabel('Time (days)', fontsize=11)
    axes[0].set_ylabel('Infectious Students', fontsize=11)
    axes[0].set_title('Infectious Prevalence Over Time', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # plot 2: Cumulative infections
    for idx, (scenario, history) in enumerate(scenario_results.items()):
        times = [record['time'] for record in history]
        cumulative = [history[0]['R'] + history[0]['S'] - record['S'] 
                     for record in history]
        axes[1].plot(times, cumulative, linewidth=2.5, 
                    label=scenario_labels[scenario],
                    color=colors[idx], alpha=0.8)
    
    axes[1].set_xlabel('Time (days)', fontsize=11)
    axes[1].set_ylabel('Cumulative Infections', fontsize=11)
    axes[1].set_title('Cumulative Infections Over Time', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # plot 3: Peak infections comparison 
    scenario_names = []
    peak_values = []
    
    for scenario, history in scenario_results.items():
        I = [record['I'] for record in history]
        peak_values.append(max(I))
        scenario_names.append(scenario_labels[scenario])
    
    bars = axes[2].bar(range(len(scenario_names)), peak_values, color=colors, alpha=0.8)
    axes[2].set_xticks(range(len(scenario_names)))
    axes[2].set_xticklabels(scenario_names, rotation=45, ha='right', fontsize=9)
    axes[2].set_ylabel('Peak Infectious', fontsize=11)
    axes[2].set_title('Peak Infectious Comparison', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # add values on bars
    for bar, value in zip(bars, peak_values):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(value)}',
                    ha='center', va='bottom', fontsize=9)
        
    # plot 4: Final attack rates
    attack_rates = []
    
    for scenario, history in scenario_results.items():
        initial_susceptible = history[0]['S']
        final_susceptible = history[-1]['S']
        if initial_susceptible > 0:
            attack_rate = ((initial_susceptible - final_susceptible) / 
                          initial_susceptible * 100)
        else:
            attack_rate = 0
        attack_rates.append(attack_rate)
    
    bars = axes[3].bar(range(len(scenario_names)), attack_rates, color=colors, alpha=0.8)
    axes[3].set_xticks(range(len(scenario_names)))
    axes[3].set_xticklabels(scenario_names, rotation=45, ha='right', fontsize=9)
    axes[3].set_ylabel('Attack Rate (%)', fontsize=11)
    axes[3].set_title('Final Attack Rate Comparison', fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3, axis='y')
    axes[3].set_ylim(0, 100)
    
    # add values on bars
    for bar, value in zip(bars, attack_rates):
        height = bar.get_height()
        axes[3].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_vaccination_sensitivity(
        coverage_range: np.ndarray,
        results_dict: Dict[float, Dict[str, float]],
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
):
    """Plot sensitivity of epidemic outcomes to vaccination coverage.
    Parameters: 
    coverage_range: np.ndarray. Array of vaccination coverage values tested
    results_dict: Dict[float, Dict[str, float]]. Dictionary mapping coverage to outcomes (attack_rate, peak_infectious)
    figsize: Tuple[int, int]. Figure size in inches
    save_path: str, optional. Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # extract results
    attack_rates = [results_dict[cov]['attack_rate'] for cov in coverage_range]
    peak_infectious = [results_dict[cov]['peak_infectious'] for cov in coverage_range]

    # plot attack rate vs. coverage
    ax1.plot(coverage_range * 100, np.array(attack_rates) * 100, 
            'o-', linewidth=2, markersize=8, color='darkblue', alpha=0.7)
    ax1.set_xlabel('Vaccination Coverage (%)', fontsize=12)
    ax1.set_ylabel('Attack Rate (%)', fontsize=12)
    ax1.set_title('Attack Rate vs Vaccination Coverage', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 105)

    # add herd immunity threshold life if visible 
    # for measles with R0 ~ 15, HIT = 1 - 1/R0 = 93%
    ax1.axvline(x=93, color='red', linestyle='--', linewidth=2, 
               alpha=0.6, label='Theoretical HIT (R₀=15)')
    ax1.legend(fontsize=10)

    # plot peak infectious vs coverage 
    ax2.plot(coverage_range * 100, peak_infectious, 
            'o-', linewidth=2, markersize=8, color='darkred', alpha=0.7)
    ax2.set_xlabel('Vaccination Coverage (%)', fontsize=12)
    ax2.set_ylabel('Peak Infectious', fontsize=12)
    ax2.set_title('Peak Infectious vs Vaccination Coverage', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, 105)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def summarize_epidemic_statistics(history: List[Dict]) -> pd.DataFrame:
    """Generate summary statistics from simulation history.
    Parameters: 
    history: List[Dict]. Simulation history from MeaslesABM.run()

    Returns:
    summary: pd.DataFrame. DataFrame with epidemic statistics
    """
    times = [record['time'] for record in history]
    S = [record['S'] for record in history]
    E = [record['E'] for record in history]
    I = [record['I'] for record in history]
    R = [record['R'] for record in history]
    
    # calculate statistics
    initial_susceptible = S[0]
    final_susceptible = S[-1]
    total_infected = R[-1] - R[0]
    attack_rate = (total_infected / initial_susceptible * 100 
                  if initial_susceptible > 0 else 0)
    
    peak_infectious = max(I)
    peak_day = times[I.index(peak_infectious)]
    
    # duration metrics
    first_case_day = next((t for t, i in zip(times, I) if i > 0), None)
    last_case_day = next((t for t, i in zip(reversed(times), reversed(I)) if i > 0), None)
    epidemic_duration = last_case_day - first_case_day if last_case_day and first_case_day else 0
    
    # create summary
    summary = pd.DataFrame({
        'Metric': [
            'Initial Susceptible',
            'Final Susceptible',
            'Total Infected',
            'Attack Rate (%)',
            'Peak Infectious',
            'Peak Day',
            'First Case Day',
            'Last Case Day',
            'Epidemic Duration (days)'
        ],
        'Value': [
            initial_susceptible,
            final_susceptible,
            total_infected,
            f'{attack_rate:.2f}',
            peak_infectious,
            f'{peak_day:.0f}',
            f'{first_case_day:.0f}' if first_case_day else 'N/A',
            f'{last_case_day:.0f}' if last_case_day else 'N/A',
            f'{epidemic_duration:.0f}' if epidemic_duration else 'N/A'
        ]
    })
    
    return summary

    