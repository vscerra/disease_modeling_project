"""
===========================================================
experiments.py
Author: Veronica Scerra
Last Updated: 2025-10-11
===========================================================

Description:
    Utilities for parameter sweeps and visualizations for the
    deterministic SIR model: grid search over (beta, gamma),
    tidy results as a DataFrame, and plotting helpers.

Example Usage:
    from epimodels.experiments import grid_sweep, heatmap, contour
    df = grid_sweep(betas, gammas, N, I0, t)
    heatmap(df, x='beta', y='gamma', value='final_size')
    contour(df, x='beta', y='gamma', value='peak_prevalence')

Notes:
    - Uses only numpy, pandas, matplotlib.
    - Assumes SIRModel is available in src/epimodels/sir.py.
-----------------------------------------------------------
License: MIT
===========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .sir import SIRModel

def _summarize_one(N, beta, gamma, I0, R0_init, t):
    """Run one simulation and return a dict of summary statistics"""
    model = SIRModel(N=N, beta=beta, gamma=gamma)
    out = model.simulate(t=t, I0=I0, R0_init=R0_init)
    S, I, R = out['S'], out['I'], out['R']
    inc = out['incidence']
    N0 = S[0] + I[0] + R[0]

    peak_idx = int(np.argmax(I))
    peak_day = float(out["t"][peak_idx])
    peak_infected = float(I[peak_idx])
    peak_prevalence = float(peak_infected / N0)
    final_size = float(R[-1] / N0)
    max_incidence = float(np.max(inc))
    R0_basic = float(beta/gamma) if gamma > 0 else np.inf

    return {
        "beta": float(beta),
        "gamma": float(gamma),
        "R0": R0_basic,
        "peak_day": peak_day,
        "peak_infected": peak_infected,
        "peak_prevalence": peak_prevalence,
        "final_size": final_size,
        "max_incidence": max_incidence,
    }
    

def grid_sweep(
    betas,
    gammas,
    N: int,
    I0: int, 
    t: np.ndarray,
    R0_init: int=0) -> pd.DataFrame:
    """
    Evaluate the SIR model across a grid of (beta, gamma) values. Returns
     a tidy pandas DataFrame with one row per parameter combo
     """
    records = []
    for b in betas:
        for g in gammas:
            rec = _summarize_one(N=N, beta=float(b), gamma=float(g), I0=I0, R0_init=R0_init, t=t)
            records.append(rec)
    df = pd.DataFrame.from_records(records)
    # sorts
    return df.sort_values(["beta", "gamma"]).reset_index(drop=True)


def pivot_for_plot(df: pd.DataFrame, x: str, y: str, value: str):
    """Pivot a DataFrame to 2D arrays for plotting (heatmaps/contour)
    Return X_grid, Y_grid, Z_values
    """
    # ensure you're getting unique combos
    sub = df[[x, y, value]].drop_duplicates()
    x_vals = np.sort(sub[x].unique())
    y_vals = np.sort(sub[y].unique())
    Z = np.empty((len(y_vals), len(x_vals)), dtype=float) #rows: y, cols: x
    for i, gy in enumerate(y_vals):
        row = sub[sub[y] == gy].sort_values(x)
        Z[i, :] = row[value].to_numpy()
    X, Y = np.meshgrid(x_vals, y_vals)
    return X, Y, Z


def heatmap(df: pd.DataFrame, x: str, y: str, value: str, xlabel=None, ylabel=None, title=None):
    """Plot a heatmap of a summary metric (e.g., final_size, peak_prevalence)"""
    X, Y, Z = pivot_for_plot(df, x=x, y=y, value=value)
    plt.figure()
    # imshow expects [rows, cols] -> (y, x)
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    plt.imshow(Z, origin='lower', aspect='auto', extent=extent)
    cbar = plt.colorbar()
    cbar.set_label(value.replace("_", " ").title())
    plt.xlabel(xlabel if xlabel else x)
    plt.ylabel(ylabel if ylabel else y)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def contour(df: pd.DataFrame, x: str, y: str, value: str, levels=10, xlabel=None, ylabel=None, title=None):
    """Plot contour lines of a summary statistic"""
    X, Y, Z = pivot_for_plot(df, x=x, y=y, value=value)
    plt.figure()
    CS = plt.contour(X, Y, Z, levels=levels)
    plt.clabel(CS, inline=True, fontsize=8)
    plt.xlabel(xlabel if xlabel else x)
    plt.ylabel(ylabel if ylabel else y)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()