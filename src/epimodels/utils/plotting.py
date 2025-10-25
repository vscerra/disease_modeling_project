import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_cumulative_fit(dates, y_obs, y_hat, title):
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.plot(dates, y_obs, lw=2, label="Observed (cumulative)")
    ax.plot(dates, y_hat, lw=2, linestyle="--", label="Model fit")
    ax.set_title(title)
    ax.set_ylabel("Cumulative cases")
    ax.legend()
    ax.grid(alpha=0.25)
    return ax