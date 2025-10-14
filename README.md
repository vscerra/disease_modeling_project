Disease Modeling Toolbox

A modular, research-friendly toolbox for building and comparing epidemiological models. It starts with a clean SIR baseline on synthetic data and grows into real-world fitting, parameter sweeps, and model comparisons (SEIR, time-varying β, etc.).

Key ideas

Modular code (reusable src/ modules) + exploratory notebooks.

Deterministic SIR with a lightweight RK4 integrator (no SciPy required).

Parameter sweep utilities for β, γ → heatmaps/contours of outcomes.

Parameter fitting to time series (I(t) or incidence) via grid + pattern search with scale alignment (handles under-reporting).

Clear path to real data ingestion, observation models (Poisson/NegBin), and model comparison.


