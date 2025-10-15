"""
===========================================================
loaders.py
Author: Veronica Scerra
Last Updated: 2025-10-15
===========================================================

Description:
    Minimal loaders for Our World in Data (OWID) COVID-19.
    Returns a tidy daily time series for a single country with
    'date', 'incidence', 'population', and helper columns.

Notes:
    - Default source uses OWID's compact CSV (current catalog).
    - Fallback URL points to the legacy 'owid-covid-data.csv'.
    - Negative daily diffs (backfills) are clipped at 0.
    - Optional centered rolling mean smoothing.
-----------------------------------------------------------
License: MIT
===========================================================
"""

from __future__ import annotations
import pandas as pd
from typing import Optional

# current catalog "compact" dataset 
OWID_COMPACT_CSV = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
# legacy stable full dataset
OWID_FULL_CSV = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

def load_owid_covid(
        country: str,
        source: str="compact",      # "compact" or "full"
        start: Optional[str] = None,        # e.g. "2020-03-01"
        end: Optional[str] = None, 
        smooth: Optional[int] = 7,      # window for centering rolling mean; None to disable
        ) -> pd.DataFrame:
    """
    Load OWID COVID-19 data for one country and return tidy daily series.
    
    Return columns:
        date (datetime64[ns])
        incidence (float)          # daily new cases, >= 0
        incidence_smooth (float)   # optional centered rolling mean
        population (float)
        t (int)                    # days since first included date
    """
    url = OWID_COMPACT_CSV if source == "compact" else OWID_FULL_CSV
    df = pd.read_csv(url, parse_dates=['date'])
    # column names are consistent across both files for the fields you need:
    # 'location', 'date', 'new_cases', 'population' (verify in OWID docs)

    sub = (
        df.loc[df["country"] == country, ["date", "new_cases", "population"]]
        .sort_values("date")
        .reset_index(drop=True)
    )

    # clip negative backfills, keep as float
    sub["incidence"] = sub["new_cases"].fillna(0).astype(float).clip(lower=0)

    # optional smoothing (centered)
    if smooth and smooth > 1:
        sub["incidence_smooth"] = (
            sub["incidence"].rolling(smooth, center=True, min_periods=1).mean()
        )
    else:
        sub["incidence_smoooth"] = sub["incidence"]

    # data filtering 
    if start:
        sub = sub[sub["date"] >= pd.to_datetime(start)]
    if end:
        sub = sub[sub["date"] <= pd.to_datetime(end)]

    # forward/backfill population (it's constant per location)
    sub["population"] = sub["population"].ffill().bfill()

    # model the time index
    sub = sub.reset_index(drop=True)
    sub["t"] = (sub["date"] - sub["date"].iloc[0]).dt.days.astype(int)
    return sub[["date", "t", "incidence", "incidence_smooth", "population"]]
