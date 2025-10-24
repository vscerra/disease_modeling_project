"""
===========================================================
ebola.py
Author: Veronica Scerra
Last Updated: 2025-10-24
===========================================================

Description:
   Preprocessing step to take long data csv to a dataframe suited for
   modeling - moving from cumulative to daily/weekly counts, only
   using "confirmed / probable" cases, and selecting a single country

Notes:
    - 
-----------------------------------------------------------
License: MIT
===========================================================
"""

from __future__ import annotations
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import pandas as pd
import requests

CaseDefinition = Literal[
    "confirmed",
    "confirmed_probable_suspected",
    "probable",
    "suspected",
    "confirmed_deaths",
    "confirmed_probable_suspected_deaths"
]

@dataclass
class EbolaPreprocessConfig:
    """
    Configuration for Ebola data preprocessing
    """
    country: str = "Sierra Leone"
    case_definition: CaseDefinition = "confirmed"
    date_col: str = "Date"
    country_col: str = "Country"
    indicator_col: str = "Indicator"   # <-- fixed name (was 'indication_col')
    value_col: str = "value"
    # ensure cumulative counts are monotone (WHO files sometimes revise downward)
    enforce_monotone_cumulative: bool = True
    # clip negative diffs (can still happen after rounding or nimor inconsistencies)
    clip_negative_new_cases: bool = True
    # optional resample frequency: "D" (daily), "W" (weekly), None (no resample)
    resample: Optional[Literal["D", "W"]] = "W"
    # aggregation for resampling. for new cases we sum within period
    resample_label: Literal["left", "right"] = "left"
    # output path to save cleaned CSV. If None, do not save
    save_to: Optional[Path] = None
    # networking
    timeout_s: int = 30

# ---- Public API -----------------------------------------------------------

def load_and_clean_ebola(
    source: str | Path,
    config: EbolaPreprocessConfig = EbolaPreprocessConfig(),
) -> pd.DataFrame:
    """
    Load and preprocess WHO/HDX Ebola cumulative case data into a time series.

    Parameters
    ----------
    source : str | Path
        File path or URL to the WHO/HDX Ebola CSV (e.g., 'ebola_data_db_format.csv').
    config : EbolaPreprocessConfig
        Preprocessing options (country, case definition, resampling, etc.).

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns:
        - 'date' (datetime64[ns])
        - 'country' (string)
        - 'cumulative_cases' (int)
        - 'new_cases' (int)
        If `resample` is not None, rows are resampled to that frequency.

    Notes
    -----
    - The HDX file commonly has columns: 'Indicator', 'Country', 'Date', 'value'
    - `case_definition` controls which indicator string is selected.
    """
    # Use robust reader so URLs and local files both work consistently
    df = _read_csv_robust(source, config)
    df = _filter_country_indicator(df, config)
    df = _deduplicate_and_sort(df, config)
    df = _derive_incidence(df, config)
    if config.resample is not None:
        df = _resample_incidence(df, config)
    if config.save_to:
        _ensure_parent(config.save_to)
        df.to_csv(config.save_to, index=False)
    return df


def list_available_indicators(source: str | Path) -> pd.Series:
    """
    Quick helper to inspect distinct 'Indicator' strings in a file.

    Returns a Series sorted by frequency.
    """
    df = _read_csv_robust(source, EbolaPreprocessConfig())
    col = _find_col(df.columns, "Indicator")
    return df[col].dropna().astype(str).value_counts().sort_values(ascending=False)

# ---------- Robust CSV loader ----------------------------------------------

def _read_csv_robust(source: str | Path, cfg: EbolaPreprocessConfig) -> pd.DataFrame:
    """
    Read CSV from local path or URL with fallbacks:
    1) If URL: try direct download, send a real User-Agent
    2) If preview/proxy returns HTML or empty, raise a clear error
    """
    src = str(source)

    # Local file path
    p = Path(src)
    if p.exists():
        try:
            df = pd.read_csv(p)
            return _standardize_columns(df)
        except pd.errors.EmptyDataError as e:
            raise RuntimeError(f"File exists but is empty: {p}") from e

    # URL: fetch via requests, then read from BytesIO
    headers = {"User-Agent": "Mozilla/5.0 (ebola-preprocessor)"}
    try:
        resp = requests.get(src, headers=headers, timeout=cfg.timeout_s)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch URL: {src}\n{e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} fetching {src}")

    content = resp.content or b""
    if len(content) < 10:
        raise RuntimeError(
            "Downloaded 0/very few bytes from URL. "
            "If you used the HDX hxlproxy preview link, switch to the direct download URL instead"
        )

    try:
        df = pd.read_csv(io.BytesIO(content))
    except pd.errors.EmptyDataError as e:
        raise RuntimeError("Response contained no CSV data.") from e

    return _standardize_columns(df)

# ---- Internal helpers -----------------------------------------------------

def _read_csv_standardize(source: str | Path, cfg: EbolaPreprocessConfig) -> pd.DataFrame:
    # (kept for backward compatibility; not used by default path now)
    df = pd.read_csv(source)
    return _standardize_columns(df)

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # normalize column names to strip spaces; flatten multiindex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
    df.columns = pd.Index([str(c).strip() for c in df.columns])
    return df


def _find_col(columns, expected_name: str) -> str:
    """
    Map tolerant column lookup (handles minor naming/whitespace variants).
    Works even if columns contain tuples (MultiIndex).
    """
    names = [str(c[0]) if isinstance(c, tuple) and len(c) > 0 else str(c) for c in columns]
    # exact
    for name in names:
        if name == expected_name:
            return name
    # case/space-insensitive
    exp = expected_name.strip().lower()
    for name in names:
        if name.strip().lower() == exp:
            return name
    raise KeyError(f"Expected column '{expected_name}' not found. Available: {names}")


def _indicator_regex(case_definition: CaseDefinition, deaths_ok: bool = False) -> re.Pattern:
    """
    Build a regex to match the 'Indicator' strings we want.
    """
    # Base fragments
    cases = r"(?:Ebola)\s+cases"
    deaths = r"(?:Ebola)\s+deaths"

    if case_definition == "confirmed":
        pat = rf"^Cumulative number of confirmed {cases}$"
    elif case_definition == "probable":
        pat = rf"^Cumulative number of probable {cases}$"
    elif case_definition == "suspected":
        pat = rf"^Cumulative number of suspected {cases}$"
    elif case_definition == "confirmed_probable_suspected":
        pat = rf"^Cumulative number of confirmed,\s*probable and suspected {cases}$"
    elif case_definition == "confirmed_deaths":
        pat = rf"^Cumulative number of confirmed {deaths}$"
    elif case_definition == "confirmed_probable_suspected_deaths":
        pat = rf"^Cumulative number of confirmed,\s*probable and suspected {deaths}$"
    else:
        raise ValueError(f"Unsupported case_definition: {case_definition}")

    flags = re.IGNORECASE
    return re.compile(pat, flags)


def _filter_country_indicator(df: pd.DataFrame, cfg: EbolaPreprocessConfig) -> pd.DataFrame:
    country_col = _find_col(df.columns, cfg.country_col)
    indicator_col = _find_col(df.columns, cfg.indicator_col)
    value_col = _find_col(df.columns, cfg.value_col)

    # Keep only rows for the chosen country
    sub = df.loc[df[country_col].astype(str).str.strip().eq(cfg.country)].copy()

    # Keep only the desired indicator
    pat = _indicator_regex(cfg.case_definition)
    sub = sub.loc[sub[indicator_col].astype(str).str.fullmatch(pat)]

    # Rename to standardized schema
    sub = sub.rename(
        columns={
            country_col: "country",
            indicator_col: "indicator",
            _find_col(sub.columns, cfg.date_col): "date",
            value_col: "cumulative_cases",
        }
    )[["date", "country", "indicator", "cumulative_cases"]]

    # Coerce types
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub["country"] = sub["country"].astype(str)
    sub["cumulative_cases"] = pd.to_numeric(sub["cumulative_cases"], errors="coerce")

    # Drop incomplete rows
    sub = sub.dropna(subset=["date", "cumulative_cases"]).copy()
    return sub


def _deduplicate_and_sort(df: pd.DataFrame, cfg: EbolaPreprocessConfig) -> pd.DataFrame:
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    # Optional: enforce monotone cumulative (revisions shouldn’t decrease totals)
    if cfg.enforce_monotone_cumulative:
        df["cumulative_cases"] = df["cumulative_cases"].cummax()
    return df.reset_index(drop=True)


def _derive_incidence(df: pd.DataFrame, cfg: EbolaPreprocessConfig) -> pd.DataFrame:
    df["new_cases"] = df["cumulative_cases"].diff()
    if cfg.clip_negative_new_cases:
        df["new_cases"] = df["new_cases"].clip(lower=0)
    df["new_cases"] = df["new_cases"].fillna(0).astype(int)
    df["cumulative_cases"] = df["cumulative_cases"].astype(int)
    return df


def _resample_incidence(df: pd.DataFrame, cfg: EbolaPreprocessConfig) -> pd.DataFrame:
    # For cumulative series: take the *last* within the period.
    # For new cases: sum within the period.
    # Anchor label (left/right) controls where the timestamp lands.
    rs = (
        df.set_index("date")
          .resample(cfg.resample, label=cfg.resample_label, closed=cfg.resample_label)
          .agg({"cumulative_cases": "last", "new_cases": "sum"})
    )
    rs = rs.dropna(subset=["cumulative_cases"]).reset_index()
    rs["country"] = df["country"].iloc[0]
    rs["indicator"] = df["indicator"].iloc[0]
    cols = ["date", "country", "indicator", "cumulative_cases", "new_cases"]
    return rs[cols]


def _ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# ---- Optional CLI for quick use ------------------------------------------

def _default_url() -> str:
    # HDX preview link (stable for quick prototyping). Replace with local path if preferred.
    return (
        "https://data.humdata.org/hxlproxy/api/data-preview.csv?"
        "url=https%3A//data.humdata.org/dataset/ebola-cases-2014/"
        "resource/73986f4e-9a3a-4dc3-93f1-c3fe1d3ebd5c/download/ebola_data_db_format.csv"
    )


if __name__ == "__main__":
    # Example quick run: python -m src.dataio.ebola
    cfg = EbolaPreprocessConfig(
        country="Sierra Leone",
        case_definition="confirmed",
        resample="W",
        save_to=Path("data/processed/ebola_sierra_leone_weekly.csv"),
    )
    out = load_and_clean_ebola(_default_url(), cfg)
    print(out.head())
    print(f"\nRows: {len(out):,}")
