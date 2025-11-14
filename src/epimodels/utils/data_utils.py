"""
===========================================================
data_utils.py
Author: Veronica Scerra
Last Updated: 2025-11-12
===========================================================
Data utilities for disease modeling

This module provides fucntions for loading, preprocessing, 
and managing epidemic outbreak datasets for use with 
compartmental models
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class OutbreakData:
    """ 
    Container for outbreak data with metadata.
    
    Attributes:
    -----------
    name: str
        Name of the outbreak
    time: np.ndarray
        Time points (days since outbreak start)
    incidence: np.ndarray
        New cases per time period
    prevalence: np.ndarray, optional
        Active cases at each time point
    cumulative: np.ndarray, optional
        Cumulative cases
    population: float
        Total population at risk
    description: str
        Description of the outbreak
    source: str
        Data source/reference
    """

    name: str
    time: np.ndarray
    incidence: np.ndarray
    prevalence: Optional[np.ndarray] = None
    cumulative: Optional[np.ndarray] = None
    population: float = None
    description: str = ""
    source: str = ""

    def __post_init__(self):
        """ Calculate derived quantities if not provided """
        if self.cumulative is None and self.incidence is not None:
            self.cumulative = np.cumsum(self.incidence) 

        if self.prevalence is None and self.incidence is not None: 
            # Note: This is a rough estimate: true prevalence needs removal rate
            self.prevalence = None # requires model parameters to estimate

    def to_dataframe(self) -> pd.DataFrame:
        """ Convert to pandas DataFrame for easy manipulation """
        data = {
            'time': self.time,
            'incidence': self.incidence, 
            'cumulative': self.cumulative
        }
        if self.prevalence is not None: 
            data['prevalence'] = self.prevalence
        return pd.DataFrame(data)
    
    def summary(self) -> str:
        """ Generate summary statistics of the outbreak"""
        total_cases = self.cumulative[-1] if self.cumulative is not None else np.sum(self.incidence)
        peak_incidence = np.max(self.incidence)
        peak_day = self.time[np.argmax(self.incidence)]
        duration = self.time[-1] - self.time[0]

        attack_rate = (total_cases / self.population * 100) if self.population else None

        summary = f"""
            Outbreak: {self.name}
            {'='*50}
            Duration: {duration} days
            Total cases: {total_cases:.0f}
            Peak incidence: {peak_incidence:.0f} cases (day {peak_day:.0f})
            """
        if attack_rate:
            summary += f"Attack rate: {attack_rate:.1f}%\n"
        if self.population:
            summary += f"Population: {self.population:.0f}\n"

        return summary
    
def load_boarding_school_flu() -> OutbreakData:
    """ 
    Load the 1978 English boarding school influenza outbreak data.
    
    This is a classic dataset used in epidemiological modeling. 763 boys were at risk,
    and 512 became ill over a 2-week period
    
    Returns:
    --------
    OutbreakData
        Outbreak data object containing time series and metadata
        
    References:
    -----------
    Anonymous(1978). "Influenze in a boarding school". 
    British Medical Journal, 1, 587
    """
    # Daily number of boys confined to bed
    time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    incidence = np.array([1, 3, 8, 28, 75, 221, 291, 255, 235, 190, 126, 70, 28, 12, 5])
    
    population = 763

    return OutbreakData(
        name='1978 English Boarding School Influenza',
        time=time,
        incidence=incidence,
        population=population,
        description="Influenza outbreak in a boarding school of 763 boys. "
                "One infected boy returned from holiday and sparked an epidemic.",
    source="British Medical Journal (1978), 1:587"
    )


def load_eyam_plague() -> OutbreakData:
    """
    Load the 1665-1666 Eyam plague outbreak data.
    
    The village of Eyam in Derbyshire famously quarentined itself during 
    the bubonic plague. This dataset contains monthly death records.
    
    Returns:
    --------
    OutbreakData
        Outbreak data object containing time series and metadata
        
    References:
    -----------
    Raggett, G. F. (1982). "Modeling the Eyam plague".
    The Institute of Mathematics and its Applications, 18, 221-226.
    """

    # Months since outbreak start (0 = June 1665)
    time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # Monthly deaths
    incidence = np.array([0, 0, 7, 5, 20, 29, 56, 78, 62, 23, 6, 3, 2, 0, 0])
    
    # Estimated population at start
    population = 350
    
    return OutbreakData(
        name="1665-1666 Eyam Plague",
        time=time,
        incidence=incidence,
        population=population,
        description="Bubonic plague outbreak in the village of Eyam, England. "
                "The village quarantined itself to prevent spread. Monthly death counts.",
        source="Raggett (1982), IMA Journal"
    )


