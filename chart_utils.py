"""
Chart Utilities for Economic Indicators
========================================
Common utilities for aligning charts across different economic indicator scripts.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple, List, Union

# ============================================================================
# STANDARD FIGURE SIZES FOR CONSISTENT VISUALIZATION
# ============================================================================

# Standard width for all charts to ensure horizontal alignment
STANDARD_WIDTH = 16

# Standard heights for different chart types
STANDARD_HEIGHT_SINGLE = 8      # For single panel charts
STANDARD_HEIGHT_MULTI = 20      # For multi-panel vertical charts (4-6 panels)
STANDARD_HEIGHT_MEDIUM = 14     # For medium multi-panel charts (3-4 panels)
STANDARD_HEIGHT_COMPACT = 12    # For compact multi-panel charts (2-3 panels)

def setup_aligned_time_axis(ax, start_date: Optional[pd.Timestamp] = None, 
                            end_date: Optional[pd.Timestamp] = None,
                            major_interval_years: int = 2):
    """
    Set up consistent time axis formatting across all charts.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to format
    start_date : pd.Timestamp, optional
        Explicit start date for x-axis limits
    end_date : pd.Timestamp, optional
        Explicit end date for x-axis limits
    major_interval_years : int
        Interval for major tick marks in years (default: 2)
    """
    # Set x-axis limits if provided
    if start_date is not None and end_date is not None:
        ax.set_xlim(start_date, end_date)
    
    # Consistent date formatting
    ax.xaxis.set_major_locator(mdates.YearLocator(major_interval_years))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=0)
    ax.set_xlabel('Year', fontsize=10)

def get_common_date_range(data_series_list: List[Union[pd.Series, pd.DataFrame]]) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Find the common date range across multiple time series.
    
    Parameters:
    -----------
    data_series_list : list of pd.Series or pd.DataFrame
        List of time series data to find common range for
        
    Returns:
    --------
    tuple : (start_date, end_date) as pd.Timestamp objects, or (None, None) if no valid data
    """
    valid_series = []
    
    for s in data_series_list:
        if s is None:
            continue
        if hasattr(s, 'empty') and s.empty:
            continue
        if isinstance(s, (pd.Series, pd.DataFrame)) and len(s) > 0:
            valid_series.append(s)
    
    if not valid_series:
        return None, None
    
    start_dates = []
    end_dates = []
    
    for s in valid_series:
        idx = s.index
        if len(idx) > 0:
            start_dates.append(idx[0])
            end_dates.append(idx[-1])
    
    if start_dates and end_dates:
        return max(start_dates), min(end_dates)
    
    return None, None

def add_chart_metadata(ax, title: str, ylabel: str, data_points: int = None,
                       frequency: str = None):
    """
    Add consistent metadata formatting to charts.
    
    Parameters:
    -----------
    ax : matplotlib axis
    title : str
        Chart title
    ylabel : str
        Y-axis label
    data_points : int, optional
        Number of data points (added to title)
    frequency : str, optional
        Data frequency (e.g., 'daily', 'monthly', 'quarterly')
    """
    title_text = title
    if data_points and frequency:
        title_text += f"\n({data_points} {frequency} observations)"
    
    ax.set_title(title_text, fontweight='bold', fontsize=12, pad=10)
    ax.set_ylabel(ylabel, fontsize=10)

def get_standard_figsize(num_panels: int = 1) -> Tuple[float, float]:
    """
    Get standardized figure size based on number of panels.
    
    Parameters:
    -----------
    num_panels : int
        Number of vertical panels in the figure
        - 1: Single panel chart
        - 2-3: Compact multi-panel
        - 4: Medium multi-panel
        - 5+: Full multi-panel
    
    Returns:
    --------
    tuple : (width, height) in inches
    """
    if num_panels == 1:
        return (STANDARD_WIDTH, STANDARD_HEIGHT_SINGLE)
    elif num_panels <= 3:
        return (STANDARD_WIDTH, STANDARD_HEIGHT_COMPACT)
    elif num_panels == 4:
        return (STANDARD_WIDTH, STANDARD_HEIGHT_MEDIUM)
    else:  # 5 or more panels
        return (STANDARD_WIDTH, STANDARD_HEIGHT_MULTI)
