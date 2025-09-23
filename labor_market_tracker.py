#!/usr/bin/env python3
"""
Labor Market Tracker - Real-time Economic Health Gauge
======================================================

This script tracks key labor market indicators to assess real-time economic health.
Employment conditions are foundational to economic activity - affecting consumer 
spending, confidence, and policy decisions.

Key Metrics Tracked:
- Unemployment Rate (U-3): Primary unemployment measure
- Initial Jobless Claims: Weekly leading indicator of layoffs
- Job Openings (JOLTS): Labor demand and job availability
- Labor Force Participation Rate: Economic engagement measure
- Employment-Population Ratio: Broad employment health

Features:
- Real labor data from Bureau of Labor Statistics via FRED
- Composite labor market health scoring and classification
- Comprehensive statistical analysis and visualization
- Employment strength classification (Hot/Cold labor market)
- Quality validation and error handling

Usage:
    python3 labor_market_tracker.py
    python3 labor_market_tracker.py --start 2000-01-01 --window 12
    python3 labor_market_tracker.py --min-data-points 50

Author: GitHub Copilot
Date: September 2025
"""

import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import argparse
import numpy as np
import warnings
from typing import Dict, Optional, Tuple

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# NBER recession dates (official business cycle dating)
NBER_RECESSIONS = [
    ('1990-07-01', '1991-03-01'),  # Early 1990s recession
    ('2001-03-01', '2001-11-01'),  # Dot-com recession  
    ('2007-12-01', '2009-06-01'),  # Great Recession
    ('2020-02-01', '2020-04-01')   # COVID recession
]

def add_recession_shading(ax, data_start=None, data_end=None, alpha=0.2, color='gray', label_first=True):
    """
    Add NBER recession shading to a matplotlib axis.
    
    Parameters:
    - ax: matplotlib axis object
    - data_start: start date of data (datetime or string), for filtering relevant recessions
    - data_end: end date of data (datetime or string), for filtering relevant recessions  
    - alpha: transparency of shading (0-1)
    - color: color of recession shading
    - label_first: whether to add legend label to first recession only
    """
    if data_start is not None:
        if isinstance(data_start, str):
            data_start = pd.to_datetime(data_start)
        if isinstance(data_start, pd.Timestamp):
            data_start = data_start.to_pydatetime()
    
    if data_end is not None:
        if isinstance(data_end, str):
            data_end = pd.to_datetime(data_end)
        if isinstance(data_end, pd.Timestamp):
            data_end = data_end.to_pydatetime()
    
    labeled = False
    for i, (start_str, end_str) in enumerate(NBER_RECESSIONS):
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)
        
        # Skip recessions outside our data range
        if data_start and end_date < data_start:
            continue
        if data_end and start_date > data_end:
            continue
            
        # Add shading
        label = 'NBER Recession' if label_first and not labeled else ''
        ax.axvspan(start_date, end_date, alpha=alpha, color=color, label=label)
        
        if label_first and not labeled:
            labeled = True

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def get_unemployment_rate(start_date: str = '1990-01-01') -> pd.Series:
    """
    Fetch unemployment rate (U-3) from FRED.
    Lower values indicate stronger labor market.
    """
    try:
        print("📊 Fetching Unemployment Rate from FRED...")
        data = pdr.get_data_fred('UNRATE', start=start_date)
        if not data.empty:
            print(f"✓ Successfully retrieved Unemployment Rate ({len(data)} records)")
            return data.iloc[:, 0].dropna()
        else:
            print("⚠ No unemployment rate data available")
            return pd.Series()
    except Exception as e:
        print(f"❌ Failed to get unemployment rate: {str(e)[:50]}...")
        return pd.Series()

def get_initial_jobless_claims(start_date: str = '1990-01-01') -> pd.Series:
    """
    Fetch initial jobless claims from FRED.
    Weekly data - leading indicator of labor market health.
    Lower values indicate stronger labor market.
    """
    try:
        print("📊 Fetching Initial Jobless Claims from FRED...")
        data = pdr.get_data_fred('ICSA', start=start_date)
        if not data.empty:
            print(f"✓ Successfully retrieved Initial Jobless Claims ({len(data)} records)")
            # Convert to monthly averages for consistency
            monthly_data = data.resample('M').mean()
            return monthly_data.iloc[:, 0].dropna()
        else:
            print("⚠ No jobless claims data available")
            return pd.Series()
    except Exception as e:
        print(f"❌ Failed to get jobless claims: {str(e)[:50]}...")
        return pd.Series()

def get_job_openings(start_date: str = '1990-01-01') -> pd.Series:
    """
    Fetch Job Openings and Labor Turnover Survey (JOLTS) data from FRED.
    Higher values indicate stronger labor demand.
    """
    try:
        print("📊 Fetching Job Openings (JOLTS) from FRED...")
        data = pdr.get_data_fred('JTSJOL', start=start_date)
        if not data.empty:
            print(f"✓ Successfully retrieved Job Openings ({len(data)} records)")
            return data.iloc[:, 0].dropna()
        else:
            print("⚠ No job openings data available")
            return pd.Series()
    except Exception as e:
        print(f"❌ Failed to get job openings: {str(e)[:50]}...")
        return pd.Series()

def get_labor_force_participation(start_date: str = '1990-01-01') -> pd.Series:
    """
    Fetch Labor Force Participation Rate from FRED.
    Higher values indicate more economic engagement.
    """
    try:
        print("📊 Fetching Labor Force Participation Rate from FRED...")
        data = pdr.get_data_fred('CIVPART', start=start_date)
        if not data.empty:
            print(f"✓ Successfully retrieved Labor Force Participation Rate ({len(data)} records)")
            return data.iloc[:, 0].dropna()
        else:
            print("⚠ No labor force participation data available")
            return pd.Series()
    except Exception as e:
        print(f"❌ Failed to get labor force participation: {str(e)[:50]}...")
        return pd.Series()

def get_employment_population_ratio(start_date: str = '1990-01-01') -> pd.Series:
    """
    Fetch Employment-Population Ratio from FRED.
    Broader measure of employment health.
    Higher values indicate stronger employment.
    """
    try:
        print("📊 Fetching Employment-Population Ratio from FRED...")
        data = pdr.get_data_fred('EMRATIO', start=start_date)
        if not data.empty:
            print(f"✓ Successfully retrieved Employment-Population Ratio ({len(data)} records)")
            return data.iloc[:, 0].dropna()
        else:
            print("⚠ No employment-population ratio data available")
            return pd.Series()
    except Exception as e:
        print(f"❌ Failed to get employment-population ratio: {str(e)[:50]}...")
        return pd.Series()

def get_nonfarm_payrolls(start_date: str = '1990-01-01') -> pd.Series:
    """
    Fetch Nonfarm Payrolls (monthly job creation) from FRED.
    Month-over-month change in employment.
    """
    try:
        print("📊 Fetching Nonfarm Payrolls from FRED...")
        data = pdr.get_data_fred('PAYEMS', start=start_date)
        if not data.empty:
            # Calculate month-over-month change
            payroll_change = data.diff() * 1000  # Convert to actual jobs (thousands)
            print(f"✓ Successfully calculated Nonfarm Payroll Changes ({len(payroll_change)} records)")
            return payroll_change.iloc[:, 0].dropna()
        else:
            print("⚠ No nonfarm payrolls data available")
            return pd.Series()
    except Exception as e:
        print(f"❌ Failed to get nonfarm payrolls: {str(e)[:50]}...")
        return pd.Series()

def collect_labor_market_data(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Collect all labor market indicators and return as dictionary.
    """
    print("📊 Collecting comprehensive labor market data...")
    
    data_components = {}
    
    # Get unemployment rate (inverse indicator - lower is better)
    unemployment = get_unemployment_rate(start_date)
    if not unemployment.empty:
        data_components['Unemployment_Rate'] = unemployment
    
    # Get initial jobless claims (inverse indicator - lower is better)
    jobless_claims = get_initial_jobless_claims(start_date)
    if not jobless_claims.empty:
        data_components['Jobless_Claims'] = jobless_claims
    
    # Get job openings (positive indicator - higher is better)
    job_openings = get_job_openings(start_date)
    if not job_openings.empty:
        data_components['Job_Openings'] = job_openings
    
    # Get labor force participation (positive indicator - higher is better)
    labor_participation = get_labor_force_participation(start_date)
    if not labor_participation.empty:
        data_components['Labor_Force_Participation'] = labor_participation
    
    # Get employment-population ratio (positive indicator - higher is better)
    emp_pop_ratio = get_employment_population_ratio(start_date)
    if not emp_pop_ratio.empty:
        data_components['Employment_Population_Ratio'] = emp_pop_ratio
    
    # Get nonfarm payroll changes (positive indicator - higher is better)
    payroll_changes = get_nonfarm_payrolls(start_date)
    if not payroll_changes.empty:
        data_components['Payroll_Changes'] = payroll_changes
    
    if not data_components:
        print("❌ ERROR: No labor market data could be retrieved")
        return {}
    
    print(f"✅ Successfully collected {len(data_components)} labor market indicators")
    return data_components

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def normalize_labor_indicators(data_dict: Dict[str, pd.Series], window: int = 12) -> Dict[str, pd.Series]:
    """
    Normalize labor indicators using rolling z-scores.
    Handle inverse indicators (unemployment, jobless claims) appropriately.
    """
    normalized_data = {}
    
    # Inverse indicators (lower values = better labor market)
    inverse_indicators = ['Unemployment_Rate', 'Jobless_Claims']
    
    for name, series in data_dict.items():
        if len(series) < window:
            print(f"⚠ Skipping {name}: insufficient data for {window}-month window")
            continue
            
        # Calculate rolling statistics with minimum periods for monthly data
        min_periods = max(window, 12)  # At least 12 periods for stability
        rolling_mean = series.rolling(window=window*2, min_periods=min_periods).mean()  # Shorter window
        rolling_std = series.rolling(window=window*2, min_periods=min_periods).std()
        
        # Calculate z-score
        z_score = (series - rolling_mean) / rolling_std
        
        # For inverse indicators, flip the z-score (higher unemployment = worse labor market)
        if name in inverse_indicators:
            z_score = -z_score
        
        normalized_data[f'{name}_norm'] = z_score.dropna()
        print(f"✓ Normalized {name} (inverse: {name in inverse_indicators}) - {len(z_score.dropna())} points")
    
    return normalized_data

def create_labor_market_composite(data_dict: Dict[str, pd.Series], window: int = 12) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Create composite labor market health score.
    """
    print("🔄 Creating composite labor market health index...")
    
    # Normalize all indicators
    normalized_data = normalize_labor_indicators(data_dict, window)
    
    if not normalized_data:
        print("❌ ERROR: No normalized labor data available")
        return pd.Series(), pd.DataFrame()
    
    # Combine into DataFrame
    combined_df = pd.DataFrame(normalized_data)
    
    # Remove sparse data
    min_data_threshold = 0.3  # Require 30% data availability (less aggressive)
    before_cleaning = len(combined_df.columns)
    
    for col in combined_df.columns:
        if combined_df[col].notna().sum() / len(combined_df) < min_data_threshold:
            combined_df = combined_df.drop(columns=[col])
            print(f"⚠ Removed {col}: insufficient data coverage")
    
    print(f"🧹 Data cleaning: {before_cleaning} → {len(combined_df.columns)} indicators")
    
    if combined_df.empty:
        print("❌ ERROR: No labor indicators passed data quality checks")
        return pd.Series(), pd.DataFrame()
    
    # Define weights for different labor indicators
    base_weights = {
        'Unemployment_Rate_norm': 0.25,           # Primary unemployment measure
        'Jobless_Claims_norm': 0.20,             # Leading indicator of layoffs
        'Job_Openings_norm': 0.20,               # Labor demand
        'Labor_Force_Participation_norm': 0.15,   # Economic engagement
        'Employment_Population_Ratio_norm': 0.15, # Broad employment measure
        'Payroll_Changes_norm': 0.05,            # Monthly job creation
    }
    
    # Calculate available components and adjust weights
    available_components = [col for col in base_weights.keys() if col in combined_df.columns]
    available_weights = {comp: base_weights[comp] for comp in available_components}
    
    # Normalize weights to sum to 1.0
    total_weight = sum(available_weights.values())
    if total_weight > 0:
        adjusted_weights = {comp: weight/total_weight for comp, weight in available_weights.items()}
    else:
        print("❌ ERROR: No weighted components available")
        return pd.Series(), pd.DataFrame()
    
    print(f"📊 Using components: {list(available_components)}")
    weight_strings = [f'{comp.replace("_norm", "")}: {weight:.2f}' for comp, weight in adjusted_weights.items()]
    print(f"⚖️ Weights: {', '.join(weight_strings)}")
    
    # Calculate composite index
    composite_index = pd.Series(0.0, index=combined_df.index)
    
    for component, weight in adjusted_weights.items():
        component_data = combined_df[component].fillna(0)
        composite_index += component_data * weight
    
    composite_index = composite_index.dropna()
    
    if composite_index.empty:
        print("❌ ERROR: Composite labor index calculation failed")
        return pd.Series(), pd.DataFrame()
    
    print(f"✅ Created composite labor market index with {len(composite_index)} data points")
    return composite_index, combined_df

def classify_labor_market_strength(composite_score: float, percentile: float) -> str:
    """
    Classify labor market strength based on composite score and percentile.
    """
    if pd.isna(composite_score) or pd.isna(percentile):
        return "Unknown"
    
    # Classification based on z-score and percentile
    if composite_score > 2.0:
        return "Extremely Strong"
    elif composite_score > 1.5:
        return "Very Strong"
    elif composite_score > 1.0:
        return "Strong"
    elif composite_score > 0.5:
        return "Moderately Strong"
    elif composite_score > -0.5:
        return "Normal"
    elif composite_score > -1.0:
        return "Weak"
    elif composite_score > -1.5:
        return "Very Weak"
    else:
        return "Extremely Weak"

def get_labor_market_interpretation(strength_level: str) -> str:
    """Get interpretation text for labor market strength level."""
    interpretations = {
        "Extremely Strong": "Labor market extremely tight.\nVery low unemployment, high job creation.",
        "Very Strong": "Labor market very healthy.\nStrong employment growth and low joblessness.", 
        "Strong": "Labor market performing well.\nGood employment conditions and job availability.",
        "Moderately Strong": "Labor market above average.\nSteady employment with positive trends.",
        "Normal": "Labor market balanced.\nTypical employment conditions for economic cycle.",
        "Weak": "Labor market showing stress.\nRising unemployment or declining job creation.",
        "Very Weak": "Labor market in significant distress.\nHigh unemployment and poor job market.",
        "Extremely Weak": "Labor market in crisis.\nSevere unemployment and economic distress.",
        "Unknown": "Insufficient data for assessment."
    }
    return interpretations.get(strength_level, "Data quality insufficient for interpretation.")

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_labor_market_visualization(data_dict: Dict[str, pd.Series], composite_index: pd.Series, 
                                    combined_df: pd.DataFrame, window: int = 12) -> None:
    """
    Create comprehensive labor market visualization.
    """
    print("📈 Generating comprehensive labor market visualization...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))
    
    # Define color scheme
    colors = {
        'strong': '#2E8B57',      # Sea Green
        'normal': '#4682B4',      # Steel Blue
        'weak': '#DC143C',        # Crimson
        'neutral': '#708090'      # Slate Gray
    }
    
    # Subplot 1: Unemployment Rate
    ax1 = plt.subplot(3, 3, 1)
    if 'Unemployment_Rate' in data_dict:
        unemployment = data_dict['Unemployment_Rate'].dropna()
        
        # Add recession shading first
        add_recession_shading(ax1, data_start=unemployment.index[0], data_end=unemployment.index[-1], 
                             alpha=0.15, color='red', label_first=True)
        
        ax1.plot(unemployment.index, unemployment.values, linewidth=1.5, color=colors['weak'])
        ax1.fill_between(unemployment.index, unemployment.values, alpha=0.3, color=colors['weak'])
        ax1.set_title('Unemployment Rate (%)\n(Lower = Better)')
        ax1.set_ylabel('Unemployment Rate (%)')
        ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Initial Jobless Claims
    ax2 = plt.subplot(3, 3, 2)
    if 'Jobless_Claims' in data_dict:
        claims = data_dict['Jobless_Claims'].dropna()
        ax2.plot(claims.index, claims.values, linewidth=1.5, color=colors['weak'])
        ax2.fill_between(claims.index, claims.values, alpha=0.3, color=colors['weak'])
        ax2.set_title('Initial Jobless Claims\n(Lower = Better)')
        ax2.set_ylabel('Claims (Thousands)')
        ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Job Openings
    ax3 = plt.subplot(3, 3, 3)
    if 'Job_Openings' in data_dict:
        openings = data_dict['Job_Openings'].dropna()
        
        # Add recession shading
        add_recession_shading(ax3, data_start=openings.index[0], data_end=openings.index[-1], 
                             alpha=0.15, color='red', label_first=False)
        
        ax3.plot(openings.index, openings.values, linewidth=1.5, color=colors['strong'])
        ax3.fill_between(openings.index, openings.values, alpha=0.3, color=colors['strong'])
        ax3.set_title('Job Openings (JOLTS)\n(Higher = Better)')
        ax3.set_ylabel('Openings (Millions)')
        ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Labor Force Participation
    ax4 = plt.subplot(3, 3, 4)
    if 'Labor_Force_Participation' in data_dict:
        participation = data_dict['Labor_Force_Participation'].dropna()
        ax4.plot(participation.index, participation.values, linewidth=1.5, color=colors['normal'])
        ax4.fill_between(participation.index, participation.values, alpha=0.3, color=colors['normal'])
        ax4.set_title('Labor Force Participation Rate (%)\n(Higher = Better)')
        ax4.set_ylabel('Participation Rate (%)')
        ax4.grid(True, alpha=0.3)
    
    # Subplot 5: Employment-Population Ratio
    ax5 = plt.subplot(3, 3, 5)
    if 'Employment_Population_Ratio' in data_dict:
        emp_ratio = data_dict['Employment_Population_Ratio'].dropna()
        ax5.plot(emp_ratio.index, emp_ratio.values, linewidth=1.5, color=colors['strong'])
        ax5.fill_between(emp_ratio.index, emp_ratio.values, alpha=0.3, color=colors['strong'])
        ax5.set_title('Employment-Population Ratio (%)\n(Higher = Better)')
        ax5.set_ylabel('Ratio (%)')
        ax5.grid(True, alpha=0.3)
    
    # Subplot 6: Nonfarm Payroll Changes
    ax6 = plt.subplot(3, 3, 6)
    if 'Payroll_Changes' in data_dict:
        payrolls = data_dict['Payroll_Changes'].dropna()
        # Color positive/negative changes
        positive = payrolls >= 0
        ax6.bar(payrolls[positive].index, payrolls[positive].values, 
               width=20, color=colors['strong'], alpha=0.7, label='Job Gains')
        ax6.bar(payrolls[~positive].index, payrolls[~positive].values,
               width=20, color=colors['weak'], alpha=0.7, label='Job Losses')
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax6.set_title('Monthly Payroll Changes\n(Higher = Better)')
        ax6.set_ylabel('Jobs (Thousands)')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    # Subplot 7: Composite Labor Market Index
    ax7 = plt.subplot(3, 3, 7)
    if not composite_index.empty:
        # Color code by strength level
        composite_colors = []
        for score in composite_index.values:
            if score > 1.5:
                composite_colors.append(colors['strong'])
            elif score > 0.5:
                composite_colors.append('#90EE90')  # Light Green
            elif score > -0.5:
                composite_colors.append(colors['normal'])
            elif score > -1.5:
                composite_colors.append('#FFA500')  # Orange
            else:
                composite_colors.append(colors['weak'])
        
        ax7.scatter(composite_index.index, composite_index.values, c=composite_colors, alpha=0.6, s=8)
        ax7.plot(composite_index.index, composite_index.values, linewidth=1, color='gray', alpha=0.5)
        
        # Add strength level lines
        ax7.axhline(y=1.5, color=colors['strong'], linestyle='--', alpha=0.5, label='Very Strong')
        ax7.axhline(y=0.5, color='#90EE90', linestyle='--', alpha=0.5, label='Strong')
        ax7.axhline(y=-0.5, color='#FFA500', linestyle='--', alpha=0.5, label='Weak')
        ax7.axhline(y=-1.5, color=colors['weak'], linestyle='--', alpha=0.5, label='Very Weak')
    
    ax7.set_title('Composite Labor Market Index\n(Z-Score)')
    ax7.set_ylabel('Composite Score')
    ax7.grid(True, alpha=0.3)
    ax7.legend(loc='upper left', fontsize=8)
    
    # Subplot 8: All Normalized Indicators
    ax8 = plt.subplot(3, 3, 8)
    if not combined_df.empty:
        for col in combined_df.columns:
            if '_norm' in col:
                indicator_data = combined_df[col].dropna()
                if not indicator_data.empty:
                    label = col.replace('_norm', '').replace('_', ' ')
                    ax8.plot(indicator_data.index, indicator_data.values, 
                            linewidth=1.2, label=label, alpha=0.8)
        
        ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax8.set_title('Normalized Labor Indicators\n(Z-Scores)')
        ax8.set_ylabel('Z-Score')
        ax8.grid(True, alpha=0.3)
        ax8.legend(loc='upper left', fontsize=7)
    
    # Subplot 9: Current Status Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Get latest data for summary
    if not composite_index.empty:
        latest_date = composite_index.index[-1]
        latest_score = composite_index.iloc[-1]
        
        # Calculate percentile
        percentile = (composite_index.rank(pct=True).iloc[-1]) * 100
        strength_level = classify_labor_market_strength(latest_score, percentile)
        
        summary_text = f"""LABOR MARKET STATUS
        
📅 Latest: {latest_date.strftime('%Y-%m-%d')}

🎯 STRENGTH: {strength_level.upper()}
📊 Score: {latest_score:.2f} (Z-Score)
📈 Percentile: {percentile:.1f}%

KEY INDICATORS:"""
        
        y_pos = 0.95
        ax9.text(0.05, y_pos, summary_text, transform=ax9.transAxes, fontsize=11, 
                 verticalalignment='top', fontweight='bold')
        
        y_pos -= 0.45
        
        # Show latest values for key indicators
        key_indicators = ['Unemployment_Rate', 'Job_Openings', 'Labor_Force_Participation']
        for indicator in key_indicators:
            if indicator in data_dict and not data_dict[indicator].empty:
                latest_val = data_dict[indicator].iloc[-1]
                if indicator == 'Unemployment_Rate':
                    ax9.text(0.05, y_pos, f"Unemployment: {latest_val:.1f}%", 
                            transform=ax9.transAxes, fontsize=9)
                elif indicator == 'Job_Openings':
                    ax9.text(0.05, y_pos, f"Job Openings: {latest_val:.1f}M", 
                            transform=ax9.transAxes, fontsize=9)
                elif indicator == 'Labor_Force_Participation':
                    ax9.text(0.05, y_pos, f"Participation: {latest_val:.1f}%", 
                            transform=ax9.transAxes, fontsize=9)
                y_pos -= 0.08
        
        # Add interpretation
        interpretation = f"""
💡 INTERPRETATION:
{get_labor_market_interpretation(strength_level)}"""
        
        ax9.text(0.05, y_pos-0.05, interpretation, transform=ax9.transAxes, fontsize=9,
                 verticalalignment='top', style='italic')
    
    plt.tight_layout()
    plt.savefig('labor_market_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Labor market visualization saved as 'labor_market_analysis.png'")

# ============================================================================
# MAIN ANALYSIS SCRIPT
# ============================================================================

def main():
    """Main analysis function"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Labor Market Tracker - Real-time Economic Health Gauge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 labor_market_tracker.py
  python3 labor_market_tracker.py --start 2000-01-01 --window 12
  python3 labor_market_tracker.py --min-data-points 50
        """
    )
    parser.add_argument('--start', type=str, default='1990-01-01',
                       help='Start date in YYYY-MM-DD format (default: 1990-01-01)')
    parser.add_argument('--window', type=int, default=12,
                       help='Rolling window in months for analysis (default: 12)')
    parser.add_argument('--min-data-points', type=int, default=50,
                       help='Minimum data points required for analysis (default: 50)')
    args = parser.parse_args()

    try:
        start = datetime.datetime.strptime(args.start, '%Y-%m-%d')
    except Exception:
        start = datetime.datetime(1990, 1, 1)
    end = datetime.datetime.today()

    print("💼 LABOR MARKET TRACKER - REAL-TIME ECONOMIC HEALTH")
    print("=" * 65)
    print("📊 Data Sources: Bureau of Labor Statistics via FRED")
    print(f"📅 Analysis Period: {args.start} to {end.strftime('%Y-%m-%d')}")
    print(f"📊 Rolling Window: {args.window} months")
    print(f"🎯 Min Data Points: {args.min_data_points}")
    print("=" * 65)

    # Collect labor market data
    labor_data = collect_labor_market_data(args.start)
    
    if not labor_data:
        print("❌ ERROR: Could not retrieve any labor market data")
        print("Please check your internet connection and FRED data availability")
        return
    
    # Check minimum data requirement
    max_length = max(len(series.dropna()) for series in labor_data.values())
    if max_length < args.min_data_points:
        print(f"❌ ERROR: Insufficient data points ({max_length} < {args.min_data_points})")
        print("Try reducing --min-data-points or changing --start date")
        return
    
    print(f"✅ Successfully processed labor market data")
    
    # Create composite labor market index
    composite_index, combined_df = create_labor_market_composite(labor_data, args.window)
    
    if composite_index.empty:
        print("❌ ERROR: Could not create composite labor market index")
        return
    
    print(f"📅 Data coverage: {composite_index.index[0].strftime('%Y-%m-%d')} to {composite_index.index[-1].strftime('%Y-%m-%d')}")
    
    # Generate comprehensive analysis
    print("\n📊 LABOR MARKET ANALYSIS RESULTS")
    print("=" * 55)
    
    # Latest values
    latest_date = composite_index.index[-1]
    latest_score = composite_index.iloc[-1]
    
    # Calculate percentile and classification
    percentile = (composite_index.rank(pct=True).iloc[-1]) * 100
    strength_level = classify_labor_market_strength(latest_score, percentile)
    
    print(f"📅 Latest Analysis ({latest_date.strftime('%Y-%m-%d')}):")
    print(f"   🎯 Labor Market Strength: {strength_level}")
    print(f"   📊 Composite Score: {latest_score:.2f} (Z-Score)")
    print(f"   📈 Percentile Rank: {percentile:.1f}%")
    
    # Key indicators current values
    print(f"\n📈 Current Labor Indicators:")
    key_indicators = {
        'Unemployment_Rate': ('Unemployment Rate', '%', 'lower'),
        'Job_Openings': ('Job Openings', 'M', 'higher'),
        'Labor_Force_Participation': ('Labor Force Participation', '%', 'higher'),
        'Employment_Population_Ratio': ('Employment-Population Ratio', '%', 'higher')
    }
    
    for indicator, (name, unit, direction) in key_indicators.items():
        if indicator in labor_data and not labor_data[indicator].empty:
            latest_val = labor_data[indicator].iloc[-1]
            trend_emoji = "📈" if direction == 'higher' else "📉"
            print(f"   {trend_emoji} {name}: {latest_val:.1f}{unit}")
    
    # Historical context
    if len(composite_index) > 24:
        print(f"\n📊 Historical Context:")
        print(f"   📈 Current strength higher than {percentile:.1f}% of historical values")
        
        # Recent trend
        recent_trend = np.polyfit(range(min(12, len(composite_index))), composite_index.tail(12), 1)[0]
        trend_direction = "IMPROVING" if recent_trend > 0.05 else "DETERIORATING" if recent_trend < -0.05 else "STABLE"
        print(f"   📈 Recent Trend (12M): {trend_direction} (slope: {recent_trend:.3f})")
    
    # Economic cycle context
    print(f"\n💡 INTERPRETATION:")
    print(f"   {get_labor_market_interpretation(strength_level)}")
    
    # Create comprehensive output DataFrame
    output_df = pd.DataFrame(index=composite_index.index)
    
    # Add raw labor indicators
    for name, series in labor_data.items():
        aligned_series = series.reindex(composite_index.index, method='ffill')
        output_df[f'{name}_raw'] = aligned_series
    
    # Add normalized indicators
    for col in combined_df.columns:
        if col in combined_df.columns:
            aligned_series = combined_df[col].reindex(composite_index.index)
            output_df[col] = aligned_series
    
    # Add composite metrics
    output_df['Composite_Index'] = composite_index
    output_df['Percentile_Rank'] = composite_index.rank(pct=True) * 100
    output_df['Strength_Level'] = composite_index.apply(lambda x: classify_labor_market_strength(x, composite_index.rank(pct=True).loc[composite_index.index[composite_index == x].tolist()[0]] * 100 if x in composite_index.values else 50))
    
    # Add rolling statistics
    rolling_mean = composite_index.rolling(window=args.window).mean()
    rolling_std = composite_index.rolling(window=args.window).std()
    output_df['Rolling_Mean'] = rolling_mean
    output_df['Rolling_Std'] = rolling_std
    output_df['Z_Score_12M'] = (composite_index - rolling_mean) / rolling_std
    
    # Save results
    output_df.to_csv('labor_market_analysis.csv')
    print(f"\n📁 Files saved:")
    print(f"   📊 labor_market_analysis.csv (complete data and analysis)")
    
    # Generate visualization
    create_labor_market_visualization(labor_data, composite_index, combined_df, args.window)
    print(f"   📈 labor_market_analysis.png (comprehensive visualization)")
    
    print(f"\n📊 Analysis window: {args.window} months")
    print(f"📡 Data source: Bureau of Labor Statistics via FRED")
    data_quality = (1 - output_df.isnull().sum().sum() / (len(output_df) * len(output_df.columns))) * 100
    print(f"✅ Data quality: {data_quality:.1f}% coverage")
    print("=" * 65)

if __name__ == "__main__":
    main()