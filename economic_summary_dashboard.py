#!/usr/bin/env python3
"""
Economic Summary Dashboard - Comprehensive Economic Indicators Alignment
=======================================================================

This script creates a comprehensive dashboard showing key economic indicators
aligned over the same time period, making it easy to see relationships between:
- Buffet Indicator (market valuation)
- Yield Curve (10Y-2Y spread)
- Unemployment Rate 
- Corporate Earnings Health
- Consumer Health
- Recession periods (NBER shading)

All indicators are synchronized to the same time axis for easy comparison
and correlation analysis during different economic cycles.

Features:
- Synchronized time series from 2000 onwards
- NBER recession shading on all panels
- Standardized Y-axis scaling for comparison
- Professional styling with clear legends
- Export as high-resolution PNG

Usage:
    python3 economic_summary_dashboard.py
    python3 economic_summary_dashboard.py --start 2000-01-01
    python3 economic_summary_dashboard.py --start 1990-01-01 --save-path summary.png

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
import yfinance as yf
import warnings
from typing import Dict, Optional, Tuple, List

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

def add_recession_shading(ax, data_start=None, data_end=None, alpha=0.2, color='red', label_first=True):
    """
    Add NBER recession shading to a matplotlib axis.
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
# DATA COLLECTION FUNCTIONS
# ============================================================================

def get_buffet_indicator_data(start_date: str = '2000-01-01') -> pd.Series:
    """Fetch Buffet Indicator data (Market Cap / GDP)."""
    print("📊 Fetching Buffet Indicator data...")
    
    try:
        start = pd.to_datetime(start_date)
        end = datetime.datetime.today()
        
        # Get Wilshire 5000 and GDP data
        wilshire_df = yf.download("^W5000", start=start.strftime('%Y-%m-%d'))
        wilshire = wilshire_df[("Close", "^W5000")]
        gdp = pdr.DataReader('GDP', 'fred', start, end)
        
        # Convert to quarterly and calculate Buffet indicator
        wilshire_q = wilshire.resample('QE').last()
        gdp_q = gdp.resample('QE').last()['GDP']
        buffet = (wilshire_q / gdp_q) * 100.0
        
        print(f"✓ Buffet Indicator: {len(buffet)} quarterly observations")
        return buffet.dropna()
        
    except Exception as e:
        print(f"❌ Error fetching Buffet Indicator: {e}")
        return pd.Series()

def get_yield_spread_data(start_date: str = '2000-01-01') -> pd.Series:
    """Fetch 10Y-2Y Treasury spread data."""
    print("📊 Fetching Yield Curve data...")
    
    try:
        start = pd.to_datetime(start_date)
        
        # Get Treasury yields
        dgs10 = pdr.get_data_fred('DGS10', start=start)
        dgs2 = pdr.get_data_fred('DGS2', start=start)
        
        # Calculate 10Y-2Y spread
        spread = dgs10['DGS10'] - dgs2['DGS2']
        spread = spread.dropna()
        
        print(f"✓ 10Y-2Y Spread: {len(spread)} daily observations")
        return spread
        
    except Exception as e:
        print(f"❌ Error fetching yield spread: {e}")
        return pd.Series()

def get_unemployment_data(start_date: str = '2000-01-01') -> pd.Series:
    """Fetch unemployment rate data."""
    print("📊 Fetching Unemployment data...")
    
    try:
        start = pd.to_datetime(start_date)
        
        unemployment = pdr.get_data_fred('UNRATE', start=start)['UNRATE']
        unemployment = unemployment.dropna()
        
        print(f"✓ Unemployment Rate: {len(unemployment)} monthly observations")
        return unemployment
        
    except Exception as e:
        print(f"❌ Error fetching unemployment: {e}")
        return pd.Series()

def get_corporate_profits_data(start_date: str = '2000-01-01') -> pd.Series:
    """Fetch corporate profits data."""
    print("📊 Fetching Corporate Profits data...")
    
    try:
        start = pd.to_datetime(start_date)
        
        # Corporate profits after tax
        profits = pdr.get_data_fred('A446RC1Q027SBEA', start=start)
        if profits.empty:
            # Alternative: corporate profits before tax
            profits = pdr.get_data_fred('A445RC1Q027SBEA', start=start)
        
        if not profits.empty:
            profits = profits.iloc[:, 0].dropna()
            # Calculate year-over-year growth rate
            profits_growth = profits.pct_change(periods=4) * 100  # YoY % change
            
            print(f"✓ Corporate Profits Growth: {len(profits_growth)} quarterly observations")
            return profits_growth.dropna()
        
    except Exception as e:
        print(f"❌ Error fetching corporate profits: {e}")
        
    return pd.Series()

def get_consumer_confidence_data(start_date: str = '2000-01-01') -> pd.Series:
    """Fetch consumer confidence data."""
    print("📊 Fetching Consumer Confidence data...")
    
    try:
        start = pd.to_datetime(start_date)
        
        # University of Michigan Consumer Sentiment
        confidence = pdr.get_data_fred('UMCSENT', start=start)
        if not confidence.empty:
            confidence = confidence.iloc[:, 0].dropna()
            print(f"✓ Consumer Confidence: {len(confidence)} monthly observations")
            return confidence
        
    except Exception as e:
        print(f"❌ Error fetching consumer confidence: {e}")
        
    return pd.Series()

# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================

def create_economic_summary_dashboard(start_date: str = '2000-01-01', save_path: str = 'economic_summary_dashboard.png'):
    """
    Create comprehensive economic summary dashboard.
    """
    print("🎯 ECONOMIC SUMMARY DASHBOARD")
    print("=" * 60)
    print(f"📅 Analysis Period: {start_date} to present")
    print("🎨 Creating synchronized economic indicators visualization...")
    print("=" * 60)
    
    # Collect all data
    buffet_data = get_buffet_indicator_data(start_date)
    yield_spread = get_yield_spread_data(start_date)
    unemployment = get_unemployment_data(start_date)
    corporate_profits = get_corporate_profits_data(start_date)
    consumer_confidence = get_consumer_confidence_data(start_date)
    
    # Determine common time range
    all_series = [s for s in [buffet_data, yield_spread, unemployment, corporate_profits, consumer_confidence] if not s.empty]
    if not all_series:
        print("❌ No data available for dashboard")
        return
        
    common_start = max([s.index[0] for s in all_series])
    common_end = min([s.index[-1] for s in all_series])
    
    print(f"📅 Common data range: {common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(6, 1, figsize=(16, 20), sharex=True)
    
    # Define color scheme
    colors = {
        'buffet': '#1f77b4',      # Blue
        'yield': '#ff7f0e',       # Orange  
        'unemployment': '#2ca02c', # Green
        'corporate': '#d62728',    # Red
        'consumer': '#9467bd',     # Purple
        'recession': '#ff6b6b'     # Light red
    }
    
    # Panel 1: Buffet Indicator
    ax1 = axes[0]
    if not buffet_data.empty:
        buffet_filtered = buffet_data[(buffet_data.index >= common_start) & (buffet_data.index <= common_end)]
        add_recession_shading(ax1, data_start=common_start, data_end=common_end, 
                             alpha=0.15, color=colors['recession'], label_first=True)
        
        ax1.plot(buffet_filtered.index, buffet_filtered.values, 
                linewidth=2, color=colors['buffet'], label='Buffet Indicator')
        ax1.set_ylabel('Market Cap / GDP (%)')
        ax1.set_title('Buffet Indicator - Market Valuation Relative to GDP', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Panel 2: 10Y-2Y Yield Spread
    ax2 = axes[1]
    if not yield_spread.empty:
        yield_filtered = yield_spread[(yield_spread.index >= common_start) & (yield_spread.index <= common_end)]
        add_recession_shading(ax2, data_start=common_start, data_end=common_end, 
                             alpha=0.15, color=colors['recession'], label_first=False)
        
        ax2.plot(yield_filtered.index, yield_filtered.values, 
                linewidth=1.5, color=colors['yield'], label='10Y-2Y Spread')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Inversion Level')
        ax2.fill_between(yield_filtered.index, yield_filtered.values, 0,
                        where=(yield_filtered.values < 0), color='red', alpha=0.2, label='Inverted')
        ax2.set_ylabel('Spread (percentage points)')
        ax2.set_title('10Y-2Y Treasury Spread - Yield Curve Inversion Indicator', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # Panel 3: Unemployment Rate
    ax3 = axes[2]
    if not unemployment.empty:
        unemployment_filtered = unemployment[(unemployment.index >= common_start) & (unemployment.index <= common_end)]
        add_recession_shading(ax3, data_start=common_start, data_end=common_end, 
                             alpha=0.15, color=colors['recession'], label_first=False)
        
        ax3.plot(unemployment_filtered.index, unemployment_filtered.values, 
                linewidth=2, color=colors['unemployment'], label='Unemployment Rate')
        ax3.set_ylabel('Unemployment Rate (%)')
        ax3.set_title('Unemployment Rate - Labor Market Health', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # Panel 4: Corporate Profits Growth
    ax4 = axes[3]
    if not corporate_profits.empty:
        corporate_filtered = corporate_profits[(corporate_profits.index >= common_start) & (corporate_profits.index <= common_end)]
        add_recession_shading(ax4, data_start=common_start, data_end=common_end, 
                             alpha=0.15, color=colors['recession'], label_first=False)
        
        ax4.plot(corporate_filtered.index, corporate_filtered.values, 
                linewidth=2, color=colors['corporate'], label='Corporate Profits Growth')
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax4.set_ylabel('YoY Growth (%)')
        ax4.set_title('Corporate Profits Growth - Business Sector Health', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # Panel 5: Consumer Confidence
    ax5 = axes[4]
    if not consumer_confidence.empty:
        consumer_filtered = consumer_confidence[(consumer_confidence.index >= common_start) & (consumer_confidence.index <= common_end)]
        add_recession_shading(ax5, data_start=common_start, data_end=common_end, 
                             alpha=0.15, color=colors['recession'], label_first=False)
        
        ax5.plot(consumer_filtered.index, consumer_filtered.values, 
                linewidth=2, color=colors['consumer'], label='Consumer Confidence')
        ax5.set_ylabel('Index Level')
        ax5.set_title('Consumer Confidence - Consumer Sentiment and Spending Outlook', fontweight='bold', fontsize=14)
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    
    # Panel 6: Economic Correlation Summary
    ax6 = axes[5]
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"""ECONOMIC INDICATORS CORRELATION ANALYSIS
    
📅 Analysis Period: {common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}
🔴 Red Shaded Areas: NBER Official Recession Periods

KEY RELATIONSHIPS TO OBSERVE:

🎯 Buffet Indicator (Market Valuation):
   • High values often precede market corrections
   • Peaks before recessions, troughs after recessions
   • Compare with other indicators for timing signals

📈 10Y-2Y Yield Spread (Recession Predictor):
   • Inversions (negative values) typically precede recessions by 6-18 months
   • Most reliable leading indicator of economic downturns
   • Fed policy and expectations reflected in curve shape

👥 Unemployment Rate (Lagging Indicator):
   • Rises during and after recessions (lagging indicator)
   • Peak unemployment often signals recession end
   • Key measure of labor market health and recovery

💼 Corporate Profits Growth (Business Health):
   • Leading indicator of economic cycles
   • Negative growth often precedes broader economic weakness
   • Recovery typically leads overall economic recovery

🛍️ Consumer Confidence (Spending Outlook):
   • Forward-looking sentiment indicator
   • Drops before recessions as consumers become cautious
   • Recovery indicates returning economic optimism

💡 INTERPRETATION FRAMEWORK:
   1. Watch for yield curve inversions as early warning
   2. Monitor Buffet Indicator for market overvaluation
   3. Corporate profits signal business cycle turns
   4. Consumer confidence reflects sentiment shifts
   5. Unemployment confirms cycle phases (lagging)

🎯 Use this dashboard to identify:
   • Leading indicators turning before recessions
   • Correlation patterns across economic cycles  
   • Market timing and valuation assessment opportunities
   • Economic cycle phase identification"""
    
    ax6.text(0.02, 0.98, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    # Format x-axis
    for ax in axes[:-1]:  # All except summary panel
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=0)
    
    # Main title
    fig.suptitle('Economic Summary Dashboard - Key Indicators with Recession Alignment', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.97)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Economic summary dashboard saved as '{save_path}'")
    print("📊 Dashboard shows synchronized view of key economic indicators")
    print("🎯 Use this view to analyze relationships across economic cycles")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Economic Summary Dashboard')
    parser.add_argument('--start', type=str, default='2000-01-01', 
                       help='Start date in YYYY-MM-DD format (default: 2000-01-01)')
    parser.add_argument('--save-path', type=str, default='economic_summary_dashboard.png',
                       help='Output file path (default: economic_summary_dashboard.png)')
    
    args = parser.parse_args()
    
    try:
        start_date = pd.to_datetime(args.start).strftime('%Y-%m-%d')
    except Exception:
        raise ValueError('Invalid start date format. Use YYYY-MM-DD.')
    
    create_economic_summary_dashboard(start_date, args.save_path)

if __name__ == "__main__":
    main()