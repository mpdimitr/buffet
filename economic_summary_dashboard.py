#!/usr/bin/env python3
"""
Enhanced Economic Summary Dashboard - Comprehensive Economic Indicators with Full Charts
========================================================================================

This script creates a comprehensive dashboard showing key economic indicators using
the actual chart implementations from individual analysis tools, providing full
detail and rich visualizations aligned over the same time period.

Features:
- SP500 and QQQ market index charts at the top
- Full Buffet Indicator chart (reusing buffet.py visualization)
- 10Y-2Y Treasury spread chart (from yield_curve_tracker.py)
- Unemployment rate chart (from labor_market_tracker.py)
- Corporate health and consumer indicators
- NBER recession shading on all panels
- Code reuse from existing analysis tools

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
import sys
import os
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
# DATA COLLECTION FUNCTIONS (Reusing Individual Tracker Logic)
# ============================================================================

def get_market_indices_data(start_date: str = '2000-01-01') -> Dict[str, pd.Series]:
    """Fetch SP500 and QQQ market index data."""
    print("📊 Fetching Market Indices data...")
    
    indices = {}
    try:
        start = pd.to_datetime(start_date)
        
        # Get SP500 data
        sp500_df = yf.download("^GSPC", start=start.strftime('%Y-%m-%d'))
        if not sp500_df.empty:
            indices['SP500'] = sp500_df[('Close', '^GSPC')].dropna()
            print(f"✓ S&P 500: {len(indices['SP500'])} daily observations")
        
        # Get NASDAQ-100 (QQQ ETF)
        qqq_df = yf.download("QQQ", start=start.strftime('%Y-%m-%d'))
        if not qqq_df.empty:
            indices['QQQ'] = qqq_df[('Close', 'QQQ')].dropna()
            print(f"✓ QQQ (NASDAQ-100): {len(indices['QQQ'])} daily observations")
            
    except Exception as e:
        print(f"❌ Error fetching market indices: {e}")
        
    return indices

def get_buffet_indicator_full_data(start_date: str = '2000-01-01') -> Dict[str, pd.Series]:
    """Get full Buffet Indicator data with all derived metrics (reusing buffet.py logic)."""
    print("📊 Fetching Buffet Indicator with full metrics...")
    
    try:
        start = pd.to_datetime(start_date)
        end = datetime.datetime.today()
        window = 60  # Default window from buffet.py
        
        # Reuse exact logic from buffet.py
        wilshire_df = yf.download("^W5000", start=start.strftime('%Y-%m-%d'))
        wilshire = wilshire_df[("Close", "^W5000")]
        gdp = pdr.DataReader('GDP', 'fred', start, end)
        
        # Align frequencies: convert both to quarterly end
        wilshire_q = wilshire.resample('QE').last().rename('Wilshire_MktCap_Bil')
        gdp_q = gdp.resample('QE').last().rename(columns={'GDP':'GDP_Bil'})
        
        # Compute Buffett Indicator
        buffett = (wilshire_q / gdp_q['GDP_Bil']) * 100.0
        
        # Derived metrics (same as buffet.py)
        roll_mean = buffett.rolling(window).mean()
        roll_std = buffett.rolling(window).std()
        z_score = (buffett - roll_mean) / roll_std
        
        # Log-linear trend
        if buffett.notna().sum() > 10:
            t_index = np.arange(len(buffett))
            mask = buffett.notna()
            coef = np.polyfit(t_index[mask], np.log(buffett[mask]), 1)
            log_trend = np.polyval(coef, t_index)
            trend_series = pd.Series(np.exp(log_trend), index=buffett.index, name='Trend')
            trend_residual = buffett / trend_series
        else:
            trend_series = pd.Series(index=buffett.index, dtype=float)
            trend_residual = pd.Series(index=buffett.index, dtype=float)
        
        result = {
            'Buffett_pct_of_GDP': buffett.dropna(),
            'RollingMean': roll_mean.dropna(),
            'RollingStd': roll_std.dropna(),
            'ZScore': z_score.dropna(),
            'Trend': trend_series.dropna(),
            'TrendResidual': trend_residual.dropna()
        }
        
        print(f"✓ Buffet Indicator with full metrics: {len(buffett)} quarterly observations")
        return result
        
    except Exception as e:
        print(f"❌ Error fetching Buffet Indicator: {e}")
        return {}

def get_yield_curve_data(start_date: str = '2000-01-01') -> Dict[str, pd.Series]:
    """Get yield curve data (reusing yield_curve_tracker.py logic)."""
    print("📊 Fetching Yield Curve data...")
    
    try:
        start = pd.to_datetime(start_date)
        
        # Reuse logic from yield_curve_tracker.py
        yields_data = {}
        
        # Treasury yield series
        series_map = {
            'DGS3MO': '3M Treasury',
            'DGS2': '2Y Treasury', 
            'DGS5': '5Y Treasury',
            'DGS10': '10Y Treasury',
            'DGS30': '30Y Treasury'
        }
        
        yields_df = pd.DataFrame()
        for series_code, description in series_map.items():
            try:
                data = pdr.get_data_fred(series_code, start=start)
                if not data.empty:
                    yields_df[series_code] = data.iloc[:, 0]
            except Exception:
                continue
        
        # Calculate spreads
        spreads = {}
        if 'DGS10' in yields_df.columns and 'DGS2' in yields_df.columns:
            spreads['10Y-2Y'] = (yields_df['DGS10'] - yields_df['DGS2']).dropna()
            print(f"✓ 10Y-2Y Spread: {len(spreads['10Y-2Y'])} daily observations")
        
        if 'DGS10' in yields_df.columns and 'DGS3MO' in yields_df.columns:
            spreads['10Y-3M'] = (yields_df['DGS10'] - yields_df['DGS3MO']).dropna()
        
        return spreads
        
    except Exception as e:
        print(f"❌ Error fetching yield curve data: {e}")
        return {}

def get_labor_market_data(start_date: str = '2000-01-01') -> Dict[str, pd.Series]:
    """Get labor market data (reusing labor_market_tracker.py logic)."""
    print("📊 Fetching Labor Market data...")
    
    try:
        start = pd.to_datetime(start_date)
        
        data = {}
        
        # Unemployment rate
        unemployment = pdr.get_data_fred('UNRATE', start=start)
        if not unemployment.empty:
            data['Unemployment_Rate'] = unemployment.iloc[:, 0].dropna()
            print(f"✓ Unemployment Rate: {len(data['Unemployment_Rate'])} monthly observations")
        
        # Job openings
        try:
            job_openings = pdr.get_data_fred('JTSJOL', start=start)
            if not job_openings.empty:
                data['Job_Openings'] = job_openings.iloc[:, 0].dropna()
                print(f"✓ Job Openings: {len(data['Job_Openings'])} monthly observations")
        except Exception:
            pass
        
        return data
        
    except Exception as e:
        print(f"❌ Error fetching labor market data: {e}")
        return {}

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

# ============================================================================
# CHART CREATION FUNCTIONS (Reusing Individual Tracker Styles)
# ============================================================================

def create_market_indices_chart(ax, indices_data: Dict[str, pd.Series], common_start, common_end):
    """Create market indices chart with separate y-axes for SP500 and QQQ."""
    add_recession_shading(ax, data_start=common_start, data_end=common_end, 
                         alpha=0.15, color='red', label_first=True)
    
    colors = {'SP500': '#1f77b4', 'QQQ': '#ff7f0e'}
    
    # Plot SP500 on the left y-axis
    if 'SP500' in indices_data and not indices_data['SP500'].empty:
        sp500_data = indices_data['SP500']
        sp500_filtered = sp500_data[(sp500_data.index >= common_start) & (sp500_data.index <= common_end)]
        if not sp500_filtered.empty:
            line1 = ax.plot(sp500_filtered.index, sp500_filtered.values, 
                           linewidth=2, color=colors['SP500'], label='S&P 500', alpha=0.8)
    
    ax.set_ylabel('S&P 500 Price ($)', color=colors['SP500'])
    ax.tick_params(axis='y', labelcolor=colors['SP500'])
    
    # Create second y-axis for QQQ
    ax2 = ax.twinx()
    
    # Plot QQQ on the right y-axis
    if 'QQQ' in indices_data and not indices_data['QQQ'].empty:
        qqq_data = indices_data['QQQ']
        qqq_filtered = qqq_data[(qqq_data.index >= common_start) & (qqq_data.index <= common_end)]
        if not qqq_filtered.empty:
            line2 = ax2.plot(qqq_filtered.index, qqq_filtered.values, 
                            linewidth=2, color=colors['QQQ'], label='QQQ (NASDAQ-100)', alpha=0.8)
    
    ax2.set_ylabel('QQQ Price ($)', color=colors['QQQ'])
    ax2.tick_params(axis='y', labelcolor=colors['QQQ'])
    
    ax.set_title('Market Indices - S&P 500 & NASDAQ-100 (QQQ) - Dual Axis', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Create combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

def create_buffet_chart(ax, buffet_data: Dict[str, pd.Series], common_start, common_end):
    """Create Buffet indicator chart reusing buffet.py style."""
    if 'Buffett_pct_of_GDP' not in buffet_data or buffet_data['Buffett_pct_of_GDP'].empty:
        return
    
    # Filter data to common range
    buffett = buffet_data['Buffett_pct_of_GDP']
    buffett_filtered = buffett[(buffett.index >= common_start) & (buffett.index <= common_end)]
    
    # Add recession shading first
    add_recession_shading(ax, data_start=common_start, data_end=common_end, 
                         alpha=0.15, color='red', label_first=False)
    
    # Main Buffet indicator line
    ax.plot(buffett_filtered.index, buffett_filtered.values, 
           linewidth=2, color='#1f77b4', label='Buffett Indicator (%)')
    
    # Add rolling mean and trend if available
    if 'RollingMean' in buffet_data and not buffet_data['RollingMean'].empty:
        rolling_mean = buffet_data['RollingMean']
        mean_filtered = rolling_mean[(rolling_mean.index >= common_start) & (rolling_mean.index <= common_end)]
        ax.plot(mean_filtered.index, mean_filtered.values, 
               linewidth=1.5, linestyle='--', color='green', alpha=0.7, label='Rolling Mean')
    
    if 'Trend' in buffet_data and not buffet_data['Trend'].empty:
        trend = buffet_data['Trend']
        trend_filtered = trend[(trend.index >= common_start) & (trend.index <= common_end)]
        ax.plot(trend_filtered.index, trend_filtered.values, 
               linewidth=1.5, linestyle=':', color='purple', alpha=0.7, label='Long-term Trend')
    
    ax.set_ylabel('Market Cap / GDP (%)')
    ax.set_title('Buffett Indicator - Market Valuation Relative to GDP', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

def create_yield_curve_chart(ax, yield_data: Dict[str, pd.Series], common_start, common_end):
    """Create 10Y-2Y yield spread chart reusing yield_curve_tracker.py style."""
    if '10Y-2Y' not in yield_data or yield_data['10Y-2Y'].empty:
        return
    
    spread_data = yield_data['10Y-2Y']
    spread_filtered = spread_data[(spread_data.index >= common_start) & (spread_data.index <= common_end)]
    
    # Add recession shading
    add_recession_shading(ax, data_start=common_start, data_end=common_end, 
                         alpha=0.15, color='red', label_first=False)
    
    # Main spread line
    ax.plot(spread_filtered.index, spread_filtered.values, 
           linewidth=1.5, color='#ff7f0e', label='10Y-2Y Spread')
    
    # Zero line and inversion shading
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Inversion Level')
    ax.fill_between(spread_filtered.index, spread_filtered.values, 0,
                   where=(spread_filtered.values < 0), color='red', alpha=0.2, label='Inverted')
    ax.fill_between(spread_filtered.index, spread_filtered.values, 0,
                   where=(spread_filtered.values >= 0), color='green', alpha=0.1, label='Normal')
    
    ax.set_ylabel('Spread (percentage points)')
    ax.set_title('10Y-2Y Treasury Spread - Primary Recession Indicator', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

def create_labor_chart(ax, labor_data: Dict[str, pd.Series], common_start, common_end):
    """Create unemployment chart reusing labor_market_tracker.py style."""
    if 'Unemployment_Rate' not in labor_data or labor_data['Unemployment_Rate'].empty:
        return
    
    unemployment = labor_data['Unemployment_Rate']
    unemployment_filtered = unemployment[(unemployment.index >= common_start) & (unemployment.index <= common_end)]
    
    # Add recession shading
    add_recession_shading(ax, data_start=common_start, data_end=common_end, 
                         alpha=0.15, color='red', label_first=False)
    
    # Main unemployment line
    ax.plot(unemployment_filtered.index, unemployment_filtered.values, 
           linewidth=2, color='#2ca02c', label='Unemployment Rate')
    
    ax.set_ylabel('Unemployment Rate (%)')
    ax.set_title('Unemployment Rate - Labor Market Health', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

def create_corporate_profits_chart(ax, corporate_data: pd.Series, common_start, common_end):
    """Create corporate profits chart."""
    if corporate_data.empty:
        return
    
    corporate_filtered = corporate_data[(corporate_data.index >= common_start) & (corporate_data.index <= common_end)]
    
    # Add recession shading
    add_recession_shading(ax, data_start=common_start, data_end=common_end, 
                         alpha=0.15, color='red', label_first=False)
    
    # Main corporate profits line
    ax.plot(corporate_filtered.index, corporate_filtered.values, 
           linewidth=2, color='#d62728', label='Corporate Profits Growth')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    ax.set_ylabel('YoY Growth (%)')
    ax.set_title('Corporate Profits Growth - Business Sector Health', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

# ============================================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================================

def create_economic_summary_dashboard(start_date: str = '2000-01-01', save_path: str = 'economic_summary_dashboard.png'):
    """
    Create enhanced economic summary dashboard with full charts.
    """
    print("🎯 ENHANCED ECONOMIC SUMMARY DASHBOARD")
    print("=" * 60)
    print(f"📅 Analysis Period: {start_date} to present")
    print("🎨 Creating comprehensive visualization with full charts...")
    print("=" * 60)
    
    # Collect all data using enhanced functions
    market_indices = get_market_indices_data(start_date)
    buffet_data = get_buffet_indicator_full_data(start_date)
    yield_data = get_yield_curve_data(start_date)
    labor_data = get_labor_market_data(start_date)
    corporate_profits = get_corporate_profits_data(start_date)
    
    # Determine common time range
    all_series = []
    
    # Add non-empty series for time range calculation
    for data in market_indices.values():
        if not data.empty:
            all_series.append(data)
    
    for data in buffet_data.values():
        if not data.empty:
            all_series.append(data)
    
    for data in yield_data.values():
        if not data.empty:
            all_series.append(data)
            
    for data in labor_data.values():
        if not data.empty:
            all_series.append(data)
    
    if not corporate_profits.empty:
        all_series.append(corporate_profits)
    
    if not all_series:
        print("❌ No data available for dashboard")
        return
    
    common_start = max([s.index[0] for s in all_series])
    common_end = min([s.index[-1] for s in all_series])
    
    print(f"📅 Common data range: {common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}")
    
    # Create figure with enhanced layout
    fig = plt.figure(figsize=(20, 24))
    
    # Create 6 subplots in a 3x2 grid
    gs = fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1, 1, 0.4], hspace=0.3)
    
    # Panel 1: Market Indices (SP500 & QQQ)
    ax1 = fig.add_subplot(gs[0])
    create_market_indices_chart(ax1, market_indices, common_start, common_end)
    
    # Panel 2: Buffet Indicator (Full Chart)
    ax2 = fig.add_subplot(gs[1])
    create_buffet_chart(ax2, buffet_data, common_start, common_end)
    
    # Panel 3: 10Y-2Y Yield Spread
    ax3 = fig.add_subplot(gs[2])
    create_yield_curve_chart(ax3, yield_data, common_start, common_end)
    
    # Panel 4: Unemployment Rate
    ax4 = fig.add_subplot(gs[3])
    create_labor_chart(ax4, labor_data, common_start, common_end)
    
    # Panel 5: Corporate Profits Growth
    ax5 = fig.add_subplot(gs[4])
    create_corporate_profits_chart(ax5, corporate_profits, common_start, common_end)
    
    # Panel 6: Summary and Interpretation
    ax6 = fig.add_subplot(gs[5])
    ax6.axis('off')
    
    # Create enhanced summary text
    summary_text = f"""ENHANCED ECONOMIC DASHBOARD - COMPREHENSIVE ANALYSIS WITH FULL CHARTS

📅 Analysis Period: {common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}
🔴 Red Shaded Areas: NBER Official Recession Periods

CHART DETAILS & INTERPRETATION:

📈 Market Indices (Top Panel):
   • S&P 500 (left axis, blue) & NASDAQ-100/QQQ (right axis, orange)
   • Dual y-axes allow comparison of relative movements despite different price scales
   • Compare trends and timing with Buffet Indicator for valuation context

🎯 Buffett Indicator (Panel 2):
   • Full chart with rolling mean and long-term trend lines (reusing buffet.py)
   • High values often precede market corrections and recessions
   • Rolling mean adapts to structural changes, trend shows secular drift

📊 10Y-2Y Treasury Spread (Panel 3):
   • Full yield curve analysis with inversion detection (from yield_curve_tracker.py)  
   • Red shading = inverted curve (negative spread)
   • Inversions typically precede recessions by 6-18 months

👥 Unemployment Rate (Panel 4):
   • Labor market health indicator (from labor_market_tracker.py style)
   • Lagging indicator - peaks often signal recession end
   • Rising unemployment confirms economic weakness

💼 Corporate Profits Growth (Panel 5):
   • Year-over-year business sector health indicator
   • Leading indicator often turns before broader economy
   • Negative growth signals business cycle weakness

🎯 KEY RELATIONSHIPS TO OBSERVE:
   1. Market peaks often coincide with Buffet Indicator extremes
   2. Yield curve inversions provide 6-18 month recession warning
   3. Corporate profits typically lead the cycle
   4. Unemployment rises during recessions (confirms weakness)
   5. Market recovery often begins before unemployment peaks

💡 TRADING/INVESTMENT INSIGHTS:
   • Use yield curve for recession timing
   • Monitor Buffet Indicator for market valuation extremes  
   • Corporate profits signal business cycle turns
   • Market indices show actual price action and trends"""
    
    ax6.text(0.02, 0.98, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    # Format x-axis for all chart panels
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=0)
    
    # Main title
    fig.suptitle('Enhanced Economic Summary Dashboard - Full Charts with Market Context', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.97)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Enhanced economic summary dashboard saved as '{save_path}'")
    print("📊 Dashboard shows full charts reused from individual analysis tools")
    print("🎯 Complete view includes market indices, detailed indicators, and recession alignment")

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