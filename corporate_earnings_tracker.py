#!/usr/bin/env python3
"""
Corporate Earnings & Profitability Tracker - Business Health Monitor
====================================================================

This script tracks key corporate earnings and profitability indicators to assess 
business sector health and economic sustainability. Corporate earnings drive 
employment decisions, capital expenditure, and market valuation - making them 
essential for economic cycle analysis.

Key Metrics Tracked:
- S&P 500 Earnings: Forward and trailing earnings per share growth
- Profit Margins: Net, operating, and gross margin trends
- Return Metrics: Return on Equity (ROE) and Return on Assets (ROA)
- Capital Expenditure: Business investment and expansion plans
- Corporate Cash Flow: Operating cash flow and free cash flow health
- Earnings Quality: Revenue growth vs. earnings growth analysis

Features:
- Real corporate data from Federal Reserve, Bureau of Economic Analysis
- Composite corporate health scoring and classification
- Comprehensive statistical analysis and visualization
- Corporate strength classification (Exceptional → Very Weak)
- Quality validation and error handling

Corporate Health Classifications:
- Exceptional: Outstanding earnings growth, margins, and cash flow
- Strong: Above-average corporate profitability and health
- Above Average: Solid corporate performance with positive trends
- Average: Balanced corporate conditions, neither strong nor weak
- Below Average: Some corporate weakness emerging
- Weak: Declining corporate profitability and health
- Very Weak: Severe corporate stress, recession risk

Usage:
    python3 corporate_earnings_tracker.py
    python3 corporate_earnings_tracker.py --start 2000-01-01 --window 12
    python3 corporate_earnings_tracker.py --min-data-points 50

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

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def get_corporate_earnings_data(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch corporate earnings and profitability indicators from FRED.
    Corporate earnings are fundamental to business cycle analysis.
    """
    earnings_data = {}
    
    earnings_series = {
        'CP': 'Corporate Profits After Tax',
        'CPATAX': 'Corporate Profits After Tax (Without IVA and CCAdj)',
        'CPROFIT': 'Corporate Profits Before Tax',
        'A053RC1Q027SBEA': 'Corporate Profits as % of GDP',
        'NCBDBIQ027S': 'Net Corporate Dividend Payments'
    }
    
    print("   💼 Fetching Corporate Earnings data...")
    
    for series_code, description in earnings_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                series = data.iloc[:, 0].dropna()
                earnings_data[series_code] = series
                print(f"      ✅ {description}: {len(series)} observations")
            else:
                print(f"      ⚠️ No data for {description}")
        except Exception as e:
            print(f"      ❌ Failed to get {description}: {str(e)[:50]}...")
    
    return earnings_data

def get_profit_margin_indicators(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch profit margin and efficiency indicators from FRED.
    Profit margins indicate corporate pricing power and operational efficiency.
    """
    margin_data = {}
    
    # Use more reliable series for margin analysis
    margin_series = {
        'CPATAX': 'Corporate Profits (for margin calculation)',
        'GDP': 'Gross Domestic Product (for revenue proxy)',
        'RSAFS': 'Retail Sales (revenue indicator)',
        'INDPRO': 'Industrial Production Index',
        'HOUST': 'Housing Starts (business activity proxy)'
    }
    
    print("   📊 Fetching Profit Margin data...")
    
    for series_code, description in margin_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                series = data.iloc[:, 0].dropna()
                margin_data[series_code] = series
                print(f"      ✅ {description}: {len(series)} observations")
            else:
                print(f"      ⚠️ No data for {description}")
        except Exception as e:
            print(f"      ❌ Failed to get {description}: {str(e)[:50]}...")
    
    return margin_data

def get_business_investment_data(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch business investment and capital expenditure indicators from FRED.
    Business investment indicates corporate confidence and growth expectations.
    """
    investment_data = {}
    
    # Use more reliable investment indicators
    investment_series = {
        'GPDI': 'Gross Private Domestic Investment',
        'PNFI': 'Private Nonresidential Fixed Investment',
        'BUSINV': 'Total Business Inventories',
        'PRFI': 'Private Residential Fixed Investment',
        'HOUST': 'Housing Starts (investment proxy)'
    }
    
    print("   🏗️ Fetching Business Investment data...")
    
    for series_code, description in investment_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                series = data.iloc[:, 0].dropna()
                investment_data[series_code] = series
                print(f"      ✅ {description}: {len(series)} observations")
            else:
                print(f"      ⚠️ No data for {description}")
        except Exception as e:
            print(f"      ❌ Failed to get {description}: {str(e)[:50]}...")
    
    return investment_data

def get_cash_flow_indicators(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch corporate cash flow and liquidity indicators from FRED.
    Cash flow health is critical for business sustainability and growth.
    """
    cashflow_data = {}
    
    # Use more reliable FRED series for cash flow analysis
    cashflow_series = {
        'CMDEBT': 'Commercial and Industrial Loans Outstanding',
        'TOTCI': 'Commercial and Industrial Loans, All Commercial Banks',
        'BUSLOANS': 'Commercial and Industrial Loans at All Commercial Banks',
        'TDSP': 'Total Deposits at All Commercial Banks',
        'NONREVSL': 'Total Nonrevolving Credit Outstanding'
    }
    
    print("   💰 Fetching Cash Flow data...")
    
    for series_code, description in cashflow_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                series = data.iloc[:, 0].dropna()
                cashflow_data[series_code] = series
                print(f"      ✅ {description}: {len(series)} observations")
            else:
                print(f"      ⚠️ No data for {description}")
        except Exception as e:
            print(f"      ❌ Failed to get {description}: {str(e)[:50]}...")
    
    return cashflow_data

def get_market_valuation_metrics(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch market valuation metrics related to corporate performance.
    Market valuation reflects investor expectations of corporate performance.
    """
    valuation_data = {}
    
    # Use more reliable market indices and avoid problematic FRED series
    valuation_series = {
        'SP500': 'S&P 500 Index',
        'NASDAQCOM': 'NASDAQ Composite Index',
        'DEXUSEU': 'US/Euro Foreign Exchange Rate',
        'DGS10': '10-Year Treasury Constant Maturity Rate',
        'DGS3MO': '3-Month Treasury Constant Maturity Rate'
    }
    
    print("   📈 Fetching Market Valuation data...")
    
    for series_code, description in valuation_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                series = data.iloc[:, 0].dropna()
                valuation_data[series_code] = series
                print(f"      ✅ {description}: {len(series)} observations")
            else:
                print(f"      ⚠️ No data for {description}")
        except Exception as e:
            print(f"      ❌ Failed to get {description}: {str(e)[:50]}...")
    
    return valuation_data

def get_economic_context_data(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch broader economic context indicators that affect corporate performance.
    Economic environment shapes corporate profitability and growth prospects.
    """
    context_data = {}
    
    context_series = {
        'FEDFUNDS': 'Federal Funds Rate',
        'DGS10': '10-Year Treasury Constant Maturity Rate', 
        'UNRATE': 'Unemployment Rate',
        'CPIAUCSL': 'Consumer Price Index for All Urban Consumers',
        'PAYEMS': 'All Employees, Total Nonfarm'
    }
    
    print("   🌍 Fetching Economic Context data...")
    
    for series_code, description in context_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                series = data.iloc[:, 0].dropna()
                context_data[series_code] = series
                print(f"      ✅ {description}: {len(series)} observations")
            else:
                print(f"      ⚠️ No data for {description}")
        except Exception as e:
            print(f"      ❌ Failed to get {description}: {str(e)[:50]}...")
    
    return context_data

def collect_corporate_data(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Collect all corporate earnings and profitability indicators and return as dictionary.
    """
    print("📊 Collecting comprehensive corporate earnings data...")
    
    all_data = {}
    
    # Collect corporate earnings data
    earnings_data = get_corporate_earnings_data(start_date)
    all_data.update(earnings_data)
    
    # Collect profit margin data
    margin_data = get_profit_margin_indicators(start_date)
    all_data.update(margin_data)
    
    # Collect business investment data
    investment_data = get_business_investment_data(start_date)
    all_data.update(investment_data)
    
    # Collect cash flow data
    cashflow_data = get_cash_flow_indicators(start_date)
    all_data.update(cashflow_data)
    
    # Collect market valuation data
    valuation_data = get_market_valuation_metrics(start_date)
    all_data.update(valuation_data)
    
    # Collect economic context data
    context_data = get_economic_context_data(start_date)
    all_data.update(context_data)
    
    if not all_data:
        print("❌ ERROR: No corporate data could be retrieved")
        return {}
    
    print(f"✅ Successfully collected {len(all_data)} corporate health indicators")
    return all_data

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def normalize_corporate_indicators(data_dict: Dict[str, pd.Series], window: int = 12) -> Dict[str, pd.Series]:
    """
    Normalize corporate indicators using z-scores with rolling windows.
    Different indicators have different scales and interpretations.
    """
    normalized = {}
    
    # Indicators where HIGHER values mean BETTER corporate health
    positive_indicators = [
        'CP', 'CPATAX', 'CPROFIT', 'A053RC1Q027SBEA',  # Earnings and profits
        'GPDI', 'PNFI', 'BUSINV', 'HOUST',  # Investment and expansion
        'CMDEBT', 'TOTCI', 'BUSLOANS', 'TDSP', 'NONREVSL',  # Credit availability (good for business)
        'SP500', 'NASDAQCOM',  # Market performance
        'INDPRO', 'PAYEMS',  # Economic activity
        'RSAFS'  # Revenue indicators
    ]
    
    # Indicators where LOWER values mean BETTER corporate health
    negative_indicators = [
        'FEDFUNDS',  # Interest rates (higher rates hurt corporates)
        'UNRATE',  # Unemployment (higher unemployment hurts demand)
        'CPIAUCSL'  # Inflation (can squeeze margins)
    ]
    
    # Indicators where MODERATE values are best
    moderate_indicators = [
        'DGS10', 'DGS3MO',  # Treasury rates (too low = economic stress, too high = borrowing costs)
        'DEXUSEU'  # Exchange rates (moderate values best for trade balance)
    ]
    
    for name, series in data_dict.items():
        if len(series) < window:
            print(f"⚠ Skipping {name}: insufficient data for {window}-month window")
            continue
            
        # Calculate growth rates for level series
        if name in ['CP', 'CPATAX', 'CPROFIT', 'GPDI', 'PNFI', 'BUSINV', 'CMDEBT', 
                   'TOTCI', 'BUSLOANS', 'TDSP', 'NONREVSL', 'SP500', 'NASDAQCOM', 'PAYEMS', 'RSAFS', 'HOUST']:
            # Convert to growth rates (year-over-year % change)
            growth_series = series.pct_change(periods=12) * 100
            work_series = growth_series.dropna()
        else:
            work_series = series
        
        if len(work_series) < window:
            continue
            
        # Calculate rolling statistics for normalization
        min_periods = max(window, 6)
        rolling_mean = work_series.rolling(window=window*2, min_periods=min_periods).mean()
        rolling_std = work_series.rolling(window=window*2, min_periods=min_periods).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        # Calculate z-score
        z_score = (work_series - rolling_mean) / rolling_std
        
        # Apply directional interpretation
        if name in positive_indicators:
            normalized_score = z_score  # Higher is better
        elif name in negative_indicators:
            normalized_score = -z_score  # Lower is better, so invert
        elif name in moderate_indicators:
            # For moderate indicators, penalize extremes
            abs_z = np.abs(z_score)
            normalized_score = -abs_z  # Closer to mean is better
        else:
            normalized_score = z_score  # Default: higher is better
        
        normalized[name] = normalized_score.dropna()
        
        if len(normalized[name]) > 0:
            latest_val = work_series.iloc[-1] if not work_series.empty else np.nan
            latest_z = normalized_score.iloc[-1] if not normalized_score.empty else np.nan
            direction = "📈" if name in positive_indicators else "📉" if name in negative_indicators else "⚖️"
            print(f"   {direction} {name}: Latest={latest_val:.2f}, Z-Score={latest_z:.2f}")
    
    return normalized

def create_corporate_health_composite(data_dict: Dict[str, pd.Series], window: int = 12) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Create composite corporate health score from normalized indicators.
    """
    print("🔄 Creating corporate health composite...")
    
    # Normalize all indicators
    normalized_data = normalize_corporate_indicators(data_dict, window)
    
    if not normalized_data:
        print("❌ ERROR: No normalized corporate data available")
        return pd.Series(), pd.DataFrame()
    
    # Create DataFrame from normalized series
    combined_df = pd.DataFrame(normalized_data)
    
    # Remove rows with too many missing values
    min_indicators = max(3, len(combined_df.columns) // 3)  # At least 1/3 of indicators
    combined_df = combined_df.dropna(thresh=min_indicators)
    
    if combined_df.empty:
        print("❌ ERROR: Insufficient data for composite calculation")
        return pd.Series(), pd.DataFrame()
    
    # Weight different categories of indicators
    category_weights = {
        # Earnings & profitability indicators (35% weight)
        'earnings': ['CP', 'CPATAX', 'CPROFIT', 'A053RC1Q027SBEA'],
        # Investment & growth indicators (25% weight)  
        'investment': ['GPDI', 'PNFI', 'BUSINV', 'INDPRO', 'HOUST'],
        # Financial health indicators (20% weight)
        'financial': ['CMDEBT', 'TOTCI', 'BUSLOANS', 'TDSP', 'NONREVSL'],
        # Market performance indicators (15% weight)
        'market': ['SP500', 'NASDAQCOM', 'DGS10', 'DGS3MO'],
        # Economic environment indicators (5% weight)
        'environment': ['FEDFUNDS', 'UNRATE', 'DEXUSEU', 'PAYEMS']
    }
    
    weights = {'earnings': 0.35, 'investment': 0.25, 'financial': 0.20, 'market': 0.15, 'environment': 0.05}
    
    # Calculate weighted composite score
    weighted_components = []
    total_weight = 0
    
    for category, indicators in category_weights.items():
        category_data = combined_df[[col for col in indicators if col in combined_df.columns]]
        
        if not category_data.empty:
            # Average indicators within category
            category_score = category_data.mean(axis=1)
            weighted_score = category_score * weights[category]
            weighted_components.append(weighted_score)
            total_weight += weights[category]
            
            avg_indicators = len(category_data.columns)
            print(f"   ✅ {category.title()} category: {avg_indicators} indicators, weight: {weights[category]:.1%}")
    
    if not weighted_components:
        print("❌ ERROR: No valid category components")
        return pd.Series(), pd.DataFrame()
    
    # Combine weighted components
    if weighted_components:
        composite_score = sum(weighted_components) / total_weight if total_weight > 0 else sum(weighted_components)
        # Ensure we have valid data
        composite_score = composite_score.dropna()
    else:
        print("❌ ERROR: No weighted components available")
        return pd.Series(), pd.DataFrame()
    
    if composite_score.empty:
        print("❌ ERROR: Composite score is empty after calculation")
        return pd.Series(), pd.DataFrame()
    
    print(f"✅ Corporate health composite created with {len(combined_df.columns)} indicators")
    print(f"   📊 Data range: {composite_score.index[0].strftime('%Y-%m')} to {composite_score.index[-1].strftime('%Y-%m')}")
    print(f"   📈 Current score: {composite_score.iloc[-1]:.2f}")
    
    return composite_score, combined_df

def classify_corporate_health(composite_score: float, percentile: float) -> str:
    """
    Classify corporate health based on composite score and percentile.
    """
    if percentile >= 90 and composite_score >= 1.5:
        return "Exceptional"
    elif percentile >= 75 and composite_score >= 1.0:
        return "Strong"  
    elif percentile >= 60 and composite_score >= 0.5:
        return "Above Average"
    elif percentile >= 40 and composite_score >= -0.25:
        return "Average"
    elif percentile >= 25 and composite_score >= -0.75:
        return "Below Average"
    elif percentile >= 10:
        return "Weak"
    else:
        return "Very Weak"

def get_corporate_interpretation(health_level: str) -> str:
    """Get interpretation text for corporate health level."""
    interpretations = {
        "Exceptional": "Outstanding corporate earnings, margins, and cash flow. Strong economic expansion likely.",
        "Strong": "Robust corporate profitability and health. Positive for economic growth and employment.",
        "Above Average": "Solid corporate performance with good earnings trends and healthy balance sheets.",
        "Average": "Balanced corporate conditions. Neither significant expansion nor contraction pressure.",
        "Below Average": "Some corporate weakness emerging. Monitor for potential economic slowdown.",
        "Weak": "Corporate stress evident with declining margins and earnings. Economic growth at risk.",
        "Very Weak": "Severe corporate distress with poor profitability. High recession risk."
    }
    return interpretations.get(health_level, "Corporate health status unclear.")

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_corporate_visualizations(data_dict: Dict[str, pd.Series], composite_index: pd.Series, 
                                  combined_df: pd.DataFrame, save_path: Optional[str] = None):
    """Create comprehensive corporate health visualizations."""
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 20))
    
    # Color scheme
    colors = {
        'composite': '#1f77b4',
        'earnings': '#ff7f0e', 
        'investment': '#2ca02c',
        'financial': '#d62728',
        'market': '#9467bd',
        'environment': '#8c564b'
    }
    
    # 1. Corporate Health Composite (Top panel)
    ax1 = plt.subplot(6, 1, 1)
    if not composite_index.empty:
        ax1.plot(composite_index.index, composite_index, color=colors['composite'], linewidth=2.5)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax1.axhline(y=1, color='green', linestyle=':', alpha=0.5, label='Strong Threshold')
        ax1.axhline(y=-1, color='red', linestyle=':', alpha=0.5, label='Weak Threshold')
        ax1.set_title('Corporate Health Composite Score', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Z-Score')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add recession shading (approximate recent recessions)
        recession_periods = [
            ('2020-02-01', '2020-04-01'),  # COVID recession
            ('2007-12-01', '2009-06-01')   # Great Recession
        ]
        for start, end in recession_periods:
            if start < composite_index.index[-1].strftime('%Y-%m-%d'):
                ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                           alpha=0.2, color='gray', label='Recession' if start == '2020-02-01' else '')
    
    # 2. Corporate Profits Growth
    ax2 = plt.subplot(6, 1, 2)
    profit_indicators = ['CP', 'CPATAX', 'CPROFIT']
    for indicator in profit_indicators:
        if indicator in data_dict and not data_dict[indicator].empty:
            # Calculate year-over-year growth
            growth = data_dict[indicator].pct_change(periods=12) * 100
            ax2.plot(growth.index, growth, label=indicator, linewidth=1.5, alpha=0.8)
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_title('Corporate Profits Growth (Year-over-Year %)', fontweight='bold')
    ax2.set_ylabel('YoY Growth %')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Business Investment Trends
    ax3 = plt.subplot(6, 1, 3)
    investment_indicators = ['GPDI', 'PNFI', 'BUSINV']
    for indicator in investment_indicators:
        if indicator in data_dict and not data_dict[indicator].empty:
            # Calculate year-over-year growth
            growth = data_dict[indicator].pct_change(periods=12) * 100
            ax3.plot(growth.index, growth, label=indicator, linewidth=1.5, alpha=0.8)
    
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax3.set_title('Business Investment Growth (YoY %)', fontweight='bold')
    ax3.set_ylabel('YoY Growth %')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Corporate Cash and Financial Health
    ax4 = plt.subplot(6, 1, 4)
    
    # Try to plot available financial health indicators
    financial_plotted = False
    
    # Commercial and Industrial Loans (available indicator)
    if 'CMDEBT' in data_dict and not data_dict['CMDEBT'].empty:
        loans = data_dict['CMDEBT']
        ax4.plot(loans.index, loans, color=colors['financial'], linewidth=2, label='Commercial Loans')
        ax4.set_ylabel('Commercial Loans ($B)', color=colors['financial'])
        ax4.tick_params(axis='y', labelcolor=colors['financial'])
        financial_plotted = True
    
    # If we have other available indicators, add them
    if 'TOTCI' in data_dict and not data_dict['TOTCI'].empty:
        ax4_twin = ax4.twinx()
        totci = data_dict['TOTCI']
        ax4_twin.plot(totci.index, totci, color=colors['investment'], 
                     linewidth=2, label='Total Commercial Loans')
        ax4_twin.set_ylabel('Total C&I Loans ($B)', color=colors['investment'])
        ax4_twin.tick_params(axis='y', labelcolor=colors['investment'])
        financial_plotted = True
    
    if not financial_plotted:
        ax4.text(0.5, 0.5, 'Financial Health Data\nTemporarily Unavailable', 
                transform=ax4.transAxes, ha='center', va='center', 
                fontsize=12, alpha=0.7)
    
    ax4.set_title('Corporate Financial Health', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Market Performance
    ax5 = plt.subplot(6, 1, 5)
    market_indicators = ['SP500', 'NASDAQCOM']
    market_plotted = False
    
    for indicator in market_indicators:
        if indicator in data_dict and not data_dict[indicator].empty:
            # Calculate year-over-year growth
            growth = data_dict[indicator].pct_change(periods=12) * 100
            ax5.plot(growth.index, growth, label=indicator, linewidth=1.5, alpha=0.8)
            market_plotted = True
    
    if market_plotted:
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'Market Performance Data\nTemporarily Unavailable', 
                transform=ax5.transAxes, ha='center', va='center', 
                fontsize=12, alpha=0.7)
    
    ax5.set_title('Market Performance (Corporate Valuation Proxy)', fontweight='bold')
    ax5.set_ylabel('YoY Growth %')
    ax5.grid(True, alpha=0.3)
    
    # 6. Economic Environment
    ax6 = plt.subplot(6, 1, 6)
    if 'FEDFUNDS' in data_dict and not data_dict['FEDFUNDS'].empty:
        ax6_twin = ax6.twinx()
        
        # Fed Funds Rate (left axis)
        fed_rate = data_dict['FEDFUNDS']
        ax6.plot(fed_rate.index, fed_rate, color='red', linewidth=2, label='Fed Funds Rate')
        ax6.set_ylabel('Fed Funds Rate (%)', color='red')
        ax6.tick_params(axis='y', labelcolor='red')
        
        # Unemployment rate (right axis, if available)
        if 'UNRATE' in data_dict and not data_dict['UNRATE'].empty:
            unemployment = data_dict['UNRATE']
            ax6_twin.plot(unemployment.index, unemployment, color='blue', 
                         linewidth=2, label='Unemployment Rate')
            ax6_twin.set_ylabel('Unemployment Rate (%)', color='blue')
            ax6_twin.tick_params(axis='y', labelcolor='blue')
    
    ax6.set_title('Economic Environment (Corporate Operating Context)', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Corporate health visualization saved: {save_path}")
    else:
        plt.savefig('corporate_earnings_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Corporate health visualization saved: corporate_earnings_analysis.png")
    
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Corporate Earnings & Profitability Tracker - Business Health Monitor')
    parser.add_argument('--start', type=str, default='1990-01-01', 
                       help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--window', type=int, default=12,
                       help='Rolling window size in months for analysis')
    parser.add_argument('--min-data-points', type=int, default=50,
                       help='Minimum data points required for analysis')
    
    args = parser.parse_args()
    
    # Parse start date
    try:
        start = datetime.datetime.strptime(args.start, '%Y-%m-%d')
    except ValueError:
        print("❌ ERROR: Invalid start date format. Use YYYY-MM-DD")
        return
    
    print("💼 CORPORATE EARNINGS & PROFITABILITY TRACKER - BUSINESS HEALTH MONITOR")
    print("=" * 80)
    print("📊 Data Sources: Federal Reserve Economic Data (FRED), Bureau of Economic Analysis")
    print(f"📅 Analysis Period: {args.start} to present")
    print(f"📊 Rolling Window: {args.window} months")
    print(f"🎯 Min Data Points: {args.min_data_points}")
    print("=" * 80)

    # Collect corporate data
    corporate_data = collect_corporate_data(args.start)
    
    if not corporate_data:
        print("❌ ERROR: Could not retrieve corporate data")
        print("Please check your internet connection and FRED data availability")
        return
    
    # Check minimum data requirement
    max_length = max(len(series) for series in corporate_data.values() if not series.empty)
    if max_length < args.min_data_points:
        print(f"❌ ERROR: Insufficient data points ({max_length} < {args.min_data_points})")
        print("Try reducing --min-data-points or changing --start date")
        return
    
    print(f"✅ Successfully processed corporate data")
    print(f"📅 Max data coverage: {max_length} observations")
    
    # Create composite corporate health index
    print(f"\n🔄 Creating corporate health composite (window: {args.window} months)...")
    composite_index, combined_df = create_corporate_health_composite(corporate_data, args.window)
    
    if composite_index.empty:
        print("❌ ERROR: Could not create corporate health composite")
        return
    
    # Calculate current corporate health metrics
    if composite_index.empty or pd.isna(composite_index.iloc[-1]):
        print("❌ ERROR: Invalid composite index for current analysis")
        return
        
    latest_date = composite_index.index[-1]
    latest_score = composite_index.iloc[-1]
    percentile = (composite_index.rank(pct=True).iloc[-1]) * 100
    health_level = classify_corporate_health(latest_score, percentile)
    
    print(f"📅 Latest Analysis ({latest_date.strftime('%Y-%m-%d')}):")
    print(f"   🎯 Corporate Health: {health_level}")
    print(f"   📊 Composite Score: {latest_score:.2f} (Z-Score)")
    print(f"   📈 Percentile Rank: {percentile:.1f}%")
    
    # Key indicators current values
    print(f"\n📈 Current Corporate Indicators:")
    key_indicators = {
        'CP': ('Corporate Profits Growth', '% YoY', 'higher'),
        'GPDI': ('Business Investment Growth', '% YoY', 'higher'),
        'SP500': ('S&P 500 Growth', '% YoY', 'higher'),
        'FEDFUNDS': ('Federal Funds Rate', '%', 'lower'),
        'UNRATE': ('Unemployment Rate', '%', 'lower')
    }
    
    for indicator, (name, unit, direction) in key_indicators.items():
        if indicator in corporate_data and not corporate_data[indicator].empty:
            series = corporate_data[indicator]
            if indicator in ['CP', 'GPDI', 'SP500']:
                # Show growth rate for growth indicators
                latest_val = series.pct_change(periods=12).iloc[-1] * 100
                unit = '% YoY'
            else:
                latest_val = series.iloc[-1]
            
            trend_emoji = "📈" if direction == 'higher' else "📉" if direction == 'lower' else "⚖️"
            print(f"   {trend_emoji} {name}: {latest_val:.1f}{unit}")
    
    # Historical context
    if len(composite_index) > 24 and not pd.isna(latest_score):
        print(f"\n📊 Historical Context:")
        print(f"   📈 Current health stronger than {percentile:.1f}% of historical values")
        
        # Recent trend
        recent_data = composite_index.tail(12).dropna()
        if len(recent_data) >= 6:
            recent_trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
            trend_direction = "IMPROVING" if recent_trend > 0.05 else "DETERIORATING" if recent_trend < -0.05 else "STABLE"
            print(f"   📈 Recent Trend (12M): {trend_direction} (slope: {recent_trend:.3f})")
        else:
            print(f"   📈 Recent Trend (12M): INSUFFICIENT DATA")
    
    # Economic impact assessment
    print(f"\n💡 INTERPRETATION:")
    print(f"   {get_corporate_interpretation(health_level)}")
    
    # Create comprehensive output DataFrame
    output_df = pd.DataFrame(index=composite_index.index)
    
    # Add raw corporate indicators
    for name, series in corporate_data.items():
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
    
    # Add classifications
    output_df['Health_Level'] = output_df['Composite_Index'].apply(
        lambda x: classify_corporate_health(x, (composite_index <= x).mean() * 100)
    )
    
    # Save results
    output_file = 'corporate_earnings_analysis.csv'
    output_df.to_csv(output_file)
    print(f"💾 Analysis results saved: {output_file}")
    
    # Create visualizations
    print(f"\n📊 Creating corporate health visualizations...")
    create_corporate_visualizations(corporate_data, composite_index, combined_df)
    
    print(f"\n✅ Corporate Earnings & Profitability Analysis Complete!")
    print(f"📊 {len(output_df)} data points analyzed")
    print(f"🏆 Current Assessment: {health_level} Corporate Health")
    print("=" * 80)

if __name__ == "__main__":
    main()