#!/usr/bin/env python3
"""
Consumer Health Tracker - Economic Spending & Confidence Monitor
================================================================

This script tracks key consumer sector indicators to assess economic health and
recession risk. Consumer spending represents ~70% of US GDP, making it the most
critical sector for economic growth and stability.

Key Metrics Tracked:
- Retail Sales: Month-over-month and year-over-year growth
- Consumer Confidence: Conference Board Consumer Confidence Index
- Personal Consumption Expenditures (PCE): Real consumer spending
- Consumer Credit: Household borrowing and credit growth
- Personal Savings Rate: Consumer financial health
- Gas Prices & Energy Costs: Consumer purchasing power impact

Features:
- Real consumer data from Federal Reserve, Census Bureau, and Conference Board
- Composite consumer health scoring and classification
- Comprehensive statistical analysis and visualization
- Consumer strength classification (Very Strong → Very Weak)
- Quality validation and error handling

Consumer Health Classifications:
- Very Strong: Robust spending, high confidence, healthy credit growth
- Strong: Above-average consumer activity and sentiment
- Above Average: Moderate consumer strength with positive trends
- Average: Balanced consumer conditions, neither strong nor weak
- Below Average: Some consumer weakness emerging
- Weak: Declining consumer activity and confidence
- Very Weak: Severe consumer retrenchment, recession risk

Usage:
    python3 consumer_health_tracker.py
    python3 consumer_health_tracker.py --start 2000-01-01 --window 12
    python3 consumer_health_tracker.py --min-data-points 50

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

def get_retail_sales_data(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch retail sales indicators from FRED.
    Retail sales are a key real-time indicator of consumer spending.
    """
    retail_data = {}
    
    retail_series = {
        'RSAFS': 'Advance Retail Sales (Total)',
        'RSGASS': 'Gas Station Sales',
        'RSCCAS': 'Clothing and Clothing Accessories',
        'RSFHFS': 'Food and Beverage Sales',
        'RSAFSNA': 'Retail Sales (Not Seasonally Adjusted)'
    }
    
    print("   🛒 Fetching Retail Sales data...")
    
    for series_code, description in retail_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                series = data.iloc[:, 0].dropna()
                retail_data[series_code] = series
                print(f"      ✅ {description}: {len(series)} observations")
            else:
                print(f"      ⚠️ No data for {description}")
        except Exception as e:
            print(f"      ❌ Failed to get {description}: {str(e)[:50]}...")
    
    return retail_data

def get_consumer_confidence_data(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch consumer confidence and sentiment indicators from FRED.
    Consumer confidence leads spending decisions and economic activity.
    """
    confidence_data = {}
    
    confidence_series = {
        'CSCICP03USM665S': 'Consumer Confidence Index (OECD)',
        'UMCSENT': 'University of Michigan Consumer Sentiment',
        'USRECQ': 'US Recession Probability (Economic Stress)',
        'USSLIND': 'US Leading Economic Index',
        'PAYEMS': 'Total Nonfarm Employment (Consumer Job Security)'
    }
    
    print("   😊 Fetching Consumer Confidence data...")
    
    for series_code, description in confidence_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                series = data.iloc[:, 0].dropna()
                confidence_data[series_code] = series
                print(f"      ✅ {description}: {len(series)} observations")
            else:
                print(f"      ⚠️ No data for {description}")
        except Exception as e:
            print(f"      ❌ Failed to get {description}: {str(e)[:50]}...")
    
    return confidence_data

def get_personal_consumption_data(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch personal consumption expenditure indicators from FRED.
    PCE is the Fed's preferred measure of consumer spending.
    """
    pce_data = {}
    
    pce_series = {
        'PCE': 'Personal Consumption Expenditures',
        'PCEDG': 'PCE: Durable Goods',
        'PCEND': 'PCE: Nondurable Goods',
        'PCES': 'PCE: Services',
        'RSAFS': 'Retail Sales (Consumer Spending Proxy)',
        'MEHOINUSA672N': 'Real Median Household Income'
    }
    
    print("   💰 Fetching Personal Consumption data...")
    
    for series_code, description in pce_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                series = data.iloc[:, 0].dropna()
                pce_data[series_code] = series
                print(f"      ✅ {description}: {len(series)} observations")
            else:
                print(f"      ⚠️ No data for {description}")
        except Exception as e:
            print(f"      ❌ Failed to get {description}: {str(e)[:50]}...")
    
    return pce_data

def get_consumer_credit_data(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch consumer credit and debt indicators from FRED.
    Consumer credit health affects spending capacity and financial stability.
    """
    credit_data = {}
    
    credit_series = {
        'TOTALSL': 'Total Consumer Credit Outstanding',
        'REVOLSL': 'Revolving Consumer Credit (Credit Cards)', 
        'NONREVSL': 'Non-Revolving Consumer Credit (Auto, Student)',
        'CCLACBW027SBOG': 'Consumer Credit Loans Outstanding (Alternative)',
        'FODSP': 'Financial Obligations as % of Disposable Income'
    }
    
    print("   💳 Fetching Consumer Credit data...")
    
    for series_code, description in credit_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                series = data.iloc[:, 0].dropna()
                credit_data[series_code] = series
                print(f"      ✅ {description}: {len(series)} observations")
            else:
                print(f"      ⚠️ No data for {description}")
        except Exception as e:
            print(f"      ❌ Failed to get {description}: {str(e)[:50]}...")
    
    return credit_data

def get_savings_and_income_data(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch personal savings and income indicators from FRED.
    Savings rate and income growth affect consumer spending capacity.
    """
    savings_data = {}
    
    savings_series = {
        'PSAVERT': 'Personal Saving Rate',
        'PI': 'Personal Income',
        'DPIC96': 'Real Disposable Personal Income',
        'MEHOINUSA672N': 'Real Median Household Income',
        'WASCUR': 'Average Weekly Earnings of Production Workers'
    }
    
    print("   💰 Fetching Savings & Income data...")
    
    for series_code, description in savings_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                series = data.iloc[:, 0].dropna()
                savings_data[series_code] = series
                print(f"      ✅ {description}: {len(series)} observations")
            else:
                print(f"      ⚠️ No data for {description}")
        except Exception as e:
            print(f"      ❌ Failed to get {description}: {str(e)[:50]}...")
    
    return savings_data

def get_energy_cost_indicators(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch energy and gas price indicators that affect consumer purchasing power.
    Energy costs are a major consumer expense affecting discretionary spending.
    """
    energy_data = {}
    
    energy_series = {
        'GASREGW': 'US Regular All Formulations Gas Price',
        'DCOILWTICO': 'Crude Oil Prices: West Texas Intermediate',
        'CUURA000SA0E': 'CPI: Energy Commodities',
        'NATURALGAS': 'Natural Gas Prices',
        'CPIENGSL': 'Consumer Price Index: Energy'
    }
    
    print("   ⛽ Fetching Energy Cost data...")
    
    for series_code, description in energy_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                series = data.iloc[:, 0].dropna()
                energy_data[series_code] = series
                print(f"      ✅ {description}: {len(series)} observations")
            else:
                print(f"      ⚠️ No data for {description}")
        except Exception as e:
            print(f"      ❌ Failed to get {description}: {str(e)[:50]}...")
    
    return energy_data

def collect_consumer_data(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Collect all consumer health indicators and return as dictionary.
    """
    print("📊 Collecting comprehensive consumer health data...")
    
    all_data = {}
    
    # Collect retail sales data
    retail_data = get_retail_sales_data(start_date)
    all_data.update(retail_data)
    
    # Collect consumer confidence data
    confidence_data = get_consumer_confidence_data(start_date)
    all_data.update(confidence_data)
    
    # Collect personal consumption data
    pce_data = get_personal_consumption_data(start_date)
    all_data.update(pce_data)
    
    # Collect consumer credit data
    credit_data = get_consumer_credit_data(start_date)
    all_data.update(credit_data)
    
    # Collect savings and income data
    savings_data = get_savings_and_income_data(start_date)
    all_data.update(savings_data)
    
    # Collect energy cost data
    energy_data = get_energy_cost_indicators(start_date)
    all_data.update(energy_data)
    
    if not all_data:
        print("❌ ERROR: No consumer data could be retrieved")
        return {}
    
    print(f"✅ Successfully collected {len(all_data)} consumer health indicators")
    return all_data

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def normalize_consumer_indicators(data_dict: Dict[str, pd.Series], window: int = 12) -> Dict[str, pd.Series]:
    """
    Normalize consumer indicators using z-scores with rolling windows.
    Different indicators have different scales and interpretations.
    """
    normalized = {}
    
    # Indicators where HIGHER values mean BETTER consumer health
    positive_indicators = [
        'RSAFS', 'RSCCAS', 'RSFHFS',  # Retail sales (growth is good)
        'CSCICP03USM665S', 'UMCSENT', 'USSLIND', 'PAYEMS',  # Consumer confidence and employment
        'PCE', 'PCEDG', 'PCEND', 'PCES',  # Personal consumption
        'PI', 'DPIC96', 'MEHOINUSA672N', 'WASCUR',  # Income measures
        'TOTALSL', 'NONREVSL', 'CCLACBW027SBOG'  # Some credit growth is healthy
    ]
    
    # Indicators where LOWER values mean BETTER consumer health  
    negative_indicators = [
        'RSGASS',  # Gas station sales (high = high gas prices)
        'FODSP',  # Financial obligations burden
        'GASREGW', 'DCOILWTICO', 'CUURA000SA0E', 'CPIENGSL',  # Energy costs
        'REVOLSL', 'USRECQ'  # Excessive credit card debt and recession risk
    ]
    
    # Indicators where MODERATE values are best (inverted U-shape)
    moderate_indicators = [
        'PSAVERT'  # Savings rate (too low = no cushion, too high = not spending)
    ]
    
    for name, series in data_dict.items():
        if len(series) < window:
            print(f"⚠ Skipping {name}: insufficient data for {window}-month window")
            continue
            
        # Calculate growth rates for level series
        if name in ['RSAFS', 'PCE', 'PCEDG', 'PCEND', 'PCES', 'PI', 'TOTALSL', 'REVOLSL', 'NONREVSL']:
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
        
        # Calculate z-score
        z_score = (work_series - rolling_mean) / rolling_std
        
        # Apply directional interpretation
        if name in positive_indicators:
            normalized_score = z_score  # Higher is better
        elif name in negative_indicators:
            normalized_score = -z_score  # Lower is better, so invert
        elif name in moderate_indicators:
            # For savings rate, penalize extremes (both very high and very low are bad)
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

def create_consumer_health_composite(data_dict: Dict[str, pd.Series], window: int = 12) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Create composite consumer health score from normalized indicators.
    """
    print("🔄 Creating consumer health composite...")
    
    # Normalize all indicators
    normalized_data = normalize_consumer_indicators(data_dict, window)
    
    if not normalized_data:
        print("❌ ERROR: No normalized consumer data available")
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
        # Consumer spending indicators (40% weight) 
        'spending': ['RSAFS', 'RSGASS', 'RSCCAS', 'RSFHFS', 'RSAFSNA', 'PCE', 'PCEDG', 'PCEND', 'PCES'],
        # Consumer confidence indicators (25% weight)
        'confidence': ['CSCICP03USM665S', 'UMCSENT', 'USRECQ', 'USSLIND', 'PAYEMS'],
        # Financial health indicators (20% weight)
        'financial': ['TOTALSL', 'REVOLSL', 'NONREVSL', 'CCLACBW027SBOG', 'FODSP'],
        # Income & purchasing power (15% weight)
        'income': ['PSAVERT', 'PI', 'DPIC96', 'MEHOINUSA672N', 'WASCUR', 'GASREGW', 'DCOILWTICO', 'CUURA000SA0E', 'NATURALGAS', 'CPIENGSL']
    }
    
    weights = {'spending': 0.40, 'confidence': 0.25, 'financial': 0.20, 'income': 0.15}
    
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
    
    print(f"✅ Consumer health composite created with {len(combined_df.columns)} indicators")
    print(f"   📊 Data range: {composite_score.index[0].strftime('%Y-%m')} to {composite_score.index[-1].strftime('%Y-%m')}")
    print(f"   📈 Current score: {composite_score.iloc[-1]:.2f}")
    
    return composite_score, combined_df

def classify_consumer_health(composite_score: float, percentile: float) -> str:
    """
    Classify consumer health based on composite score and percentile.
    """
    if percentile >= 85 and composite_score >= 1.0:
        return "Very Strong"
    elif percentile >= 70 and composite_score >= 0.5:
        return "Strong"  
    elif percentile >= 55 and composite_score >= 0.0:
        return "Above Average"
    elif percentile >= 45 and composite_score >= -0.5:
        return "Average"
    elif percentile >= 30 and composite_score >= -1.0:
        return "Below Average"
    elif percentile >= 15:
        return "Weak"
    else:
        return "Very Weak"

def get_consumer_interpretation(health_level: str) -> str:
    """Get interpretation text for consumer health level."""
    interpretations = {
        "Very Strong": "Exceptional consumer spending and confidence. Strong economic growth likely.",
        "Strong": "Robust consumer activity supporting economic expansion. Positive GDP growth expected.",
        "Above Average": "Healthy consumer sector with good spending momentum and confidence levels.",
        "Average": "Balanced consumer conditions. Neither significant growth nor contraction pressure.",
        "Below Average": "Some consumer weakness emerging. Monitor for potential economic slowdown.",
        "Weak": "Consumer retrenchment underway. Economic growth likely slowing significantly.",
        "Very Weak": "Severe consumer weakness. High recession risk with declining spending and confidence."
    }
    return interpretations.get(health_level, "Consumer health status unclear.")

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_consumer_visualizations(data_dict: Dict[str, pd.Series], composite_index: pd.Series, 
                                 combined_df: pd.DataFrame, save_path: Optional[str] = None):
    """Create comprehensive consumer health visualizations."""
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 20))
    
    # Color scheme
    colors = {
        'composite': '#2E86AB',
        'retail': '#A23B72', 
        'confidence': '#F18F01',
        'spending': '#C73E1D',
        'credit': '#A288A6',
        'savings': '#73A942'
    }
    
    # 1. Consumer Health Composite (Top panel)
    ax1 = plt.subplot(6, 1, 1)
    if not composite_index.empty:
        ax1.plot(composite_index.index, composite_index, color=colors['composite'], linewidth=2.5)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax1.axhline(y=1, color='green', linestyle=':', alpha=0.5, label='Strong Threshold')
        ax1.axhline(y=-1, color='red', linestyle=':', alpha=0.5, label='Weak Threshold')
        ax1.set_title('Consumer Health Composite Score', fontweight='bold', fontsize=14)
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
    
    # 2. Retail Sales Growth
    ax2 = plt.subplot(6, 1, 2)
    retail_indicators = ['RSAFS', 'RSCCAS', 'RSFHFS']
    for indicator in retail_indicators:
        if indicator in data_dict and not data_dict[indicator].empty:
            # Calculate year-over-year growth
            growth = data_dict[indicator].pct_change(periods=12) * 100
            ax2.plot(growth.index, growth, label=indicator, linewidth=1.5, alpha=0.8)
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_title('Retail Sales Growth (Year-over-Year %)', fontweight='bold')
    ax2.set_ylabel('YoY Growth %')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Consumer Confidence
    ax3 = plt.subplot(6, 1, 3)
    confidence_indicators = ['UMCSENT', 'CSCICP03USM665S']
    for indicator in confidence_indicators:
        if indicator in data_dict and not data_dict[indicator].empty:
            series = data_dict[indicator]
            ax3.plot(series.index, series, label=indicator, linewidth=1.5, alpha=0.8)
    
    ax3.set_title('Consumer Confidence Indicators', fontweight='bold')
    ax3.set_ylabel('Index Level')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Personal Consumption Expenditures
    ax4 = plt.subplot(6, 1, 4)
    pce_indicators = ['PCEDG', 'PCEND', 'PCES']
    for indicator in pce_indicators:
        if indicator in data_dict and not data_dict[indicator].empty:
            # Calculate year-over-year growth
            growth = data_dict[indicator].pct_change(periods=12) * 100
            ax4.plot(growth.index, growth, label=indicator, linewidth=1.5, alpha=0.8)
    
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax4.set_title('Personal Consumption Expenditures Growth (YoY %)', fontweight='bold')
    ax4.set_ylabel('YoY Growth %')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Financial Health Indicators
    ax5 = plt.subplot(6, 1, 5)
    if 'PSAVERT' in data_dict and not data_dict['PSAVERT'].empty:
        ax5_twin = ax5.twinx()
        
        # Savings rate (left axis)
        savings = data_dict['PSAVERT']
        ax5.plot(savings.index, savings, color=colors['savings'], linewidth=2, label='Personal Savings Rate')
        ax5.set_ylabel('Savings Rate (%)', color=colors['savings'])
        ax5.tick_params(axis='y', labelcolor=colors['savings'])
        
        # Debt service burden (right axis, if available)
        if 'DTCOLCL' in data_dict and not data_dict['DTCOLCL'].empty:
            debt_service = data_dict['DTCOLCL']
            ax5_twin.plot(debt_service.index, debt_service, color=colors['credit'], 
                         linewidth=2, label='Debt Service Burden')
            ax5_twin.set_ylabel('Debt Service (% of Income)', color=colors['credit'])
            ax5_twin.tick_params(axis='y', labelcolor=colors['credit'])
    
    ax5.set_title('Consumer Financial Health', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Energy Costs Impact
    ax6 = plt.subplot(6, 1, 6)
    if 'GASREGW' in data_dict and not data_dict['GASREGW'].empty:
        gas_prices = data_dict['GASREGW']
        ax6.plot(gas_prices.index, gas_prices, color='red', linewidth=2, label='Gas Prices ($/gallon)')
        ax6.set_ylabel('Price ($/gallon)')
        ax6.set_title('Energy Costs (Consumer Purchasing Power Impact)', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Consumer health visualization saved: {save_path}")
    else:
        plt.savefig('consumer_health_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Consumer health visualization saved: consumer_health_analysis.png")
    
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Consumer Health Tracker - Economic Spending & Confidence Monitor')
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
    
    print("🛒 CONSUMER HEALTH TRACKER - ECONOMIC SPENDING & CONFIDENCE MONITOR")
    print("=" * 75)
    print("📊 Data Sources: Federal Reserve Economic Data (FRED), Census Bureau")
    print(f"📅 Analysis Period: {args.start} to present")
    print(f"📊 Rolling Window: {args.window} months")
    print(f"🎯 Min Data Points: {args.min_data_points}")
    print("=" * 75)

    # Collect consumer data
    consumer_data = collect_consumer_data(args.start)
    
    if not consumer_data:
        print("❌ ERROR: Could not retrieve consumer data")
        print("Please check your internet connection and FRED data availability")
        return
    
    # Check minimum data requirement
    max_length = max(len(series) for series in consumer_data.values() if not series.empty)
    if max_length < args.min_data_points:
        print(f"❌ ERROR: Insufficient data points ({max_length} < {args.min_data_points})")
        print("Try reducing --min-data-points or changing --start date")
        return
    
    print(f"✅ Successfully processed consumer data")
    print(f"📅 Max data coverage: {max_length} observations")
    
    # Create composite consumer health index
    print(f"\n🔄 Creating consumer health composite (window: {args.window} months)...")
    composite_index, combined_df = create_consumer_health_composite(consumer_data, args.window)
    
    if composite_index.empty:
        print("❌ ERROR: Could not create consumer health composite")
        return
    
    # Calculate current consumer health metrics
    if composite_index.empty or pd.isna(composite_index.iloc[-1]):
        print("❌ ERROR: Invalid composite index for current analysis")
        return
        
    latest_date = composite_index.index[-1]
    latest_score = composite_index.iloc[-1]
    percentile = (composite_index.rank(pct=True).iloc[-1]) * 100
    health_level = classify_consumer_health(latest_score, percentile)
    
    print(f"📅 Latest Analysis ({latest_date.strftime('%Y-%m-%d')}):")
    print(f"   🎯 Consumer Health: {health_level}")
    print(f"   📊 Composite Score: {latest_score:.2f} (Z-Score)")
    print(f"   📈 Percentile Rank: {percentile:.1f}%")
    
    # Key indicators current values
    print(f"\n📈 Current Consumer Indicators:")
    key_indicators = {
        'RSAFS': ('Retail Sales Growth', '% YoY', 'higher'),
        'UMCSENT': ('Consumer Sentiment', 'Index', 'higher'),
        'PCE': ('Personal Consumption Growth', '% YoY', 'higher'),
        'PSAVERT': ('Personal Savings Rate', '%', 'moderate'),
        'GASREGW': ('Gas Prices', '$/gallon', 'lower')
    }
    
    for indicator, (name, unit, direction) in key_indicators.items():
        if indicator in consumer_data and not consumer_data[indicator].empty:
            series = consumer_data[indicator]
            if indicator in ['RSAFS', 'PCE']:
                # Show growth rate for spending indicators
                latest_val = series.pct_change(periods=12).iloc[-1] * 100
                unit = '% YoY'
            else:
                latest_val = series.iloc[-1]
            
            trend_emoji = "📈" if direction == 'higher' else "📉" if direction == 'lower' else "⚖️"
            print(f"   {trend_emoji} {name}: {latest_val:.1f}{unit}")
    
    # Historical context
    if len(composite_index) > 24:
        print(f"\n📊 Historical Context:")
        print(f"   📈 Current health stronger than {percentile:.1f}% of historical values")
        
        # Recent trend
        recent_trend = np.polyfit(range(min(12, len(composite_index))), composite_index.tail(12), 1)[0]
        trend_direction = "IMPROVING" if recent_trend > 0.05 else "DETERIORATING" if recent_trend < -0.05 else "STABLE"
        print(f"   📈 Recent Trend (12M): {trend_direction} (slope: {recent_trend:.3f})")
    
    # Economic impact assessment
    print(f"\n💡 INTERPRETATION:")
    print(f"   {get_consumer_interpretation(health_level)}")
    
    # Create comprehensive output DataFrame
    output_df = pd.DataFrame(index=composite_index.index)
    
    # Add raw consumer indicators
    for name, series in consumer_data.items():
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
        lambda x: classify_consumer_health(x, (composite_index <= x).mean() * 100)
    )
    
    # Save results
    output_file = 'consumer_health_analysis.csv'
    output_df.to_csv(output_file)
    print(f"💾 Analysis results saved: {output_file}")
    
    # Create visualizations
    print(f"\n📊 Creating consumer health visualizations...")
    create_consumer_visualizations(consumer_data, composite_index, combined_df)
    
    print(f"\n✅ Consumer Health Analysis Complete!")
    print(f"📊 {len(output_df)} data points analyzed")
    print(f"🏆 Current Assessment: {health_level} Consumer Health")
    print("=" * 75)

if __name__ == "__main__":
    main()