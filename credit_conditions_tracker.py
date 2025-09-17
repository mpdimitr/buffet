#!/usr/bin/env python3
"""
Credit Conditions Tracker - Financial System Health Monitor
===========================================================

This script tracks key credit market indicators to assess the health and tightness
of financial conditions. Credit availability drives economic activity - when credit
tightens, economic growth slows; when credit is loose, growth accelerates.

Key Metrics Tracked:
- Corporate Credit Spreads: Investment grade and high yield bond spreads
- Bank Lending Standards: Fed's Senior Loan Officer Opinion Survey (SLOOS)
- Commercial & Industrial Loans: Business lending growth
- Consumer Credit: Household borrowing trends
- Mortgage Rates vs Treasury: Housing credit conditions
- Term Structure of Credit: Short vs long-term credit availability

Features:
- Real credit data from Federal Reserve and market sources
- Composite credit conditions scoring and classification
- Comprehensive statistical analysis and visualization
- Credit tightness classification (Extremely Tight → Extremely Loose)
- Quality validation and error handling

Usage:
    python3 credit_conditions_tracker.py
    python3 credit_conditions_tracker.py --start 2000-01-01 --window 12
    python3 credit_conditions_tracker.py --min-data-points 50

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

def get_corporate_credit_spreads(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch corporate credit spreads from FRED.
    Higher spreads indicate tighter credit conditions.
    """
    credit_spreads = {}
    
    # Key corporate credit spread series
    spread_series = {
        'BAMLH0A0HYM2': 'High Yield Corporate Spread',
        'BAMLC0A0CM': 'Investment Grade Corporate Spread',
        'BAMLH0A1HYBB': 'BB High Yield Spread',
        'BAMLC0A1CAAAEY': 'AAA Corporate Spread',
    }
    
    print("📊 Fetching Corporate Credit Spreads from FRED...")
    
    for series_code, description in spread_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                credit_spreads[series_code] = data.iloc[:, 0].dropna()
                print(f"✓ Successfully retrieved {description} ({len(data)} records)")
            else:
                print(f"⚠ No data available for {description}")
        except Exception as e:
            print(f"❌ Failed to get {description}: {str(e)[:50]}...")
    
    return credit_spreads

def get_bank_lending_standards(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch bank lending standards from FRED Senior Loan Officer Opinion Survey.
    Higher values indicate tighter lending standards.
    """
    lending_standards = {}
    
    # Bank lending standards series (SLOOS data)
    # Note: FRED series codes change periodically, these are the most stable ones
    lending_series = {
        'DRTSCILM': 'Commercial & Industrial Loans - Large/Medium Firms',
        'DRTSCIS': 'Commercial & Industrial Loans - Small Firms', 
        'DRTSCLCC': 'Commercial Real Estate Loans',
        # Try alternative series codes for other loan types
        'DRTSCONSUMER': 'Consumer Loans (Alternative)',  # May not exist
        'DRTSRE': 'Real Estate Loans (Alternative)',     # May not exist
    }
    
    print("📊 Fetching Bank Lending Standards from FRED...")
    
    for series_code, description in lending_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                lending_standards[series_code] = data.iloc[:, 0].dropna()
                print(f"✓ Successfully retrieved {description} ({len(data)} records)")
            else:
                print(f"⚠ No data available for {description}")
        except Exception as e:
            print(f"⚠ Skipping {description}: Series may be discontinued ({str(e)[:30]}...)")
    
    # If we didn't get much lending standards data, that's OK - the tracker adapts
    if not lending_standards:
        print("⚠ No bank lending standards data available - using credit spreads and loan growth")
    
    return lending_standards

def get_loan_growth_indicators(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch loan growth indicators from FRED.
    Higher growth indicates looser credit conditions.
    """
    loan_growth = {}
    
    # Loan outstanding series
    loan_series = {
        'BUSLOANS': 'Commercial & Industrial Loans Outstanding',
        'CONSUMER': 'Consumer Credit Outstanding',
        'REALLN': 'Real Estate Loans Outstanding', 
        'TOTCI': 'Total Credit to Private Non-Financial Sector',
    }
    
    print("📊 Fetching Loan Growth Indicators from FRED...")
    
    for series_code, description in loan_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                # Calculate year-over-year growth rate
                loan_level = data.iloc[:, 0]
                loan_growth_rate = loan_level.pct_change(periods=12) * 100  # YoY % change
                loan_growth[f'{series_code}_Growth'] = loan_growth_rate.dropna()
                print(f"✓ Successfully calculated {description} Growth ({len(loan_growth_rate.dropna())} records)")
            else:
                print(f"⚠ No data available for {description}")
        except Exception as e:
            print(f"❌ Failed to get {description}: {str(e)[:50]}...")
    
    return loan_growth

def get_interest_rate_indicators(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch interest rate and credit pricing indicators.
    """
    rate_indicators = {}
    
    # Interest rate series
    rate_series = {
        'AAA': 'AAA Corporate Bond Yield',
        'BAA': 'BAA Corporate Bond Yield',
        'MORTGAGE30US': '30-Year Fixed Mortgage Rate',
        'DPRIME': 'Bank Prime Loan Rate',
        'DGS10': '10-Year Treasury Rate',
    }
    
    print("📊 Fetching Interest Rate Indicators from FRED...")
    
    for series_code, description in rate_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                rate_indicators[series_code] = data.iloc[:, 0].dropna()
                print(f"✓ Successfully retrieved {description} ({len(data)} records)")
            else:
                print(f"⚠ No data available for {description}")
        except Exception as e:
            print(f"❌ Failed to get {description}: {str(e)[:50]}...")
    
    return rate_indicators

def calculate_credit_spreads(rate_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    """
    Calculate additional credit spreads from interest rate data.
    """
    calculated_spreads = {}
    
    # Calculate spreads if we have the data
    if 'AAA' in rate_data and 'DGS10' in rate_data:
        aaa_spread = rate_data['AAA'] - rate_data['DGS10']
        calculated_spreads['AAA_Treasury_Spread'] = aaa_spread.dropna()
        print("✓ Calculated AAA-Treasury Spread")
    
    if 'BAA' in rate_data and 'DGS10' in rate_data:
        baa_spread = rate_data['BAA'] - rate_data['DGS10']
        calculated_spreads['BAA_Treasury_Spread'] = baa_spread.dropna()
        print("✓ Calculated BAA-Treasury Spread")
    
    if 'BAA' in rate_data and 'AAA' in rate_data:
        credit_risk_spread = rate_data['BAA'] - rate_data['AAA']
        calculated_spreads['BAA_AAA_Spread'] = credit_risk_spread.dropna()
        print("✓ Calculated BAA-AAA Credit Risk Spread")
    
    if 'MORTGAGE30US' in rate_data and 'DGS10' in rate_data:
        mortgage_spread = rate_data['MORTGAGE30US'] - rate_data['DGS10']
        calculated_spreads['Mortgage_Treasury_Spread'] = mortgage_spread.dropna()
        print("✓ Calculated Mortgage-Treasury Spread")
    
    return calculated_spreads

def collect_credit_conditions_data(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Collect all credit conditions indicators and return as dictionary.
    """
    print("📊 Collecting comprehensive credit conditions data...")
    
    all_data = {}
    
    # Get corporate credit spreads
    credit_spreads = get_corporate_credit_spreads(start_date)
    all_data.update(credit_spreads)
    
    # Get bank lending standards
    lending_standards = get_bank_lending_standards(start_date)
    all_data.update(lending_standards)
    
    # Get loan growth indicators  
    loan_growth = get_loan_growth_indicators(start_date)
    all_data.update(loan_growth)
    
    # Get interest rate indicators
    rate_indicators = get_interest_rate_indicators(start_date)
    all_data.update(rate_indicators)
    
    # Calculate additional spreads
    calculated_spreads = calculate_credit_spreads(rate_indicators)
    all_data.update(calculated_spreads)
    
    if not all_data:
        print("❌ ERROR: No credit conditions data could be retrieved")
        return {}
    
    print(f"✅ Successfully collected {len(all_data)} credit indicators")
    return all_data

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def normalize_credit_indicators(data_dict: Dict[str, pd.Series], window: int = 12) -> Dict[str, pd.Series]:
    """
    Normalize credit indicators using rolling z-scores.
    Handle directional interpretation correctly.
    """
    normalized_data = {}
    
    # Indicators where HIGHER values mean TIGHTER credit (worse for economy)
    tightening_indicators = [
        'BAMLH0A0HYM2', 'BAMLC0A0CM', 'BAMLH0A1HYBB', 'BAMLC0A1CAAAEY',  # Credit spreads
        'DRTSCILM', 'DRTSCIS', 'DRTSCLCC', 'DRTSRC', 'DRTSCONSUMER',      # Lending standards
        'AAA_Treasury_Spread', 'BAA_Treasury_Spread', 'BAA_AAA_Spread',    # Calculated spreads
        'Mortgage_Treasury_Spread', 'AAA', 'BAA', 'MORTGAGE30US', 'DPRIME' # Interest rates
    ]
    
    # Indicators where HIGHER values mean LOOSER credit (better for economy)  
    loosening_indicators = [
        'BUSLOANS_Growth', 'CONSUMER_Growth', 'REALLN_Growth', 'TOTCI_Growth'  # Loan growth
    ]
    
    for name, series in data_dict.items():
        if len(series) < window:
            print(f"⚠ Skipping {name}: insufficient data for {window}-month window")
            continue
            
        # Calculate rolling statistics
        min_periods = max(window, 6)  # At least 6 periods for stability
        rolling_mean = series.rolling(window=window*2, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window*2, min_periods=min_periods).std()
        
        # Calculate z-score
        z_score = (series - rolling_mean) / rolling_std
        
        # For tightening indicators, flip the z-score (higher spreads/rates = tighter credit = worse)
        if name in tightening_indicators:
            z_score = -z_score  # Flip so positive z-score = loose credit conditions
        
        direction = "tightening" if name in tightening_indicators else "loosening"
        normalized_data[f'{name}_norm'] = z_score.dropna()
        print(f"✓ Normalized {name} ({direction} indicator) - {len(z_score.dropna())} points")
    
    return normalized_data

def create_credit_conditions_composite(data_dict: Dict[str, pd.Series], window: int = 12) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Create composite credit conditions score.
    """
    print("🔄 Creating composite credit conditions index...")
    
    # Normalize all indicators
    normalized_data = normalize_credit_indicators(data_dict, window)
    
    if not normalized_data:
        print("❌ ERROR: No normalized credit data available")
        return pd.Series(), pd.DataFrame()
    
    # Combine into DataFrame
    combined_df = pd.DataFrame(normalized_data)
    
    # Remove sparse data
    min_data_threshold = 0.3  # Require 30% data availability
    before_cleaning = len(combined_df.columns)
    
    for col in combined_df.columns:
        if combined_df[col].notna().sum() / len(combined_df) < min_data_threshold:
            combined_df = combined_df.drop(columns=[col])
            print(f"⚠ Removed {col}: insufficient data coverage")
    
    print(f"🧹 Data cleaning: {before_cleaning} → {len(combined_df.columns)} indicators")
    
    if combined_df.empty:
        print("❌ ERROR: No credit indicators passed data quality checks")
        return pd.Series(), pd.DataFrame()
    
    # Define weights for different credit indicators
    base_weights = {
        # Corporate credit spreads (high weight - market-based, real-time)
        'BAMLH0A0HYM2_norm': 0.15,           # High Yield Spread
        'BAMLC0A0CM_norm': 0.12,             # Investment Grade Spread
        'BAA_Treasury_Spread_norm': 0.10,     # BAA-Treasury Spread
        
        # Bank lending standards (medium-high weight - forward-looking)
        'DRTSCILM_norm': 0.08,               # C&I Lending Standards Large/Medium
        'DRTSCIS_norm': 0.08,                # C&I Lending Standards Small
        'DRTSCLCC_norm': 0.06,               # Commercial RE Lending Standards
        
        # Loan growth (medium weight - real economy impact)
        'BUSLOANS_Growth_norm': 0.10,        # Business Loan Growth
        'CONSUMER_Growth_norm': 0.08,        # Consumer Credit Growth
        'REALLN_Growth_norm': 0.06,          # Real Estate Loan Growth
        
        # Interest rate indicators (lower weight - influenced by Fed policy)
        'AAA_norm': 0.05,                    # AAA Corporate Rate
        'BAA_norm': 0.05,                    # BAA Corporate Rate
        'MORTGAGE30US_norm': 0.04,           # Mortgage Rate
        'DPRIME_norm': 0.03,                 # Prime Rate
    }
    
    # Calculate available components and adjust weights
    available_components = [col for col in base_weights.keys() if col in combined_df.columns]
    available_weights = {comp: base_weights[comp] for comp in available_components}
    
    # Add equal weights for any additional components not in base_weights
    remaining_components = [col for col in combined_df.columns if col not in available_weights]
    if remaining_components:
        remaining_weight = 0.1 / len(remaining_components) if remaining_components else 0
        for comp in remaining_components:
            available_weights[comp] = remaining_weight
    
    # Normalize weights to sum to 1.0
    total_weight = sum(available_weights.values())
    if total_weight > 0:
        adjusted_weights = {comp: weight/total_weight for comp, weight in available_weights.items()}
    else:
        print("❌ ERROR: No weighted components available")
        return pd.Series(), pd.DataFrame()
    
    print(f"📊 Using {len(available_components)} primary components")
    weight_strings = [f'{comp.replace("_norm", "")}: {weight:.3f}' for comp, weight in list(adjusted_weights.items())[:6]]
    print(f"⚖️ Top weights: {', '.join(weight_strings)}...")
    
    # Calculate composite index
    composite_index = pd.Series(0.0, index=combined_df.index)
    
    for component, weight in adjusted_weights.items():
        component_data = combined_df[component].fillna(0)
        composite_index += component_data * weight
    
    composite_index = composite_index.dropna()
    
    if composite_index.empty:
        print("❌ ERROR: Composite credit conditions index calculation failed")
        return pd.Series(), pd.DataFrame()
    
    print(f"✅ Created composite credit conditions index with {len(composite_index)} data points")
    return composite_index, combined_df

def classify_credit_conditions(composite_score: float, percentile: float) -> str:
    """
    Classify credit conditions based on composite score and percentile.
    """
    if pd.isna(composite_score) or pd.isna(percentile):
        return "Unknown"
    
    # Classification based on z-score
    if composite_score > 2.0:
        return "Extremely Loose"
    elif composite_score > 1.5:
        return "Very Loose"
    elif composite_score > 1.0:
        return "Loose"
    elif composite_score > 0.5:
        return "Moderately Loose"
    elif composite_score > -0.5:
        return "Normal"
    elif composite_score > -1.0:
        return "Moderately Tight"
    elif composite_score > -1.5:
        return "Tight"
    elif composite_score > -2.0:
        return "Very Tight"
    else:
        return "Extremely Tight"

def get_credit_interpretation(conditions_level: str) -> str:
    """Get interpretation text for credit conditions level."""
    interpretations = {
        "Extremely Loose": "Credit extremely easy to obtain.\nLow spreads, loose standards, rapid loan growth.",
        "Very Loose": "Credit very accessible.\nFavorable lending conditions across markets.",
        "Loose": "Credit readily available.\nGood financing conditions for businesses and consumers.",
        "Moderately Loose": "Credit somewhat easier than normal.\nGenerally favorable lending environment.",
        "Normal": "Credit conditions balanced.\nTypical lending standards and spreads.",
        "Moderately Tight": "Credit somewhat restricted.\nModestly higher standards and spreads.",
        "Tight": "Credit conditions restrictive.\nElevated spreads and tighter lending standards.",
        "Very Tight": "Credit significantly constrained.\nHigh spreads and strict lending criteria.",
        "Extremely Tight": "Credit severely restricted.\nCredit crisis conditions with limited access.",
        "Unknown": "Insufficient data for assessment."
    }
    return interpretations.get(conditions_level, "Data quality insufficient for interpretation.")

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_credit_conditions_visualization(data_dict: Dict[str, pd.Series], composite_index: pd.Series, 
                                         combined_df: pd.DataFrame, window: int = 12) -> None:
    """
    Create comprehensive credit conditions visualization.
    """
    print("📈 Generating comprehensive credit conditions visualization...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    
    # Define color scheme
    colors = {
        'loose': '#2E8B57',       # Sea Green (good for economy)
        'normal': '#4682B4',      # Steel Blue
        'tight': '#DC143C',       # Crimson (bad for economy)
        'neutral': '#708090'      # Slate Gray
    }
    
    # Subplot 1: High Yield Credit Spreads
    ax1 = plt.subplot(3, 4, 1)
    if 'BAMLH0A0HYM2' in data_dict:
        hy_spread = data_dict['BAMLH0A0HYM2'].dropna()
        ax1.plot(hy_spread.index, hy_spread.values, linewidth=1.5, color=colors['tight'])
        ax1.fill_between(hy_spread.index, hy_spread.values, alpha=0.3, color=colors['tight'])
        ax1.set_title('High Yield Credit Spread (%)\n(Lower = Looser Credit)')
        ax1.set_ylabel('Spread (%)')
        ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Investment Grade Credit Spreads
    ax2 = plt.subplot(3, 4, 2)
    if 'BAMLC0A0CM' in data_dict:
        ig_spread = data_dict['BAMLC0A0CM'].dropna()
        ax2.plot(ig_spread.index, ig_spread.values, linewidth=1.5, color=colors['tight'])
        ax2.fill_between(ig_spread.index, ig_spread.values, alpha=0.3, color=colors['tight'])
        ax2.set_title('Investment Grade Spread (%)\n(Lower = Looser Credit)')
        ax2.set_ylabel('Spread (%)')
        ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Bank Lending Standards
    ax3 = plt.subplot(3, 4, 3)
    if 'DRTSCILM' in data_dict:
        lending_std = data_dict['DRTSCILM'].dropna()
        ax3.plot(lending_std.index, lending_std.values, linewidth=1.5, color=colors['tight'])
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.fill_between(lending_std.index, lending_std.values, 0,
                        where=(lending_std.values > 0), color=colors['tight'], alpha=0.3, label='Tightening')
        ax3.fill_between(lending_std.index, lending_std.values, 0,
                        where=(lending_std.values <= 0), color=colors['loose'], alpha=0.3, label='Loosening')
        ax3.set_title('Bank Lending Standards\nC&I Loans (>0 = Tightening)')
        ax3.set_ylabel('Net % Tightening')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # Subplot 4: Business Loan Growth
    ax4 = plt.subplot(3, 4, 4)
    if 'BUSLOANS_Growth' in data_dict:
        loan_growth = data_dict['BUSLOANS_Growth'].dropna()
        positive = loan_growth >= 0
        ax4.bar(loan_growth[positive].index, loan_growth[positive].values, 
               width=20, color=colors['loose'], alpha=0.7, label='Growth')
        ax4.bar(loan_growth[~positive].index, loan_growth[~positive].values,
               width=20, color=colors['tight'], alpha=0.7, label='Contraction')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_title('Business Loan Growth (%)\n(Higher = Looser Credit)')
        ax4.set_ylabel('YoY Growth (%)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # Subplot 5: Consumer Credit Growth
    ax5 = plt.subplot(3, 4, 5)
    if 'CONSUMER_Growth' in data_dict:
        consumer_growth = data_dict['CONSUMER_Growth'].dropna()
        ax5.plot(consumer_growth.index, consumer_growth.values, linewidth=1.5, color=colors['normal'])
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax5.fill_between(consumer_growth.index, consumer_growth.values, 0,
                        where=(consumer_growth.values >= 0), color=colors['loose'], alpha=0.3)
        ax5.fill_between(consumer_growth.index, consumer_growth.values, 0,
                        where=(consumer_growth.values < 0), color=colors['tight'], alpha=0.3)
        ax5.set_title('Consumer Credit Growth (%)\n(Higher = Looser Credit)')
        ax5.set_ylabel('YoY Growth (%)')
        ax5.grid(True, alpha=0.3)
    
    # Subplot 6: Corporate Bond Yields
    ax6 = plt.subplot(3, 4, 6)
    if 'AAA' in data_dict and 'BAA' in data_dict:
        aaa_yield = data_dict['AAA'].dropna()
        baa_yield = data_dict['BAA'].dropna()
        ax6.plot(aaa_yield.index, aaa_yield.values, linewidth=1.5, label='AAA Corporate', color=colors['loose'])
        ax6.plot(baa_yield.index, baa_yield.values, linewidth=1.5, label='BAA Corporate', color=colors['tight'])
        ax6.set_title('Corporate Bond Yields (%)\n(Lower = Looser Credit)')
        ax6.set_ylabel('Yield (%)')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    # Subplot 7: Composite Credit Conditions Index
    ax7 = plt.subplot(3, 4, 7)
    if not composite_index.empty:
        # Color code by conditions level
        composite_colors = []
        for score in composite_index.values:
            if score > 1.5:
                composite_colors.append(colors['loose'])
            elif score > 0.5:
                composite_colors.append('#90EE90')  # Light Green
            elif score > -0.5:
                composite_colors.append(colors['normal'])
            elif score > -1.5:
                composite_colors.append('#FFA500')  # Orange
            else:
                composite_colors.append(colors['tight'])
        
        ax7.scatter(composite_index.index, composite_index.values, c=composite_colors, alpha=0.6, s=8)
        ax7.plot(composite_index.index, composite_index.values, linewidth=1, color='gray', alpha=0.5)
        
        # Add condition level lines
        ax7.axhline(y=1.5, color=colors['loose'], linestyle='--', alpha=0.5, label='Very Loose')
        ax7.axhline(y=0.5, color='#90EE90', linestyle='--', alpha=0.5, label='Loose')
        ax7.axhline(y=-0.5, color='#FFA500', linestyle='--', alpha=0.5, label='Tight')
        ax7.axhline(y=-1.5, color=colors['tight'], linestyle='--', alpha=0.5, label='Very Tight')
    
    ax7.set_title('Composite Credit Conditions\n(Z-Score)')
    ax7.set_ylabel('Composite Score')
    ax7.grid(True, alpha=0.3)
    ax7.legend(loc='upper left', fontsize=8)
    
    # Subplot 8: Credit Spreads Comparison
    ax8 = plt.subplot(3, 4, 8)
    spread_indicators = ['BAA_Treasury_Spread', 'AAA_Treasury_Spread', 'Mortgage_Treasury_Spread']
    for indicator in spread_indicators:
        if indicator in data_dict:
            spread_data = data_dict[indicator].dropna()
            if not spread_data.empty:
                label = indicator.replace('_', '-').replace('Spread', '')
                ax8.plot(spread_data.index, spread_data.values, 
                        linewidth=1.2, label=label, alpha=0.8)
    
    ax8.set_title('Treasury Spreads (%)\n(Lower = Looser Credit)')
    ax8.set_ylabel('Spread (%)')
    ax8.grid(True, alpha=0.3)
    ax8.legend(loc='upper left', fontsize=8)
    
    # Subplot 9: Normalized Credit Indicators
    ax9 = plt.subplot(3, 4, 9)
    if not combined_df.empty:
        # Show key normalized indicators
        key_indicators = [col for col in combined_df.columns if any(key in col for key in 
                         ['BAMLH0A0HYM2', 'DRTSCILM', 'BUSLOANS_Growth', 'BAA_Treasury'])][:4]
        
        for col in key_indicators:
            if col in combined_df.columns:
                indicator_data = combined_df[col].dropna()
                if not indicator_data.empty:
                    label = col.replace('_norm', '').replace('_', ' ')[:15]
                    ax9.plot(indicator_data.index, indicator_data.values, 
                            linewidth=1.2, label=label, alpha=0.8)
        
        ax9.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax9.set_title('Key Normalized Indicators\n(Z-Scores)')
        ax9.set_ylabel('Z-Score')
        ax9.grid(True, alpha=0.3)
        ax9.legend(loc='upper left', fontsize=7)
    
    # Subplot 10: Current Status Summary
    ax10 = plt.subplot(3, 4, 10)
    ax10.axis('off')
    
    # Get latest data for summary
    if not composite_index.empty:
        latest_date = composite_index.index[-1]
        latest_score = composite_index.iloc[-1]
        
        # Calculate percentile
        percentile = (composite_index.rank(pct=True).iloc[-1]) * 100
        conditions_level = classify_credit_conditions(latest_score, percentile)
        
        summary_text = f"""CREDIT CONDITIONS STATUS
        
📅 Latest: {latest_date.strftime('%Y-%m-%d')}

🎯 CONDITIONS: {conditions_level.upper()}
📊 Score: {latest_score:.2f} (Z-Score)
📈 Percentile: {percentile:.1f}%

KEY SPREADS:"""
        
        y_pos = 0.95
        ax10.text(0.05, y_pos, summary_text, transform=ax10.transAxes, fontsize=11, 
                 verticalalignment='top', fontweight='bold')
        
        y_pos -= 0.45
        
        # Show latest values for key indicators
        key_indicators = ['BAMLH0A0HYM2', 'BAMLC0A0CM', 'BAA_Treasury_Spread']
        for indicator in key_indicators:
            if indicator in data_dict and not data_dict[indicator].empty:
                latest_val = data_dict[indicator].iloc[-1]
                if 'BAMLH0A0HYM2' in indicator:
                    ax10.text(0.05, y_pos, f"High Yield: {latest_val:.2f}%", 
                            transform=ax10.transAxes, fontsize=9)
                elif 'BAMLC0A0CM' in indicator:
                    ax10.text(0.05, y_pos, f"Inv Grade: {latest_val:.2f}%", 
                            transform=ax10.transAxes, fontsize=9)
                elif 'BAA_Treasury_Spread' in indicator:
                    ax10.text(0.05, y_pos, f"BAA Spread: {latest_val:.2f}%", 
                            transform=ax10.transAxes, fontsize=9)
                y_pos -= 0.08
        
        # Add interpretation
        interpretation = f"""
💡 INTERPRETATION:
{get_credit_interpretation(conditions_level)}"""
        
        ax10.text(0.05, y_pos-0.05, interpretation, transform=ax10.transAxes, fontsize=9,
                 verticalalignment='top', style='italic')
    
    # Subplot 11: Lending Standards Detail
    ax11 = plt.subplot(3, 4, 11)
    lending_indicators = ['DRTSCILM', 'DRTSCIS', 'DRTSCLCC']
    for indicator in lending_indicators:
        if indicator in data_dict:
            lending_data = data_dict[indicator].dropna()
            if not lending_data.empty:
                if 'DRTSCILM' in indicator:
                    label = 'C&I Large/Med'
                elif 'DRTSCIS' in indicator:
                    label = 'C&I Small'
                elif 'DRTSCLCC' in indicator:
                    label = 'Commercial RE'
                else:
                    label = indicator
                ax11.plot(lending_data.index, lending_data.values, 
                         linewidth=1.2, label=label, alpha=0.8)
    
    ax11.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax11.set_title('Bank Lending Standards\n(>0 = Tightening)')
    ax11.set_ylabel('Net % Tightening')
    ax11.grid(True, alpha=0.3)
    ax11.legend(loc='upper left', fontsize=8)
    
    # Subplot 12: Loan Growth Comparison
    ax12 = plt.subplot(3, 4, 12)
    growth_indicators = ['BUSLOANS_Growth', 'CONSUMER_Growth', 'REALLN_Growth']
    for indicator in growth_indicators:
        if indicator in data_dict:
            growth_data = data_dict[indicator].dropna()
            if not growth_data.empty:
                label = indicator.replace('_Growth', '').replace('BUSLOANS', 'Business').replace('REALLN', 'Real Estate')
                ax12.plot(growth_data.index, growth_data.values, 
                         linewidth=1.2, label=label, alpha=0.8)
    
    ax12.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax12.set_title('Loan Growth by Category (%)\n(Higher = Looser Credit)')
    ax12.set_ylabel('YoY Growth (%)')
    ax12.grid(True, alpha=0.3)
    ax12.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('credit_conditions_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Credit conditions visualization saved as 'credit_conditions_analysis.png'")

# ============================================================================
# MAIN ANALYSIS SCRIPT
# ============================================================================

def main():
    """Main analysis function"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Credit Conditions Tracker - Financial System Health Monitor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 credit_conditions_tracker.py
  python3 credit_conditions_tracker.py --start 2000-01-01 --window 12
  python3 credit_conditions_tracker.py --min-data-points 50
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

    print("💳 CREDIT CONDITIONS TRACKER - FINANCIAL SYSTEM HEALTH")
    print("=" * 68)
    print("📊 Data Sources: Federal Reserve Economic Data (FRED)")
    print(f"📅 Analysis Period: {args.start} to {end.strftime('%Y-%m-%d')}")
    print(f"📊 Rolling Window: {args.window} months")
    print(f"🎯 Min Data Points: {args.min_data_points}")
    print("=" * 68)

    # Collect credit conditions data
    credit_data = collect_credit_conditions_data(args.start)
    
    if not credit_data:
        print("❌ ERROR: Could not retrieve any credit conditions data")
        print("Please check your internet connection and FRED data availability")
        return
    
    # Check minimum data requirement
    max_length = max(len(series.dropna()) for series in credit_data.values())
    if max_length < args.min_data_points:
        print(f"❌ ERROR: Insufficient data points ({max_length} < {args.min_data_points})")
        print("Try reducing --min-data-points or changing --start date")
        return
    
    print(f"✅ Successfully processed credit conditions data")
    
    # Create composite credit conditions index
    composite_index, combined_df = create_credit_conditions_composite(credit_data, args.window)
    
    if composite_index.empty:
        print("❌ ERROR: Could not create composite credit conditions index")
        return
    
    print(f"📅 Data coverage: {composite_index.index[0].strftime('%Y-%m-%d')} to {composite_index.index[-1].strftime('%Y-%m-%d')}")
    
    # Generate comprehensive analysis
    print("\n📊 CREDIT CONDITIONS ANALYSIS RESULTS")
    print("=" * 58)
    
    # Latest values
    latest_date = composite_index.index[-1]
    latest_score = composite_index.iloc[-1]
    
    # Calculate percentile and classification
    percentile = (composite_index.rank(pct=True).iloc[-1]) * 100
    conditions_level = classify_credit_conditions(latest_score, percentile)
    
    print(f"📅 Latest Analysis ({latest_date.strftime('%Y-%m-%d')}):")
    print(f"   🎯 Credit Conditions: {conditions_level}")
    print(f"   📊 Composite Score: {latest_score:.2f} (Z-Score)")
    print(f"   📈 Percentile Rank: {percentile:.1f}%")
    
    # Key indicators current values
    print(f"\n📈 Current Credit Indicators:")
    key_indicators = {
        'BAMLH0A0HYM2': ('High Yield Spread', '%', 'lower'),
        'BAMLC0A0CM': ('Investment Grade Spread', '%', 'lower'),
        'DRTSCILM': ('Bank Lending Standards', '% net tightening', 'lower'),
        'BUSLOANS_Growth': ('Business Loan Growth', '% YoY', 'higher'),
    }
    
    for indicator, (name, unit, direction) in key_indicators.items():
        if indicator in credit_data and not credit_data[indicator].empty:
            latest_val = credit_data[indicator].iloc[-1]
            trend_emoji = "📈" if direction == 'higher' else "📉"
            print(f"   {trend_emoji} {name}: {latest_val:.2f}{unit}")
    
    # Historical context
    if len(composite_index) > 24:
        print(f"\n📊 Historical Context:")
        print(f"   📈 Current conditions looser than {percentile:.1f}% of historical values")
        
        # Recent trend
        recent_trend = np.polyfit(range(min(12, len(composite_index))), composite_index.tail(12), 1)[0]
        trend_direction = "LOOSENING" if recent_trend > 0.05 else "TIGHTENING" if recent_trend < -0.05 else "STABLE"
        print(f"   📈 Recent Trend (12M): {trend_direction} (slope: {recent_trend:.3f})")
    
    # Economic impact assessment
    print(f"\n💡 INTERPRETATION:")
    print(f"   {get_credit_interpretation(conditions_level)}")
    
    # Create comprehensive output DataFrame
    output_df = pd.DataFrame(index=composite_index.index)
    
    # Add raw credit indicators
    for name, series in credit_data.items():
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
    output_df['Conditions_Level'] = composite_index.apply(lambda x: classify_credit_conditions(x, composite_index.rank(pct=True).loc[composite_index.index[composite_index == x].tolist()[0]] * 100 if x in composite_index.values else 50))
    
    # Add rolling statistics
    rolling_mean = composite_index.rolling(window=args.window).mean()
    rolling_std = composite_index.rolling(window=args.window).std()
    output_df['Rolling_Mean'] = rolling_mean
    output_df['Rolling_Std'] = rolling_std
    output_df['Z_Score_12M'] = (composite_index - rolling_mean) / rolling_std
    
    # Save results
    output_df.to_csv('credit_conditions_analysis.csv')
    print(f"\n📁 Files saved:")
    print(f"   📊 credit_conditions_analysis.csv (complete data and analysis)")
    
    # Generate visualization
    create_credit_conditions_visualization(credit_data, composite_index, combined_df, args.window)
    print(f"   📈 credit_conditions_analysis.png (comprehensive visualization)")
    
    print(f"\n📊 Analysis window: {args.window} months")
    print(f"📡 Data source: Federal Reserve Economic Data (FRED)")
    data_quality = (1 - output_df.isnull().sum().sum() / (len(output_df) * len(output_df.columns))) * 100
    print(f"✅ Data quality: {data_quality:.1f}% coverage")
    print("=" * 68)

if __name__ == "__main__":
    main()