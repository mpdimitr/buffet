#!/usr/bin/env python3
"""
Yield Curve Tracker - Recession Risk Indicator
==============================================

This script tracks the yield curve and key Treasury spreads to assess recession risk.
The yield curve is one of the most reliable recession predictors - when short-term 
rates exceed long-term rates (inversion), recession typically follows within 12-18 months.

Key Metrics Tracked:
- 10Y-2Y Treasury spread (most watched inversion indicator)
- 10Y-3M Treasury spread (Fed's preferred indicator) 
- 10Y-Fed Funds spread (policy vs. long-term outlook)
- Full yield curve shape and evolution

Features:
- Real Treasury data from FRED (Federal Reserve Economic Data)
- Historical recession risk analysis and classification
- Comprehensive statistical analysis and visualization
- Inversion detection and duration tracking
- Quality validation and error handling

Usage:
    python3 yield_curve_tracker.py
    python3 yield_curve_tracker.py --start 2000-01-01 --window 24
    python3 yield_curve_tracker.py --min-data-points 100

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

def get_treasury_yields(start_date: str = '1990-01-01') -> pd.DataFrame:
    """
    Fetch Treasury yield data from FRED for key maturities.
    Returns DataFrame with yields for different maturities.
    """
    # FRED Treasury yield series codes
    treasury_series = {
        'DGS3MO': '3M Treasury',     # 3-Month Treasury
        'DGS2': '2Y Treasury',       # 2-Year Treasury  
        'DGS5': '5Y Treasury',       # 5-Year Treasury
        'DGS10': '10Y Treasury',     # 10-Year Treasury
        'DGS30': '30Y Treasury',     # 30-Year Treasury
    }
    
    yields_data = {}
    print("📊 Fetching Treasury yield data from FRED...")
    
    for series_code, description in treasury_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                yields_data[series_code] = data.iloc[:, 0]
                print(f"✓ Successfully retrieved {description} ({len(data)} records)")
            else:
                print(f"⚠ No data available for {description}")
        except Exception as e:
            print(f"❌ Failed to get {description}: {str(e)[:50]}...")
    
    if not yields_data:
        print("❌ ERROR: No Treasury yield data could be retrieved")
        return pd.DataFrame()
    
    # Combine all yields into single DataFrame
    yield_df = pd.DataFrame(yields_data)
    yield_df = yield_df.dropna(how='all')  # Remove rows with all NaN
    
    print(f"✅ Successfully combined {len(yield_df.columns)} yield series")
    print(f"📅 Data period: {yield_df.index[0].strftime('%Y-%m-%d')} to {yield_df.index[-1].strftime('%Y-%m-%d')}")
    
    return yield_df

def get_fed_funds_rate(start_date: str = '1990-01-01') -> pd.Series:
    """
    Fetch Federal Funds Rate from FRED.
    """
    try:
        print("📊 Fetching Federal Funds Rate from FRED...")
        fed_funds = pdr.get_data_fred('FEDFUNDS', start=start_date)
        if not fed_funds.empty:
            print(f"✓ Successfully retrieved Fed Funds Rate ({len(fed_funds)} records)")
            return fed_funds.iloc[:, 0]
        else:
            print("⚠ No Fed Funds Rate data available")
            return pd.Series()
    except Exception as e:
        print(f"❌ Failed to get Fed Funds Rate: {str(e)[:50]}...")
        return pd.Series()

def calculate_yield_spreads(yields_df: pd.DataFrame, fed_funds: pd.Series) -> pd.DataFrame:
    """
    Calculate key yield spreads for recession analysis.
    """
    spreads = pd.DataFrame(index=yields_df.index)
    
    # Key spreads for recession prediction
    if 'DGS10' in yields_df.columns and 'DGS2' in yields_df.columns:
        spreads['10Y-2Y'] = yields_df['DGS10'] - yields_df['DGS2']
        print("✓ Calculated 10Y-2Y spread (primary recession indicator)")
    
    if 'DGS10' in yields_df.columns and 'DGS3MO' in yields_df.columns:
        spreads['10Y-3M'] = yields_df['DGS10'] - yields_df['DGS3MO']
        print("✓ Calculated 10Y-3M spread (Fed's preferred indicator)")
    
    # Fed Funds spread (if available)
    if 'DGS10' in yields_df.columns and not fed_funds.empty:
        # Align fed funds to yield data frequency
        fed_funds_aligned = fed_funds.reindex(yields_df.index, method='ffill')
        spreads['10Y-Fed'] = yields_df['DGS10'] - fed_funds_aligned
        print("✓ Calculated 10Y-Fed Funds spread")
    
    if 'DGS5' in yields_df.columns and 'DGS2' in yields_df.columns:
        spreads['5Y-2Y'] = yields_df['DGS5'] - yields_df['DGS2']
        print("✓ Calculated 5Y-2Y spread")
    
    # Remove rows where we have no spread data
    spreads = spreads.dropna(how='all')
    
    print(f"📈 Calculated {len(spreads.columns)} yield spreads")
    return spreads

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def classify_recession_risk(spreads_df: pd.DataFrame, window: int = 24) -> pd.Series:
    """
    Classify recession risk based on yield curve inversion patterns.
    """
    risk_scores = pd.Series(index=spreads_df.index, dtype=float)
    
    for date in spreads_df.index:
        score = 0
        factors = 0
        
        # 10Y-2Y spread (most important)
        if '10Y-2Y' in spreads_df.columns and not pd.isna(spreads_df.loc[date, '10Y-2Y']):
            spread_2y = spreads_df.loc[date, '10Y-2Y']
            if spread_2y < -0.5:  # Deeply inverted
                score += 4
            elif spread_2y < 0:   # Inverted
                score += 3
            elif spread_2y < 0.5:  # Flattening
                score += 2
            elif spread_2y < 1.0:  # Normal but low
                score += 1
            # else: Normal (score += 0)
            factors += 1
        
        # 10Y-3M spread (Fed's preferred)
        if '10Y-3M' in spreads_df.columns and not pd.isna(spreads_df.loc[date, '10Y-3M']):
            spread_3m = spreads_df.loc[date, '10Y-3M']
            if spread_3m < -0.5:   # Deeply inverted
                score += 3
            elif spread_3m < 0:    # Inverted  
                score += 2.5
            elif spread_3m < 0.5:  # Flattening
                score += 1.5
            elif spread_3m < 1.0:  # Normal but low
                score += 0.5
            factors += 1
        
        # 10Y-Fed spread
        if '10Y-Fed' in spreads_df.columns and not pd.isna(spreads_df.loc[date, '10Y-Fed']):
            spread_fed = spreads_df.loc[date, '10Y-Fed']
            if spread_fed < -1.0:  # Very inverted
                score += 2
            elif spread_fed < 0:   # Inverted
                score += 1.5
            elif spread_fed < 1.0:  # Flattening
                score += 1
            factors += 1
        
        # Average the score
        if factors > 0:
            risk_scores.loc[date] = score / factors
        else:
            risk_scores.loc[date] = np.nan
    
    return risk_scores

def classify_risk_level(risk_score: float) -> str:
    """Convert numeric risk score to descriptive level."""
    if pd.isna(risk_score):
        return "Unknown"
    elif risk_score >= 3.5:
        return "Extreme Risk"
    elif risk_score >= 2.5:
        return "High Risk"
    elif risk_score >= 1.5:
        return "Moderate Risk" 
    elif risk_score >= 0.8:
        return "Low Risk"
    else:
        return "Minimal Risk"

def detect_inversions(spreads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and analyze yield curve inversions.
    """
    inversions = pd.DataFrame(index=spreads_df.index)
    
    for col in spreads_df.columns:
        if col in spreads_df.columns:
            inversions[f'{col}_inverted'] = spreads_df[col] < 0
            
            # Calculate inversion duration (consecutive days inverted)
            inverted = spreads_df[col] < 0
            inversions[f'{col}_duration'] = (inverted * (inverted.groupby((~inverted).cumsum()).cumcount() + 1))
    
    return inversions

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_yield_curve_visualization(yields_df: pd.DataFrame, spreads_df: pd.DataFrame, 
                                   risk_scores: pd.Series, window: int = 24) -> None:
    """
    Create comprehensive yield curve visualization.
    """
    print("📈 Generating comprehensive yield curve visualization...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Define color scheme
    colors = {
        'normal': '#2E8B57',      # Sea Green
        'warning': '#FF8C00',     # Dark Orange  
        'danger': '#DC143C',      # Crimson
        'neutral': '#4682B4'      # Steel Blue
    }
    
    # Subplot 1: Current Yield Curve Shape
    ax1 = plt.subplot(2, 3, 1)
    latest_yields = yields_df.dropna().iloc[-1]
    maturities = ['3M', '2Y', '5Y', '10Y', '30Y']
    maturity_values = [0.25, 2, 5, 10, 30]  # Years
    
    yield_values = []
    valid_maturities = []
    valid_maturity_values = []
    
    for i, (series, mat_name) in enumerate(zip(['DGS3MO', 'DGS2', 'DGS5', 'DGS10', 'DGS30'], maturities)):
        if series in latest_yields.index and not pd.isna(latest_yields[series]):
            yield_values.append(latest_yields[series])
            valid_maturities.append(mat_name)
            valid_maturity_values.append(maturity_values[i])
    
    if yield_values:
        ax1.plot(valid_maturity_values, yield_values, 'o-', linewidth=2, markersize=6, color=colors['neutral'])
        ax1.set_xlabel('Maturity (Years)')
        ax1.set_ylabel('Yield (%)')
        ax1.set_title(f'Current Yield Curve\n{latest_yields.name.strftime("%Y-%m-%d")}')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_xticks(valid_maturity_values)
        ax1.set_xticklabels(valid_maturities)
    
    # Subplot 2: 10Y-2Y Spread Over Time
    ax2 = plt.subplot(2, 3, 2)
    if '10Y-2Y' in spreads_df.columns:
        spread_data = spreads_df['10Y-2Y'].dropna()
        ax2.plot(spread_data.index, spread_data.values, linewidth=1.5, color=colors['neutral'])
        ax2.axhline(y=0, color=colors['danger'], linestyle='--', alpha=0.7, label='Inversion Level')
        ax2.fill_between(spread_data.index, spread_data.values, 0, 
                        where=(spread_data.values < 0), color=colors['danger'], alpha=0.3, label='Inverted')
        ax2.fill_between(spread_data.index, spread_data.values, 0,
                        where=(spread_data.values >= 0), color=colors['normal'], alpha=0.3, label='Normal')
    
    ax2.set_title('10Y-2Y Treasury Spread\n(Primary Recession Indicator)')
    ax2.set_ylabel('Spread (percentage points)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Subplot 3: 10Y-3M Spread Over Time  
    ax3 = plt.subplot(2, 3, 3)
    if '10Y-3M' in spreads_df.columns:
        spread_data = spreads_df['10Y-3M'].dropna()
        ax3.plot(spread_data.index, spread_data.values, linewidth=1.5, color=colors['neutral'])
        ax3.axhline(y=0, color=colors['danger'], linestyle='--', alpha=0.7)
        ax3.fill_between(spread_data.index, spread_data.values, 0,
                        where=(spread_data.values < 0), color=colors['danger'], alpha=0.3)
        ax3.fill_between(spread_data.index, spread_data.values, 0,
                        where=(spread_data.values >= 0), color=colors['normal'], alpha=0.3)
    
    ax3.set_title('10Y-3M Treasury Spread\n(Fed Preferred Indicator)')
    ax3.set_ylabel('Spread (percentage points)')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Recession Risk Score
    ax4 = plt.subplot(2, 3, 4)
    risk_data = risk_scores.dropna()
    if not risk_data.empty:
        # Color code risk levels
        risk_colors = []
        for score in risk_data.values:
            if score >= 3.5:
                risk_colors.append(colors['danger'])
            elif score >= 2.5:
                risk_colors.append('#FF4500')  # Orange Red
            elif score >= 1.5:
                risk_colors.append(colors['warning'])
            elif score >= 0.8:
                risk_colors.append('#FFD700')  # Gold
            else:
                risk_colors.append(colors['normal'])
        
        ax4.scatter(risk_data.index, risk_data.values, c=risk_colors, alpha=0.6, s=8)
        ax4.plot(risk_data.index, risk_data.values, linewidth=1, color='gray', alpha=0.5)
        
        # Add risk level lines
        ax4.axhline(y=3.5, color=colors['danger'], linestyle='--', alpha=0.5, label='Extreme Risk')
        ax4.axhline(y=2.5, color='#FF4500', linestyle='--', alpha=0.5, label='High Risk')
        ax4.axhline(y=1.5, color=colors['warning'], linestyle='--', alpha=0.5, label='Moderate Risk')
        ax4.axhline(y=0.8, color='#FFD700', linestyle='--', alpha=0.5, label='Low Risk')
    
    ax4.set_title('Recession Risk Score\n(Composite Yield Curve Analysis)')
    ax4.set_ylabel('Risk Score')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left', fontsize=8)
    
    # Subplot 5: All Spreads Comparison
    ax5 = plt.subplot(2, 3, 5)
    for col in spreads_df.columns:
        if col in spreads_df.columns:
            spread_data = spreads_df[col].dropna()
            if not spread_data.empty:
                ax5.plot(spread_data.index, spread_data.values, linewidth=1.5, label=col, alpha=0.8)
    
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax5.set_title('All Treasury Spreads\nComparison')
    ax5.set_ylabel('Spread (percentage points)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Subplot 6: Current Status Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Get latest data for summary
    latest_date = spreads_df.dropna().index[-1]
    latest_risk = risk_scores.dropna().iloc[-1] if not risk_scores.dropna().empty else np.nan
    risk_level = classify_risk_level(latest_risk)
    
    summary_text = f"""CURRENT YIELD CURVE STATUS
    
📅 Latest Data: {latest_date.strftime('%Y-%m-%d')}

🎯 RECESSION RISK: {risk_level.upper()}
📊 Risk Score: {latest_risk:.2f}/4.0

KEY SPREADS:"""
    
    y_pos = 0.9
    ax6.text(0.05, y_pos, summary_text, transform=ax6.transAxes, fontsize=11, 
             verticalalignment='top', fontweight='bold')
    
    y_pos -= 0.4
    for col in ['10Y-2Y', '10Y-3M', '10Y-Fed']:
        if col in spreads_df.columns and not pd.isna(spreads_df.loc[latest_date, col]):
            spread_val = spreads_df.loc[latest_date, col]
            status = "🔴 INVERTED" if spread_val < 0 else "🟢 NORMAL"
            ax6.text(0.05, y_pos, f"{col}: {spread_val:.2f}% {status}", 
                    transform=ax6.transAxes, fontsize=10)
            y_pos -= 0.08
    
    # Add interpretation
    interpretation = f"""
💡 INTERPRETATION:
{get_risk_interpretation(risk_level)}"""
    
    ax6.text(0.05, y_pos-0.05, interpretation, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', style='italic')
    
    plt.tight_layout()
    plt.savefig('yield_curve_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Yield curve visualization saved as 'yield_curve_analysis.png'")

def get_risk_interpretation(risk_level: str) -> str:
    """Get interpretation text for risk level."""
    interpretations = {
        "Extreme Risk": "Yield curve deeply inverted.\nRecession highly likely within 12-18 months.",
        "High Risk": "Significant yield curve inversion.\nElevated recession probability.",
        "Moderate Risk": "Yield curve flattening or mild inversion.\nIncreased recession watch.",
        "Low Risk": "Yield curve relatively normal.\nLow near-term recession risk.", 
        "Minimal Risk": "Normal yield curve shape.\nMinimal recession signals.",
        "Unknown": "Insufficient data for assessment."
    }
    return interpretations.get(risk_level, "Data quality insufficient for interpretation.")

# ============================================================================
# MAIN ANALYSIS SCRIPT
# ============================================================================

def main():
    """Main analysis function"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Yield Curve Tracker - Recession Risk Indicator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 yield_curve_tracker.py
  python3 yield_curve_tracker.py --start 2000-01-01 --window 36
  python3 yield_curve_tracker.py --min-data-points 100
        """
    )
    parser.add_argument('--start', type=str, default='1990-01-01',
                       help='Start date in YYYY-MM-DD format (default: 1990-01-01)')
    parser.add_argument('--window', type=int, default=24,
                       help='Rolling window in months for analysis (default: 24)')
    parser.add_argument('--min-data-points', type=int, default=50,
                       help='Minimum data points required for analysis (default: 50)')
    args = parser.parse_args()

    try:
        start = datetime.datetime.strptime(args.start, '%Y-%m-%d')
    except Exception:
        start = datetime.datetime(1990, 1, 1)
    end = datetime.datetime.today()

    print("📈 YIELD CURVE TRACKER - RECESSION RISK INDICATOR")
    print("=" * 60)
    print("📊 Data Sources: Federal Reserve Economic Data (FRED)")
    print(f"📅 Analysis Period: {args.start} to {end.strftime('%Y-%m-%d')}")
    print(f"📊 Rolling Window: {args.window} months")
    print(f"🎯 Min Data Points: {args.min_data_points}")
    print("=" * 60)

    # Fetch Treasury yield data
    yields_df = get_treasury_yields(args.start)
    
    if yields_df.empty:
        print("❌ ERROR: Could not retrieve Treasury yield data")
        print("Please check your internet connection and FRED data availability")
        return
    
    # Fetch Fed Funds rate
    fed_funds = get_fed_funds_rate(args.start)
    
    # Calculate yield spreads
    spreads_df = calculate_yield_spreads(yields_df, fed_funds)
    
    if spreads_df.empty:
        print("❌ ERROR: Could not calculate any yield spreads")
        return
    
    # Check minimum data requirement
    if len(spreads_df.dropna()) < args.min_data_points:
        print(f"❌ ERROR: Insufficient data points ({len(spreads_df.dropna())} < {args.min_data_points})")
        print("Try reducing --min-data-points or changing --start date")
        return
    
    print(f"✅ Successfully processed {len(spreads_df)} data points")
    print(f"📅 Data coverage: {spreads_df.index[0].strftime('%Y-%m-%d')} to {spreads_df.index[-1].strftime('%Y-%m-%d')}")
    
    # Calculate recession risk scores
    print("\n🔄 Calculating recession risk scores...")
    risk_scores = classify_recession_risk(spreads_df, args.window)
    
    # Detect inversions
    inversions_df = detect_inversions(spreads_df)
    
    # Generate comprehensive analysis
    print("\n📊 YIELD CURVE ANALYSIS RESULTS")
    print("=" * 50)
    
    # Latest values
    latest_date = spreads_df.dropna().index[-1]
    latest_risk = risk_scores.dropna().iloc[-1] if not risk_scores.dropna().empty else np.nan
    risk_level = classify_risk_level(latest_risk)
    
    print(f"📅 Latest Analysis ({latest_date.strftime('%Y-%m-%d')}):")
    print(f"   🎯 Recession Risk Level: {risk_level}")
    print(f"   📊 Risk Score: {latest_risk:.2f}/4.0")
    
    # Key spreads
    print(f"\n📈 Current Yield Spreads:")
    for col in ['10Y-2Y', '10Y-3M', '10Y-Fed', '5Y-2Y']:
        if col in spreads_df.columns and not pd.isna(spreads_df.loc[latest_date, col]):
            spread_val = spreads_df.loc[latest_date, col]
            status = "🔴 INVERTED" if spread_val < 0 else "🟢 NORMAL"
            print(f"   {col}: {spread_val:.2f}% {status}")
    
    # Historical context
    risk_data = risk_scores.dropna()
    if len(risk_data) > 24:
        percentile = (risk_data.rank(pct=True).iloc[-1]) * 100
        print(f"\n📊 Historical Context:")
        print(f"   📈 Current risk higher than {percentile:.1f}% of historical values")
        
        # Recent trend
        recent_trend = np.polyfit(range(min(12, len(risk_data))), risk_data.tail(12), 1)[0]
        trend_direction = "RISING" if recent_trend > 0.05 else "FALLING" if recent_trend < -0.05 else "STABLE"
        print(f"   📈 Recent Trend (12M): {trend_direction} (slope: {recent_trend:.3f})")
    
    # Inversion analysis
    print(f"\n🔄 Inversion Analysis:")
    for col in ['10Y-2Y', '10Y-3M']:
        if f'{col}_inverted' in inversions_df.columns:
            current_inverted = inversions_df[f'{col}_inverted'].iloc[-1]
            if current_inverted:
                duration = inversions_df[f'{col}_duration'].iloc[-1]
                print(f"   🚨 {col} currently inverted for {duration} days")
            else:
                # Find last inversion
                last_inversion = inversions_df[f'{col}_inverted'].iloc[::-1]
                if last_inversion.any():
                    last_inv_idx = last_inversion.idxmax()
                    days_since = (latest_date - last_inv_idx).days
                    print(f"   ✅ {col} last inverted {days_since} days ago")
                else:
                    print(f"   ✅ {col} no recent inversions detected")
    
    print(f"\n💡 INTERPRETATION:")
    print(f"   {get_risk_interpretation(risk_level)}")
    
    # Create comprehensive output DataFrame
    output_df = pd.DataFrame(index=spreads_df.index)
    
    # Add spreads
    for col in spreads_df.columns:
        output_df[f'{col}_spread'] = spreads_df[col]
    
    # Add risk scores and classification
    output_df['Risk_Score'] = risk_scores
    output_df['Risk_Level'] = risk_scores.apply(classify_risk_level)
    
    # Add inversion flags
    for col in inversions_df.columns:
        output_df[col] = inversions_df[col]
    
    # Add rolling statistics
    if '10Y-2Y' in spreads_df.columns:
        spread_data = spreads_df['10Y-2Y'].dropna()
        rolling_mean = spread_data.rolling(window=args.window*30).mean()  # Approximate monthly
        rolling_std = spread_data.rolling(window=args.window*30).std()
        output_df['10Y-2Y_rolling_mean'] = rolling_mean
        output_df['10Y-2Y_rolling_std'] = rolling_std
        output_df['10Y-2Y_zscore'] = (spread_data - rolling_mean) / rolling_std
    
    # Save results
    output_df.to_csv('yield_curve_analysis.csv')
    print(f"\n📁 Files saved:")
    print(f"   📊 yield_curve_analysis.csv (complete data and analysis)")
    
    # Generate visualization
    create_yield_curve_visualization(yields_df, spreads_df, risk_scores, args.window)
    print(f"   📈 yield_curve_analysis.png (comprehensive visualization)")
    
    print(f"\n📊 Analysis window: {args.window} months")
    print(f"📡 Data source: Federal Reserve Economic Data (FRED)")
    data_quality = (1 - output_df.isnull().sum().sum() / (len(output_df) * len(output_df.columns))) * 100
    print(f"✅ Data quality: {data_quality:.1f}% coverage")
    print("=" * 60)

if __name__ == "__main__":
    main()