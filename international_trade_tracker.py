#!/usr/bin/env python3
"""
International Trade Tracker - Global Economic Integration Monitor
================================================================

This script tracks key international trade and currency indicators to assess global
economic health and trade dynamics. International trade flows are critical indicators
of global economic activity, competitiveness, and cross-border capital allocation.

Key Metrics Tracked:
- Trade Balance: Net exports indicating trade competitiveness
- US Dollar Index (DXY): Currency strength impact on trade
- Commodity Price Index: Input costs and global demand
- Foreign Direct Investment: Cross-border capital flows
- Import/Export Growth: Trade volume dynamics
- Terms of Trade: Export vs import price relationship

Features:
- Real trade data from Bureau of Economic Analysis and Fed sources
- Composite international trade health scoring and classification
- Comprehensive statistical analysis and visualization
- Trade strength classification (Very Strong → Very Weak)
- Quality validation and error handling

Usage:
    python3 international_trade_tracker.py
    python3 international_trade_tracker.py --start 2000-01-01 --window 12
    python3 international_trade_tracker.py --min-data-points 50

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

def get_trade_balance(start_date: str = '1990-01-01') -> pd.Series:
    """
    Fetch US trade balance data from FRED.
    Positive values indicate trade surplus, negative indicate deficit.
    """
    try:
        print("   📊 Fetching Trade Balance data...")
        # US Trade Balance of Goods and Services
        trade_balance = pdr.get_data_fred('BOPGSTB', start=start_date)
        if not trade_balance.empty:
            series = trade_balance.iloc[:, 0].dropna()
            print(f"      ✅ Trade Balance: {len(series)} observations")
            return series
        else:
            print("      ⚠️ No Trade Balance data available")
            return pd.Series(dtype=float)
    except Exception as e:
        print(f"      ❌ Error fetching Trade Balance: {e}")
        return pd.Series(dtype=float)

def get_dollar_index(start_date: str = '1990-01-01') -> pd.Series:
    """
    Fetch US Dollar Index (DXY) from FRED.
    Higher values indicate stronger dollar (may hurt exports).
    """
    try:
        print("   💵 Fetching US Dollar Index data...")
        # Trade Weighted US Dollar Index: Broad
        dxy = pdr.get_data_fred('DTWEXBGS', start=start_date)
        if not dxy.empty:
            series = dxy.iloc[:, 0].dropna()
            print(f"      ✅ Dollar Index: {len(series)} observations")
            return series
        else:
            print("      ⚠️ No Dollar Index data available")
            return pd.Series(dtype=float)
    except Exception as e:
        print(f"      ❌ Error fetching Dollar Index: {e}")
        return pd.Series(dtype=float)

def get_commodity_prices(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch commodity price indices from FRED.
    Higher prices can indicate global demand strength.
    """
    commodity_data = {}
    
    commodity_series = {
        'PPIACO': 'Producer Price Index - All Commodities',
        'DCOILWTICO': 'WTI Crude Oil Price',
        'GOLDAMGBD228NLBM': 'Gold Price',
        'PCOPPUSDM': 'Copper Price'
    }
    
    print("   🏭 Fetching Commodity Price data...")
    
    for series_code, description in commodity_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                series = data.iloc[:, 0].dropna()
                commodity_data[series_code] = series
                print(f"      ✅ {description}: {len(series)} observations")
            else:
                print(f"      ⚠️ No data for {description}")
        except Exception as e:
            print(f"      ❌ Error fetching {description}: {e}")
    
    return commodity_data

def get_foreign_investment_flows(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch foreign investment flow indicators from FRED.
    Capital flows indicate international confidence and liquidity.
    """
    investment_data = {}
    
    investment_series = {
        'BOGZ1FA263164003Q': 'Foreign Direct Investment in US',
        'BOGZ1FA794190005Q': 'US Direct Investment Abroad',
        'GFDEBTN': 'Federal Debt Total Public Debt',
        'FDHBFRBN': 'Foreign Holdings of US Treasury Securities'
    }
    
    print("   💰 Fetching Foreign Investment data...")
    
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
            print(f"      ❌ Error fetching {description}: {e}")
    
    return investment_data

def get_trade_volume_indicators(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Fetch trade volume and flow indicators from FRED.
    Trade volumes indicate global economic activity levels.
    """
    volume_data = {}
    
    volume_series = {
        'EXPGS': 'Exports of Goods and Services',
        'IMPGS': 'Imports of Goods and Services',
        'NETEXP': 'Net Exports of Goods and Services',
        'IEABCN': 'Import/Export Price Index'
    }
    
    print("   📦 Fetching Trade Volume data...")
    
    for series_code, description in volume_series.items():
        try:
            data = pdr.get_data_fred(series_code, start=start_date)
            if not data.empty:
                series = data.iloc[:, 0].dropna()
                volume_data[series_code] = series
                print(f"      ✅ {description}: {len(series)} observations")
            else:
                print(f"      ⚠️ No data for {description}")
        except Exception as e:
            print(f"      ❌ Error fetching {description}: {e}")
    
    return volume_data

def collect_international_trade_data(start_date: str = '1990-01-01') -> Dict[str, pd.Series]:
    """
    Collect all international trade indicators and return as dictionary.
    """
    print("🌍 Collecting comprehensive international trade data...")
    
    data_components = {}
    
    # Get trade balance (positive indicator when not too extreme)
    trade_balance = get_trade_balance(start_date)
    if not trade_balance.empty:
        data_components['Trade_Balance'] = trade_balance
    
    # Get dollar index (inverse indicator for exports - lower better for trade)
    dollar_index = get_dollar_index(start_date)
    if not dollar_index.empty:
        data_components['Dollar_Index'] = dollar_index
    
    # Get commodity prices
    commodity_data = get_commodity_prices(start_date)
    for name, series in commodity_data.items():
        if not series.empty:
            data_components[f'Commodity_{name}'] = series
    
    # Get foreign investment flows
    investment_data = get_foreign_investment_flows(start_date)
    for name, series in investment_data.items():
        if not series.empty:
            data_components[f'Investment_{name}'] = series
    
    # Get trade volume indicators
    volume_data = get_trade_volume_indicators(start_date)
    for name, series in volume_data.items():
        if not series.empty:
            data_components[f'Volume_{name}'] = series
    
    print(f"✅ Successfully collected {len(data_components)} international trade indicators")
    
    return data_components

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def normalize_trade_indicators(data_dict: Dict[str, pd.Series], window: int = 12) -> Dict[str, pd.Series]:
    """
    Normalize all trade indicators to comparable scales.
    Uses rolling z-scores to handle non-stationary data.
    """
    normalized_data = {}
    
    for name, series in data_dict.items():
        if len(series.dropna()) < window:
            print(f"⚠️ Skipping {name}: insufficient data ({len(series.dropna())} < {window})")
            continue
            
        # Calculate rolling statistics
        rolling_mean = series.rolling(window=window, min_periods=window//2).mean()
        rolling_std = series.rolling(window=window, min_periods=window//2).std()
        
        # Create rolling z-score
        z_score = (series - rolling_mean) / rolling_std
        
        # Apply indicator-specific direction adjustments
        if any(keyword in name.lower() for keyword in ['dollar_index']):
            # Invert dollar index (lower is better for trade competitiveness)
            z_score = -z_score
        elif 'trade_balance' in name.lower():
            # Moderate trade balance is good (neither extreme deficit nor surplus)
            z_score = -np.abs(z_score)
        
        normalized_data[name] = z_score.dropna()
    
    return normalized_data

def create_international_trade_composite(data_dict: Dict[str, pd.Series], window: int = 12) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Create composite international trade health score.
    """
    normalized_data = normalize_trade_indicators(data_dict, window)
    
    if not normalized_data:
        print("❌ No normalized data available for composite creation")
        return pd.Series(dtype=float), pd.DataFrame()
    
    # Create combined DataFrame
    combined_df = pd.DataFrame(normalized_data)
    
    # Define weights for different categories
    weights = {}
    
    # Trade balance and volumes (40% total weight)
    trade_volume_indicators = [col for col in combined_df.columns if any(x in col.lower() for x in ['trade_balance', 'volume_', 'netexp'])]
    for indicator in trade_volume_indicators:
        weights[indicator] = 0.4 / max(len(trade_volume_indicators), 1)
    
    # Currency strength (25% weight)
    currency_indicators = [col for col in combined_df.columns if 'dollar_index' in col.lower()]
    for indicator in currency_indicators:
        weights[indicator] = 0.25 / max(len(currency_indicators), 1)
    
    # Commodity prices (20% weight)
    commodity_indicators = [col for col in combined_df.columns if 'commodity_' in col.lower()]
    for indicator in commodity_indicators:
        weights[indicator] = 0.20 / max(len(commodity_indicators), 1)
    
    # Investment flows (15% weight)
    investment_indicators = [col for col in combined_df.columns if 'investment_' in col.lower()]
    for indicator in investment_indicators:
        weights[indicator] = 0.15 / max(len(investment_indicators), 1)
    
    # Calculate weighted composite
    composite_index = pd.Series(0.0, index=combined_df.index, name='International_Trade_Composite')
    
    for col in combined_df.columns:
        if col in weights:
            weight = weights[col]
            component_data = combined_df[col].fillna(0)
            composite_index += component_data * weight
            print(f"   📊 {col}: weight = {weight:.3f}")
    
    print(f"✅ Created composite index with {len(composite_index.dropna())} observations")
    
    return composite_index, combined_df

def classify_trade_strength(composite_score: float, percentile: float) -> str:
    """
    Classify international trade strength based on composite score and percentile.
    """
    if percentile >= 85 and composite_score > 1.0:
        return "Very Strong"
    elif percentile >= 70 and composite_score > 0.5:
        return "Strong"
    elif percentile >= 55 and composite_score > 0:
        return "Above Average"
    elif percentile >= 45 and composite_score > -0.5:
        return "Average"
    elif percentile >= 30 and composite_score > -1.0:
        return "Below Average"
    elif percentile >= 15:
        return "Weak"
    else:
        return "Very Weak"

def get_trade_interpretation(strength_level: str) -> str:
    """Get interpretation text for trade strength level."""
    interpretations = {
        "Very Strong": "Exceptional trade conditions with strong exports, competitive currency, and robust global demand",
        "Strong": "Favorable trade environment with healthy export growth and supportive global conditions",
        "Above Average": "Generally positive trade conditions with most indicators above historical norms",
        "Average": "Balanced trade conditions with mixed signals across different indicators",
        "Below Average": "Somewhat challenging trade environment with headwinds in key areas",
        "Weak": "Difficult trade conditions with multiple indicators showing stress",
        "Very Weak": "Severely constrained trade environment with broad-based deterioration"
    }
    return interpretations.get(strength_level, "Classification unclear")

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_trade_visualization(data_dict: Dict[str, pd.Series], composite_index: pd.Series, 
                             combined_df: pd.DataFrame, window: int = 12) -> None:
    """
    Create comprehensive international trade visualization.
    """
    fig = plt.figure(figsize=(16, 20))
    
    # Color scheme
    colors = {
        'composite': '#2E86C1',
        'positive': '#28B463',
        'negative': '#E74C3C',
        'neutral': '#F39C12',
        'background': '#F8F9FA'
    }
    
    # Panel 1: Trade Balance and Currency
    ax1 = plt.subplot(6, 1, 1)
    
    if 'Trade_Balance' in data_dict:
        trade_balance = data_dict['Trade_Balance']
        ax1.plot(trade_balance.index, trade_balance, color=colors['composite'], linewidth=2, label='Trade Balance ($B)')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.fill_between(trade_balance.index, trade_balance, 0, 
                        where=(trade_balance >= 0), color=colors['positive'], alpha=0.3, label='Surplus')
        ax1.fill_between(trade_balance.index, trade_balance, 0, 
                        where=(trade_balance < 0), color=colors['negative'], alpha=0.3, label='Deficit')
    
    # Secondary y-axis for dollar index
    if 'Dollar_Index' in data_dict:
        ax1_twin = ax1.twinx()
        dollar_idx = data_dict['Dollar_Index']
        ax1_twin.plot(dollar_idx.index, dollar_idx, color=colors['neutral'], linewidth=2, 
                     linestyle='--', label='Dollar Index', alpha=0.7)
        ax1_twin.set_ylabel('Dollar Index', color=colors['neutral'])
        ax1_twin.legend(loc='upper right')
    
    ax1.set_title('Trade Balance & Currency Strength', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Trade Balance ($B)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Export/Import Dynamics
    ax2 = plt.subplot(6, 1, 2)
    
    export_cols = [col for col in data_dict.keys() if 'expgs' in col.lower() or 'export' in col.lower()]
    import_cols = [col for col in data_dict.keys() if 'impgs' in col.lower() or 'import' in col.lower()]
    
    for col in export_cols[:2]:  # Limit to avoid crowding
        if col in data_dict:
            series = data_dict[col]
            ax2.plot(series.index, series, linewidth=2, label=f'Exports', color=colors['positive'])
    
    for col in import_cols[:2]:
        if col in data_dict:
            series = data_dict[col]
            ax2.plot(series.index, series, linewidth=2, label=f'Imports', color=colors['negative'])
    
    ax2.set_title('Trade Volume Dynamics', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Volume ($B)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Commodity Prices
    ax3 = plt.subplot(6, 1, 3)
    
    commodity_cols = [col for col in data_dict.keys() if 'commodity_' in col.lower()]
    colors_commodities = ['#E74C3C', '#F39C12', '#28B463', '#8E44AD']
    
    for i, col in enumerate(commodity_cols[:4]):  # Limit to 4 commodities
        if col in data_dict:
            series = data_dict[col]
            # Normalize to show relative changes
            normalized = (series / series.iloc[0]) * 100 if not series.empty else series
            ax3.plot(normalized.index, normalized, linewidth=2, 
                    label=col.replace('Commodity_', '').replace('_', ' '), 
                    color=colors_commodities[i % len(colors_commodities)])
    
    ax3.set_title('Commodity Price Trends (Indexed to First Value)', fontsize=14, fontweight='bold', pad=20)
    ax3.set_ylabel('Index (First Value = 100)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Composite Trade Index
    ax4 = plt.subplot(6, 1, 4)
    
    if not composite_index.empty:
        ax4.plot(composite_index.index, composite_index, color=colors['composite'], 
                linewidth=3, label='Composite Trade Index')
        
        # Add rolling statistics
        if len(composite_index) >= window:
            rolling_mean = composite_index.rolling(window).mean()
            rolling_std = composite_index.rolling(window).std()
            
            ax4.plot(rolling_mean.index, rolling_mean, color='black', 
                    linestyle='--', alpha=0.7, label=f'{window}M Rolling Mean')
            
            # Confidence bands
            upper_band = rolling_mean + rolling_std
            lower_band = rolling_mean - rolling_std
            ax4.fill_between(rolling_mean.index, upper_band, lower_band, 
                           alpha=0.2, color=colors['composite'], label='±1σ Band')
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.axhline(y=1, color=colors['positive'], linestyle=':', alpha=0.7, label='Strong (+1σ)')
        ax4.axhline(y=-1, color=colors['negative'], linestyle=':', alpha=0.7, label='Weak (-1σ)')
    
    ax4.set_title('Composite International Trade Health Index', fontsize=14, fontweight='bold', pad=20)
    ax4.set_ylabel('Composite Score (Z-Score)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Trade Strength Classification
    ax5 = plt.subplot(6, 1, 5)
    
    if not composite_index.empty and len(composite_index) >= window:
        # Calculate classification over time
        percentiles = composite_index.rolling(window*4).rank(pct=True) * 100
        classifications = []
        
        for i, (score, pct) in enumerate(zip(composite_index, percentiles)):
            if pd.notna(score) and pd.notna(pct):
                classifications.append(classify_trade_strength(score, pct))
            else:
                classifications.append('Unknown')
        
        # Create classification time series
        class_series = pd.Series(classifications, index=composite_index.index)
        
        # Map classifications to colors
        class_colors = {
            'Very Strong': colors['positive'],
            'Strong': '#58D68D',
            'Above Average': '#85C1E9',
            'Average': '#F7DC6F',
            'Below Average': '#F8C471',
            'Weak': '#EC7063',
            'Very Weak': colors['negative'],
            'Unknown': '#BDC3C7'
        }
        
        # Plot classification periods
        prev_class = None
        start_idx = 0
        
        for i, (date, classification) in enumerate(class_series.items()):
            if classification != prev_class or i == len(class_series) - 1:
                if prev_class is not None:
                    end_idx = i if i < len(class_series) - 1 else i + 1
                    ax5.axvspan(class_series.index[start_idx], class_series.index[min(end_idx, len(class_series)-1)], 
                              color=class_colors.get(prev_class, '#BDC3C7'), alpha=0.6, 
                              label=prev_class if prev_class not in ax5.get_legend_handles_labels()[1] else "")
                prev_class = classification
                start_idx = i
    
    ax5.set_title('Trade Strength Classification Over Time', fontsize=14, fontweight='bold', pad=20)
    ax5.set_ylabel('Strength Level')
    ax5.set_ylim(-0.5, 0.5)
    ax5.set_yticks([])
    if ax5.get_legend_handles_labels()[0]:  # Only show legend if there are items
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Recent Trend Analysis
    ax6 = plt.subplot(6, 1, 6)
    
    if not composite_index.empty and len(composite_index) >= 24:
        recent_data = composite_index.tail(24)  # Last 24 months
        
        # Calculate trend
        x_vals = np.arange(len(recent_data))
        coeffs = np.polyfit(x_vals, recent_data.values, 1)
        trend_line = np.polyval(coeffs, x_vals)
        
        ax6.plot(recent_data.index, recent_data, color=colors['composite'], 
                linewidth=3, marker='o', markersize=4, label='Recent Trade Index')
        ax6.plot(recent_data.index, trend_line, color=colors['negative'], 
                linestyle='--', linewidth=2, label=f'Trend (slope: {coeffs[0]:.3f})')
        
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Highlight latest point
        latest_point = recent_data.iloc[-1]
        ax6.scatter(recent_data.index[-1], latest_point, color=colors['negative'], 
                   s=100, zorder=5, label=f'Latest: {latest_point:.2f}')
    
    ax6.set_title('Recent Trade Conditions Trend (Last 24 Months)', fontsize=14, fontweight='bold', pad=20)
    ax6.set_ylabel('Composite Score')
    ax6.set_xlabel('Date')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Format all date axes
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
        
        # Rotate labels for better readability
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
    
    plt.tight_layout()
    plt.savefig('international_trade_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

# ============================================================================
# MAIN ANALYSIS SCRIPT
# ============================================================================

def main():
    """Main analysis function"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='International Trade Tracker - Global Economic Integration Monitor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 international_trade_tracker.py
  python3 international_trade_tracker.py --start 2000-01-01 --window 12
  python3 international_trade_tracker.py --min-data-points 50
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

    print("🌍 INTERNATIONAL TRADE TRACKER - GLOBAL ECONOMIC INTEGRATION")
    print("=" * 70)
    print("📊 Data Sources: Federal Reserve, BEA, and Market Data via FRED")
    print(f"📅 Analysis Period: {args.start} to {end.strftime('%Y-%m-%d')}")
    print(f"📊 Rolling Window: {args.window} months")
    print(f"🎯 Min Data Points: {args.min_data_points}")
    print("=" * 70)

    # Collect international trade data
    trade_data = collect_international_trade_data(args.start)
    
    if not trade_data:
        print("❌ ERROR: Could not retrieve any international trade data")
        print("Please check your internet connection and FRED data availability")
        return
    
    # Check minimum data requirement
    max_length = max(len(series.dropna()) for series in trade_data.values())
    if max_length < args.min_data_points:
        print(f"❌ ERROR: Insufficient data points ({max_length} < {args.min_data_points})")
        print("Try reducing --min-data-points or changing --start date")
        return
    
    print(f"✅ Successfully processed international trade data")
    
    # Create composite international trade index
    composite_index, combined_df = create_international_trade_composite(trade_data, args.window)
    
    if composite_index.empty:
        print("❌ ERROR: Could not create composite international trade index")
        return
    
    print(f"📅 Data coverage: {composite_index.index[0].strftime('%Y-%m-%d')} to {composite_index.index[-1].strftime('%Y-%m-%d')}")
    
    # Generate comprehensive analysis
    print("\n🌍 INTERNATIONAL TRADE ANALYSIS RESULTS")
    print("=" * 60)
    
    # Latest values
    latest_date = composite_index.index[-1]
    latest_score = composite_index.iloc[-1]
    
    # Calculate percentile and classification
    percentile = (composite_index.rank(pct=True).iloc[-1]) * 100
    strength_level = classify_trade_strength(latest_score, percentile)
    
    print(f"📅 Latest Analysis ({latest_date.strftime('%Y-%m-%d')}):")
    print(f"   🎯 Trade Strength: {strength_level}")
    print(f"   📊 Composite Score: {latest_score:.2f} (Z-Score)")
    print(f"   📈 Percentile Rank: {percentile:.1f}%")
    
    # Key indicators current values
    print(f"\n📈 Current Trade Indicators:")
    key_indicators = {
        'Trade_Balance': ('Trade Balance', '$B', 'balanced'),
        'Dollar_Index': ('Dollar Index', 'Index', 'moderate'),
        'Volume_EXPGS': ('Exports', '$B', 'higher'),
        'Volume_IMPGS': ('Imports', '$B', 'balanced')
    }
    
    for indicator, (name, unit, direction) in key_indicators.items():
        if indicator in trade_data and not trade_data[indicator].empty:
            latest_val = trade_data[indicator].iloc[-1]
            if direction == 'higher':
                trend_emoji = "📈"
            elif direction == 'lower':
                trend_emoji = "📉"
            else:
                trend_emoji = "⚖️"
            print(f"   {trend_emoji} {name}: {latest_val:.1f}{unit}")
    
    # Historical context
    if len(composite_index) > 24:
        print(f"\n📊 Historical Context:")
        print(f"   📈 Current conditions stronger than {percentile:.1f}% of historical values")
        
        # Recent trend
        recent_trend = np.polyfit(range(min(12, len(composite_index))), composite_index.tail(12), 1)[0]
        trend_direction = "STRENGTHENING" if recent_trend > 0.05 else "WEAKENING" if recent_trend < -0.05 else "STABLE"
        print(f"   📈 Recent Trend (12M): {trend_direction} (slope: {recent_trend:.3f})")
    
    # Economic impact assessment
    print(f"\n💡 INTERPRETATION:")
    print(f"   {get_trade_interpretation(strength_level)}")
    
    # Create comprehensive output DataFrame
    output_df = pd.DataFrame(index=composite_index.index)
    
    # Add raw trade indicators
    for name, series in trade_data.items():
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
    output_df['Strength_Level'] = composite_index.apply(lambda x: classify_trade_strength(x, composite_index.rank(pct=True).loc[composite_index.index[composite_index == x].tolist()[0]] * 100 if x in composite_index.values else 50))
    
    # Add rolling statistics
    rolling_mean = composite_index.rolling(window=args.window).mean()
    rolling_std = composite_index.rolling(window=args.window).std()
    output_df['Rolling_Mean'] = rolling_mean
    output_df['Rolling_Std'] = rolling_std
    output_df['Z_Score_12M'] = (composite_index - rolling_mean) / rolling_std
    
    # Save results
    output_df.to_csv('international_trade_analysis.csv')
    print(f"\n📁 Files saved:")
    print(f"   📈 international_trade_analysis.png (comprehensive visualization)")
    print(f"   📊 international_trade_analysis.csv (all data and metrics)")
    
    # Create visualization
    create_trade_visualization(trade_data, composite_index, combined_df, args.window)
    
    # Final summary
    print(f"\n📊 Analysis Summary:")
    print(f"   📊 Analysis window: {args.window} months")
    print(f"   📡 Data source: Federal Reserve Economic Data (FRED)")
    data_quality = (1 - output_df.isnull().sum().sum() / (len(output_df) * len(output_df.columns))) * 100
    print(f"   ✅ Data quality: {data_quality:.1f}% coverage")
    print("=" * 70)

if __name__ == "__main__":
    main()