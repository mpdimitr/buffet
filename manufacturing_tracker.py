#!/usr/bin/env python3
"""
🏭 MANUFACTURING SECTOR TRACKER - FEDERAL RESERVE DATA ANALYSIS
===============================================================

A comprehensive tracker for manufacturing sector health using official Federal Reserve
data from FRED, providing a robust analysis of manufacturing activity.

This tracker uses real economic data instead of survey-based indices:
- Industrial Production: Manufacturing (actual output)
- Capacity Utilization: Manufacturing (efficiency metric)  
- Manufacturers' New Orders: Total Manufacturing (demand indicator)
- Manufacturers' New Orders: Durable Goods (core demand)

Key Components:
- Current Manufacturing Output (Industrial Production)
- Manufacturing Efficiency (Capacity Utilization)
- Forward-Looking Demand (New Orders)
- Manufacturing Health Classification

Data Sources:
- Federal Reserve Economic Data (FRED)
- Official Federal Reserve Board data

Features:
- Real-time manufacturing data analysis
- Multi-component health scoring
- Trend detection and momentum analysis
- Manufacturing strength classification
- Professional multi-panel visualizations
- Comprehensive CSV exports

Author: Economic Analysis Suite
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import argparse
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def configure_display():
    """Configure pandas display options for better output formatting."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 50)

def get_manufacturing_data(start_date, end_date):
    """
    Fetch comprehensive manufacturing data from FRED using Federal Reserve series.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        tuple: (data_dict, successful_series)
    """
    print("📊 Fetching Manufacturing data from FRED...")
    
    # Federal Reserve manufacturing data series
    manufacturing_series = {
        'Industrial_Production': 'IPMAN',      # Industrial Production: Manufacturing (NAICS)
        'Capacity_Utilization': 'MCUMFN',     # Capacity Utilization: Manufacturing (NAICS) 
        'New_Orders_Total': 'AMTMNO',         # Manufacturers' New Orders: Total Manufacturing
        'New_Orders_Durable': 'DGORDER',      # Manufacturers' New Orders: Durable Goods
        'Durable_Production': 'IPDMAN',       # Industrial Production: Durable Manufacturing
        'Nondurable_Production': 'IPNMAN'     # Industrial Production: Nondurable Manufacturing
    }
    
    manufacturing_data = {}
    successful_series = []
    
    for name, series_id in manufacturing_series.items():
        try:
            print(f"✓ Fetching {name} ({series_id})...")
            data = pdr.get_data_fred(series_id, start=start_date, end=end_date)
            if not data.empty:
                manufacturing_data[name] = data.iloc[:, 0]  # Get the series data
                successful_series.append(name)
                print(f"✓ Successfully retrieved {name} ({len(data)} records)")
            else:
                print(f"⚠ No data available for {name}")
        except Exception as e:
            print(f"⚠ Skipping {name}: {str(e)}")
            continue
    
    print(f"✅ Successfully collected {len(successful_series)} manufacturing indicators")
    return manufacturing_data, successful_series

def calculate_manufacturing_health_score(data):
    """
    Calculate a comprehensive manufacturing health score from Federal Reserve data.
    
    Args:
        data (dict): Dictionary of manufacturing data series
    
    Returns:
        pd.Series: Manufacturing health composite score
    """
    print("🔄 Creating manufacturing health composite score...")
    
    # Define weights for different components
    component_weights = {
        'Industrial_Production': 0.35,  # Current production output
        'Capacity_Utilization': 0.25,  # Manufacturing efficiency
        'New_Orders_Total': 0.20,      # Forward-looking demand
        'New_Orders_Durable': 0.10,    # Core manufacturing demand
        'Durable_Production': 0.05,    # Durable goods production
        'Nondurable_Production': 0.05  # Nondurable goods production
    }
    
    # Create normalized scores for each component
    normalized_components = {}
    valid_components = []
    
    for component, weight in component_weights.items():
        if component in data and not data[component].empty:
            series_data = data[component]
            
            if component in ['Industrial_Production', 'Durable_Production', 'Nondurable_Production']:
                # For production indices, use year-over-year growth percentile
                yoy_growth = series_data.pct_change(12) * 100
                # Convert to percentile ranking (0-100)
                normalized_score = yoy_growth.rolling(window=120, min_periods=60).apply(
                    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 
                    if x.max() > x.min() else 50
                ).fillna(50)
                
            elif component == 'Capacity_Utilization':
                # For capacity utilization, use percentile ranking directly
                normalized_score = series_data.rolling(window=120, min_periods=60).apply(
                    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 
                    if x.max() > x.min() else 50
                ).fillna(50)
                
            else:  # New Orders series
                # For new orders, use year-over-year growth percentile
                yoy_growth = series_data.pct_change(12) * 100
                # Convert to percentile ranking
                normalized_score = yoy_growth.rolling(window=120, min_periods=60).apply(
                    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 
                    if x.max() > x.min() else 50
                ).fillna(50)
            
            normalized_components[component] = normalized_score
            valid_components.append(component)
            print(f"✓ Normalized {component} - {len(series_data)} points")
    
    if not normalized_components:
        raise ValueError("No valid manufacturing components available for composite calculation")
    
    # Calculate weighted composite
    composite_data = None
    total_weight = 0
    
    # Find common date range
    common_dates = None
    for component in valid_components:
        if common_dates is None:
            common_dates = normalized_components[component].index
        else:
            common_dates = common_dates.intersection(normalized_components[component].index)
    
    if len(common_dates) == 0:
        raise ValueError("No common dates found across components")
    
    # Calculate weighted composite on common dates
    composite_data = pd.Series(0.0, index=common_dates)
    
    for component in valid_components:
        weight = component_weights[component]
        component_data = normalized_components[component].reindex(common_dates)
        composite_data += component_data * weight
        total_weight += weight
    
    # Normalize by total weight
    if total_weight > 0:
        composite_data = composite_data / total_weight
    
    # Ensure bounds are respected
    composite_data = np.clip(composite_data, 0, 100)
    
    print(f"✅ Created composite score with {len(composite_data)} data points")
    return composite_data.dropna()

def calculate_trend_analysis(data_series):
    """Calculate trend analysis for a data series."""
    if data_series.empty:
        return {}
    
    # Recent trends
    recent_12m = data_series.tail(12)
    recent_6m = data_series.tail(6)
    recent_3m = data_series.tail(3)
    
    # Calculate trends
    trend_12m = (recent_12m.iloc[-1] - recent_12m.iloc[0]) / recent_12m.iloc[0] * 100 if len(recent_12m) >= 2 else 0
    trend_6m = (recent_6m.iloc[-1] - recent_6m.iloc[0]) / recent_6m.iloc[0] * 100 if len(recent_6m) >= 2 else 0
    trend_3m = (recent_3m.iloc[-1] - recent_3m.iloc[0]) / recent_3m.iloc[0] * 100 if len(recent_3m) >= 2 else 0
    
    # Volatility
    volatility = recent_12m.std() if len(recent_12m) > 1 else 0
    
    return {
        'trend_12m': trend_12m,
        'trend_6m': trend_6m,
        'trend_3m': trend_3m,
        'volatility': volatility,
        'latest_value': data_series.iloc[-1]
    }

def classify_manufacturing_strength(ip_value, composite_score):
    """Classify manufacturing sector strength based on indicators."""
    
    # Primarily use composite score for classification
    if composite_score >= 75:
        return "🟢 Very Strong", "Manufacturing sector showing robust expansion"
    elif composite_score >= 60:
        return "🟢 Strong", "Manufacturing sector in solid growth phase"
    elif composite_score >= 50:
        return "🟡 Stable", "Manufacturing sector showing balanced conditions"
    elif composite_score >= 40:
        return "🟠 Weak", "Manufacturing sector showing signs of weakness"
    elif composite_score >= 25:
        return "🔴 Contracting", "Manufacturing sector in contraction"
    else:
        return "🔴 Severe Contraction", "Manufacturing sector in severe decline"

def create_manufacturing_visualizations(data, composite_score, save_path=None):
    """Create comprehensive manufacturing analysis visualizations."""
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 10
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('🏭 Manufacturing Sector Health Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Define colors
    colors = {
        'Industrial_Production': '#1f77b4',
        'Capacity_Utilization': '#ff7f0e', 
        'New_Orders_Total': '#2ca02c',
        'New_Orders_Durable': '#d62728',
        'Durable_Production': '#9467bd',
        'Nondurable_Production': '#8c564b'
    }
    
    # 1. Industrial Production (top left)
    ax1 = axes[0, 0]
    if 'Industrial_Production' in data:
        ip_data = data['Industrial_Production']
        ax1.plot(ip_data.index, ip_data, color=colors['Industrial_Production'], linewidth=2)
        ax1.set_title('Industrial Production: Manufacturing', fontweight='bold')
        ax1.set_ylabel('Index (2017=100)')
        ax1.grid(True, alpha=0.3)
        
        # Add recession shading (approximate recent recessions)
        recession_periods = [
            ('2020-02-01', '2020-04-01'),  # COVID recession
            ('2007-12-01', '2009-06-01')   # Great Recession
        ]
        for start, end in recession_periods:
            if start < ip_data.index[-1].strftime('%Y-%m-%d'):
                ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                           alpha=0.2, color='red', label='Recession' if start == recession_periods[0][0] else "")
    
    # 2. Capacity Utilization (top right)
    ax2 = axes[0, 1]
    if 'Capacity_Utilization' in data:
        cu_data = data['Capacity_Utilization']
        ax2.plot(cu_data.index, cu_data, color=colors['Capacity_Utilization'], linewidth=2)
        ax2.set_title('Manufacturing Capacity Utilization', fontweight='bold')
        ax2.set_ylabel('Percent')
        ax2.grid(True, alpha=0.3)
        
        # Add average line
        avg_cu = cu_data.mean()
        ax2.axhline(y=avg_cu, color='red', linestyle='--', alpha=0.7, 
                   label=f'Average: {avg_cu:.1f}%')
        ax2.legend()
    
    # 3. New Orders - Total (middle left)
    ax3 = axes[1, 0]
    if 'New_Orders_Total' in data:
        no_data = data['New_Orders_Total']
        ax3.plot(no_data.index, no_data, color=colors['New_Orders_Total'], linewidth=2)
        ax3.set_title('Total Manufacturing New Orders', fontweight='bold')
        ax3.set_ylabel('Millions of Dollars')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line for recent period
        recent_data = no_data.tail(24)  # Last 2 years
        if len(recent_data) > 1:
            z = np.polyfit(range(len(recent_data)), recent_data, 1)
            p = np.poly1d(z)
            ax3.plot(recent_data.index, p(range(len(recent_data))), 
                    color='red', linestyle='--', alpha=0.7, label='Trend')
            ax3.legend()
    
    # 4. New Orders - Durable Goods (middle right)
    ax4 = axes[1, 1]
    if 'New_Orders_Durable' in data:
        nod_data = data['New_Orders_Durable']
        ax4.plot(nod_data.index, nod_data, color=colors['New_Orders_Durable'], linewidth=2)
        ax4.set_title('Durable Goods New Orders', fontweight='bold')
        ax4.set_ylabel('Millions of Dollars')
        ax4.grid(True, alpha=0.3)
    
    # 5. Manufacturing Health Composite Score (bottom left)
    ax5 = axes[2, 0]
    if not composite_score.empty:
        ax5.plot(composite_score.index, composite_score, color='purple', linewidth=2.5, label='Health Score')
        ax5.set_title('Manufacturing Health Composite Score', fontweight='bold')
        ax5.set_ylabel('Score (0-100)')
        ax5.set_ylim(0, 100)
        ax5.grid(True, alpha=0.3)
        
        # Add strength zones
        ax5.axhspan(75, 100, alpha=0.1, color='green', label='Very Strong')
        ax5.axhspan(60, 75, alpha=0.1, color='lightgreen', label='Strong')
        ax5.axhspan(40, 60, alpha=0.1, color='yellow', label='Stable')
        ax5.axhspan(25, 40, alpha=0.1, color='orange', label='Weak')
        ax5.axhspan(0, 25, alpha=0.1, color='red', label='Contracting')
        
        # Current score annotation
        current_score = composite_score.iloc[-1]
        ax5.annotate(f'Current: {current_score:.1f}', 
                    xy=(composite_score.index[-1], current_score),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 6. Year-over-Year Growth Rates (bottom right)
    ax6 = axes[2, 1]
    growth_data = {}
    for name, series in data.items():
        if name in ['Industrial_Production', 'New_Orders_Total', 'New_Orders_Durable']:
            yoy_growth = series.pct_change(12) * 100
            if not yoy_growth.empty:
                growth_data[name] = yoy_growth
    
    if growth_data:
        for name, growth in growth_data.items():
            ax6.plot(growth.index, growth, label=name.replace('_', ' '), 
                    color=colors.get(name, 'gray'), linewidth=2)
        
        ax6.set_title('Year-over-Year Growth Rates', fontweight='bold')
        ax6.set_ylabel('Percent Change')
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    # Format x-axes for all subplots
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved to {save_path}")
    
    plt.show()

def export_data_to_csv(data, composite_score, filename="manufacturing_analysis.csv"):
    """Export analysis data to CSV file."""
    
    # Combine all data into a single DataFrame
    export_df = pd.DataFrame()
    
    # Add all individual series
    for name, series in data.items():
        export_df[name] = series
    
    # Add composite score
    if not composite_score.empty:
        export_df['Manufacturing_Health_Score'] = composite_score
    
    # Add derived metrics
    if 'Industrial_Production' in data:
        export_df['IP_YoY_Growth'] = data['Industrial_Production'].pct_change(12) * 100
    
    if 'New_Orders_Total' in data:
        export_df['Orders_YoY_Growth'] = data['New_Orders_Total'].pct_change(12) * 100
    
    # Export to CSV
    export_df.to_csv(filename)
    print(f"📄 Data exported to {filename}")
    
    return export_df

def main():
    """Main execution function with argument parsing."""
    
    # Configure pandas display
    configure_display()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Manufacturing Sector Health Tracker')
    parser.add_argument('--start', type=str, default='2010-01-01',
                       help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                       help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--save-chart', type=str, default=None,
                       help='Path to save chart (e.g., manufacturing_chart.png)')
    parser.add_argument('--export-csv', type=str, default=None,
                       help='Path to export CSV data (e.g., manufacturing_data.csv)')
    parser.add_argument('--min-data-points', type=int, default=60,
                       help='Minimum data points required for analysis')
    
    args = parser.parse_args()
    
    print("🏭 MANUFACTURING SECTOR TRACKER - FEDERAL RESERVE DATA")
    print("=" * 68)
    print("📊 Data Sources: Federal Reserve Economic Data (FRED)")
    print(f"📅 Analysis Period: {args.start} to {args.end}")
    print(f"🎯 Min Data Points: {args.min_data_points}")
    print("=" * 68)
    
    try:
        # Fetch manufacturing data
        manufacturing_data, successful_series = get_manufacturing_data(args.start, args.end)
        
        if not manufacturing_data:
            print("❌ No manufacturing data available from FRED")
            return
        
        # Calculate manufacturing health composite
        manufacturing_composite = calculate_manufacturing_health_score(manufacturing_data)
        
        # Get the main industrial production data for analysis
        ip_data = manufacturing_data.get('Industrial_Production')
        if ip_data is None or ip_data.empty:
            print("❌ No Industrial Production data available for analysis")
            return
        
        # Calculate analysis metrics
        latest_ip = ip_data.iloc[-1] if not ip_data.empty else np.nan
        latest_composite = manufacturing_composite.iloc[-1] if not manufacturing_composite.empty else np.nan
        
        # Trend analysis
        trend_analysis = calculate_trend_analysis(ip_data)
        
        # Classification
        current_classification = classify_manufacturing_strength(latest_ip, latest_composite)
        
        # Calculate percentiles
        ip_percentile = (ip_data <= latest_ip).mean() * 100 if not pd.isna(latest_ip) else np.nan
        
        # Create comprehensive analysis DataFrame for export
        analysis_df = pd.DataFrame()
        
        # Add all main series to analysis DataFrame
        for name, series in manufacturing_data.items():
            if not series.empty:
                analysis_df[name] = series
        
        # Add composite score
        if not manufacturing_composite.empty:
            analysis_df['Manufacturing_Health_Score'] = manufacturing_composite
        
        # Add derived metrics
        if 'Industrial_Production' in manufacturing_data:
            ip_series = manufacturing_data['Industrial_Production']
            analysis_df['IP_12M_MA'] = ip_series.rolling(window=12).mean()
            analysis_df['IP_YoY_Growth'] = ip_series.pct_change(12) * 100
            analysis_df['IP_ZScore'] = (ip_series - ip_series.mean()) / ip_series.std()
        
        # Summary statistics
        summary_stats = {
            'latest_date': ip_data.index[-1] if not ip_data.empty else None,
            'latest_ip': latest_ip,
            'latest_composite': latest_composite,
            'classification': current_classification[0],
            'trend_12m': trend_analysis.get('trend_12m', 0),
            'trend_6m': trend_analysis.get('trend_6m', 0),
            'trend_3m': trend_analysis.get('trend_3m', 0),
            'volatility': trend_analysis.get('volatility', 0),
            'ip_percentile': ip_percentile,
            'data_coverage': len(ip_data) if ip_data is not None else 0
        }
        
        # Print comprehensive analysis
        print("\n📈 MANUFACTURING SECTOR ANALYSIS")
        print("=" * 50)
        print(f"   🗓 Latest Data: {summary_stats['latest_date'].strftime('%Y-%m-%d') if summary_stats['latest_date'] else 'N/A'}")
        print(f"   📊 Industrial Production: {latest_ip:.1f}" if not pd.isna(latest_ip) else "   📊 Industrial Production: N/A")
        print(f"   🎯 Health Score: {latest_composite:.1f}/100" if not pd.isna(latest_composite) else "   🎯 Health Score: N/A")
        print(f"   🏷 Classification: {current_classification[0]}")
        print(f"   📏 IP Percentile: {ip_percentile:.1f}%" if not pd.isna(ip_percentile) else "   📏 IP Percentile: N/A")
        
        print(f"\n📊 COMPONENT STATUS:")
        component_names = {
            'Industrial_Production': 'Industrial Production',
            'Capacity_Utilization': 'Capacity Utilization', 
            'New_Orders_Total': 'Total New Orders',
            'New_Orders_Durable': 'Durable New Orders',
            'Durable_Production': 'Durable Production',
            'Nondurable_Production': 'Nondurable Production'
        }
        
        for name in successful_series:
            if name in manufacturing_data and not manufacturing_data[name].empty:
                latest_val = manufacturing_data[name].iloc[-1]
                print(f"   ✓ {component_names.get(name, name)}: {latest_val:.1f}")
        
        print(f"\n📈 TREND ANALYSIS:")
        print(f"   📅 12-Month Trend: {trend_analysis.get('trend_12m', 0):+.1f}%")
        print(f"   📅 6-Month Trend: {trend_analysis.get('trend_6m', 0):+.1f}%")
        print(f"   📅 3-Month Trend: {trend_analysis.get('trend_3m', 0):+.1f}%")
        print(f"   📊 Volatility: {trend_analysis.get('volatility', 0):.2f}")
        
        print(f"\n🎯 INTERPRETATION:")
        print(f"   {current_classification[1]}")
        
        # Create visualizations
        create_manufacturing_visualizations(manufacturing_data, manufacturing_composite, args.save_chart)
        
        # Export data if requested
        if args.export_csv:
            export_data_to_csv(manufacturing_data, manufacturing_composite, args.export_csv)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Processed {len(successful_series)} data series")
        print(f"📅 Coverage: {len(analysis_df)} data points")
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        print("🔍 Please check your internet connection and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()