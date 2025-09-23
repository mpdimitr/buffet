# Run this in a Python 3.10+ environment (Jupyter recommended)
# Required packages:
# pip install pandas pandas_datareader matplotlib yfinance

import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import datetime
import argparse
import numpy as np
import yfinance as yf

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

# 1) Get Wilshire 5000 market cap and US GDP data
# We'll fetch daily Wilshire from Yahoo Finance, and quarterly GDP from FRED; 
# convert both to quarterly by resampling (end of quarter).

# Parse command-line arguments for start date
parser = argparse.ArgumentParser(description='Buffett Indicator script')
parser.add_argument('--start', type=str, default='1990-01-01', help='Start date in YYYY-MM-DD format (default: 1990-01-01)')
parser.add_argument('--window', type=int, default=60, help='Rolling window in quarters for z-score/percentile (default: 60 ~ 15 years)')
args = parser.parse_args()

try:
    start = datetime.datetime.strptime(args.start, '%Y-%m-%d')
except Exception:
    raise ValueError('Invalid start date format. Use YYYY-MM-DD.')
end = datetime.datetime.today()

# Wilshire 5000 total market capitalization from Yahoo Finance
wilshire_df = yf.download("^W5000", start=start.strftime('%Y-%m-%d'))
print("Wilshire DataFrame columns:", wilshire_df.columns)
wilshire = wilshire_df[("Close", "^W5000")]
gdp = pdr.DataReader('GDP', 'fred', start, end)                 # GDP (Billions of USD), quarterly SAAR

# Align frequencies: convert both to quarterly end (take last available value in each quarter).
wilshire_q = wilshire.resample('QE').last().rename('Wilshire_MktCap_Bil')
gdp_q = gdp.resample('QE').last().rename(columns={'GDP':'GDP_Bil'})  # GDP is already quarterly typically

# Compute Buffett Indicator (market cap / GDP) as percentage or ratio
buffett = (wilshire_q / gdp_q['GDP_Bil']) * 100.0  # percent of GDP

# --- Derived normalization metrics ---
window = args.window

"""Enhancement metrics: rolling stats, z-score, percentile, trend residual, plus
exponentially weighted statistics for smoother bands."""
# Rolling statistics (require minimum window observations)
roll_mean = buffett.rolling(window).mean()
roll_std = buffett.rolling(window).std()
z_score = (buffett - roll_mean) / roll_std

# Exponentially weighted mean & std (adaptive smoothing)
ewm_mean = buffett.ewm(span=window, min_periods=max(5, window//4), adjust=False).mean()
ewm_std = buffett.ewm(span=window, min_periods=max(5, window//4), adjust=False).std()

# Rolling percentile (rank of last value within rolling window)
def _last_percentile(x):
    if x.isna().any():
        # dropna to avoid size changes; ensure we only compute when full window of non-nan exists
        if x.dropna().shape[0] < window:
            return np.nan
    if len(x) < window:
        return np.nan
    return x.rank(pct=True).iloc[-1]

percentile = buffett.rolling(window).apply(_last_percentile, raw=False)

# Simple log-linear trend on available (non-na) data
if buffett.notna().sum() > 10:  # need enough points
    t_index = np.arange(len(buffett))
    mask = buffett.notna()
    coef = np.polyfit(t_index[mask], np.log(buffett[mask]), 1)
    log_trend = np.polyval(coef, t_index)
    trend = np.exp(log_trend)
    trend_series = pd.Series(trend, index=buffett.index, name='Trend')
    trend_residual = buffett / trend_series  # ratio >1 means above fitted trend
else:
    trend_series = pd.Series(index=buffett.index, dtype=float, name='Trend')
    trend_residual = pd.Series(index=buffett.index, dtype=float, name='TrendResidual')

# 2) Prepare data for plotting
df_plot = pd.DataFrame({
    'Buffett_pct_of_GDP': buffett,
    'RollingMean': roll_mean,
    'RollingStd': roll_std,
    'EWMMean': ewm_mean,
    'EWMStd': ewm_std,
    'ZScore': z_score,
    'Percentile': percentile,
    'Trend': trend_series,
    'TrendResidual': trend_residual
})
df_plot_out = df_plot.copy()
df_plot_core = df_plot[['Buffett_pct_of_GDP','RollingMean','Trend']]

# Plot
fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True,
                         gridspec_kw={'height_ratios':[2.4,1,1]})

# Panel 1: Buffett Indicator with rolling mean & trend + bands
ax = axes[0]

# Add recession shading first (so it appears behind the data)
add_recession_shading(ax, data_start=df_plot_core.index[0], data_end=df_plot_core.index[-1], 
                     alpha=0.15, color='red', label_first=True)

ax.plot(df_plot_core.index, df_plot_core['Buffett_pct_of_GDP'], label='Buffett Indicator (%)', linewidth=1.6, color='tab:blue')
ax.plot(df_plot_core.index, df_plot_core['RollingMean'], label=f'Rolling Mean ({window}q)', linewidth=1.1, linestyle='--', color='tab:green')
ax.plot(df_plot_core.index, df_plot_core['Trend'], label='Log-Linear Trend', linewidth=1.1, linestyle=':', color='tab:purple')

# Rolling bands (±1σ shaded) and additional sigma lines
upper_band = (roll_mean + roll_std)
lower_band = (roll_mean - roll_std)
ax.fill_between(df_plot_core.index, lower_band, upper_band, color='gray', alpha=0.15, label='±1σ (Rolling)')

# +1.5σ and +2σ lines
ax.plot(df_plot_core.index, roll_mean + 1.5*roll_std, linestyle='--', linewidth=0.8, color='red', alpha=0.7, label='+1.5σ')
ax.plot(df_plot_core.index, roll_mean + 2*roll_std, linestyle='-', linewidth=0.8, color='red', alpha=0.6, label='+2σ')
ax.plot(df_plot_core.index, roll_mean - 1.5*roll_std, linestyle='--', linewidth=0.8, color='teal', alpha=0.7, label='-1.5σ')
ax.plot(df_plot_core.index, roll_mean - 2*roll_std, linestyle='-', linewidth=0.8, color='teal', alpha=0.6, label='-2σ')

# EWM bands (thin lines)
ax.plot(df_plot_core.index, ewm_mean, linewidth=0.9, linestyle='-.', color='orange', label='EWM Mean')
ax.plot(df_plot_core.index, ewm_mean + ewm_std, linewidth=0.6, linestyle='-.', color='orange', alpha=0.8, label='EWM ±1σ')
ax.plot(df_plot_core.index, ewm_mean - ewm_std, linewidth=0.6, linestyle='-.', color='orange', alpha=0.8)
ax.set_ylabel('% of GDP')
ax.set_title('Buffett Indicator (Market Cap / GDP) with Rolling Mean, Trend and ±1σ Band')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')

# Panel 2: Z-Score with background highlighting
ax2 = axes[1]

# Add recession shading
add_recession_shading(ax2, data_start=df_plot.index[0], data_end=df_plot.index[-1], 
                     alpha=0.15, color='red', label_first=False)

ax2.plot(df_plot.index, df_plot['ZScore'], color='tab:purple', linewidth=1.1)
ax2.axhline(0, color='black', linewidth=0.7)
ax2.axhline(1, color='red', linewidth=0.7, linestyle='--')
ax2.axhline(-1, color='green', linewidth=0.7, linestyle='--')
ax2.axhline(2, color='red', linewidth=0.6, linestyle=':')
ax2.axhline(-2, color='green', linewidth=0.6, linestyle=':')
ax2.set_ylabel('Z-Score')
ax2.set_title('Valuation Z-Score (Rolling)')
ax2.grid(True, alpha=0.3)

# Highlight zones where z-score outside ±1
zs = df_plot['ZScore']
ax2.fill_between(df_plot.index, zs, 1, where=zs>1, color='red', alpha=0.10, interpolate=True)
ax2.fill_between(df_plot.index, zs, -1, where=zs<-1, color='green', alpha=0.10, interpolate=True)

# Panel 3: Trend Residual & Percentile
ax3 = axes[2]

# Add recession shading
add_recession_shading(ax3, data_start=df_plot.index[0], data_end=df_plot.index[-1], 
                     alpha=0.15, color='red', label_first=False)

ax3.plot(df_plot.index, df_plot['TrendResidual'], color='tab:orange', linewidth=1.1, label='Trend Residual (Actual / Trend)')
ax3.axhline(1.0, color='black', linewidth=0.8)
ax3.axhline(1.15, color='red', linewidth=0.8, linestyle='--')
ax3.axhline(0.90, color='green', linewidth=0.8, linestyle='--')
ax3.set_ylabel('Residual Ratio')
ax3.set_title('Trend Residual (Above 1 = Above Trend)')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left')

# Regime annotations when residual crosses thresholds
resid = df_plot['TrendResidual']
cross_up = resid[(resid.shift(1) <= 1.15) & (resid > 1.15)].index
cross_down = resid[(resid.shift(1) >= 0.90) & (resid < 0.90)].index
for dt in cross_up:
    ax3.axvline(dt, color='red', alpha=0.2, linewidth=1)
    ax3.text(dt, 1.16, '↑>1.15', color='red', fontsize=7, rotation=90, va='bottom', ha='center')
for dt in cross_down:
    ax3.axvline(dt, color='green', alpha=0.2, linewidth=1)
    ax3.text(dt, 0.89, '↓<0.90', color='green', fontsize=7, rotation=90, va='top', ha='center')

plt.tight_layout()
plt.savefig('buffett_indicator_enhanced.png', dpi=150)
plt.close(fig)

# Save numeric CSV for your records
df_plot_out.to_csv('buffett_indicator_enhanced.csv', index_label='date')
print("Saved: buffett_indicator_enhanced.png and buffett_indicator_enhanced.csv")

