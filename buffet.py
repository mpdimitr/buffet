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

# Rolling statistics (require minimum window observations)
roll_mean = buffett.rolling(window).mean()
roll_std = buffett.rolling(window).std()
z_score = (buffett - roll_mean) / roll_std

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
    'ZScore': z_score,
    'Percentile': percentile,
    'Trend': trend_series,
    'TrendResidual': trend_residual
})
df_plot_out = df_plot.copy()
df_plot_core = df_plot[['Buffett_pct_of_GDP','RollingMean','Trend']]

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True,
                         gridspec_kw={'height_ratios':[2.2,1,1]})

# Panel 1: Buffett Indicator with rolling mean & trend + bands
ax = axes[0]
ax.plot(df_plot_core.index, df_plot_core['Buffett_pct_of_GDP'], label='Buffett Indicator (%)', linewidth=1.8)
ax.plot(df_plot_core.index, df_plot_core['RollingMean'], label=f'Rolling Mean ({window}q)', linewidth=1.2, linestyle='--')
ax.plot(df_plot_core.index, df_plot_core['Trend'], label='Log-Linear Trend', linewidth=1.2, linestyle=':')
upper_band = (roll_mean + roll_std)
lower_band = (roll_mean - roll_std)
ax.fill_between(df_plot_core.index, lower_band, upper_band, color='gray', alpha=0.15, label='±1σ band')
ax.set_ylabel('% of GDP')
ax.set_title('Buffett Indicator (Market Cap / GDP) with Rolling Mean, Trend and ±1σ Band')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')

# Panel 2: Z-Score
ax2 = axes[1]
ax2.plot(df_plot.index, df_plot['ZScore'], color='tab:purple', linewidth=1.2)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.axhline(1, color='red', linewidth=0.8, linestyle='--')
ax2.axhline(-1, color='green', linewidth=0.8, linestyle='--')
ax2.set_ylabel('Z-Score')
ax2.set_title('Valuation Z-Score (vs Rolling Window)')
ax2.grid(True, alpha=0.3)

# Panel 3: Trend Residual & Percentile
ax3 = axes[2]
ax3.plot(df_plot.index, df_plot['TrendResidual'], color='tab:orange', linewidth=1.2, label='Trend Residual (Actual / Trend)')
ax3.axhline(1.0, color='black', linewidth=0.8)
ax3.axhline(1.15, color='red', linewidth=0.8, linestyle='--')
ax3.axhline(0.90, color='green', linewidth=0.8, linestyle='--')
ax3.set_ylabel('Residual Ratio')
ax3.set_title('Trend Residual (Above 1 = Above Trend)')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left')

plt.tight_layout()
plt.savefig('buffett_indicator_enhanced.png', dpi=150)
plt.close(fig)

# Save numeric CSV for your records
df_plot_out.to_csv('buffett_indicator_enhanced.csv', index_label='date')
print("Saved: buffett_indicator_enhanced.png and buffett_indicator_enhanced.csv")

