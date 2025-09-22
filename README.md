# Buffett Indicator Analysis

This project computes and visualizes the **Buffett Indicator** (Total US Equity Market Capitalization / Nominal GDP) and derives several *contextual valuation metrics* to help interpret secular drift and cyclical deviations.

The enhanced output file(s):
- `buffett_indicator_enhanced.png` – Multi‑panel chart.
- `buffett_indicator_enhanced.csv` – Time series with all derived metrics.

Legacy single‑metric outputs (still produced if you ran earlier versions):
- `buffett_indicator.png`, `buffett_indicator.csv`

---
## 1. Core Series
### Buffett_pct_of_GDP
Definition: `(Wilshire 5000 Index Level) / (Nominal US GDP in billions) * 100`.

Purpose: High‑level gauge of aggregate US equity market valuation relative to the size of the domestic economy.

Caveats:
- Numerator includes revenues earned abroad; denominator (GDP) is domestic only → structural upward drift over decades.
- Intangible asset intensity, lower discount rates, globalization, and capital‑light business models all bias the ratio higher vs mid‑20th century history.
- Not a timing tool; extremes can persist for years.

---
## 2. Rolling Context Metrics
All rolling metrics use a *quarterly* frequency (resampled end‑of‑quarter) and a configurable window length in quarters (`--window`, default 60 ≈ 15 years).

### RollingMean
Local baseline level of the Buffett Indicator over the chosen window. Adapts to slow structural shifts (e.g., falling interest rates).

### RollingStd (not plotted directly but used)
Volatility of the Indicator over the same window. Periods of low dispersion shrink the band; volatile regimes widen it.

### ±1σ Band (Shaded Region)
Computed as `RollingMean ± RollingStd`.

Interpretation:
- Inside band: Within recent "normal" range.
- Sustained above +1σ: Stretched relative to its own contemporaneous regime (watch for confirmation from other metrics).
- Sustained below −1σ: Relatively depressed vs recent regime (potential long‑term opportunity context).

Bandwidth Choice:
- ±1σ = sensitive, more signals.
- You can extend logic to ±1.5σ / ±2σ for "stretched" / "extreme" zones.

### ZScore
`(Buffett - RollingMean) / RollingStd`.

Usage:
- |Z| < 1: Neutral.
- 1 ≤ Z < 1.5: Moderately elevated (or depressed if negative).
- Z ≥ 1.5 (or ≤ −1.5): Strong deviation; consider cross‑checking with macro & credit indicators.

Why early data missing: A full window is required before a finite value appears (warm‑up period). To include earlier estimates you could lower `min_periods` or use expanding statistics.

### Percentile
Percentile rank of the latest value within its rolling window (0–1 scale). Shows *where* within recent history the current level sits, independent of standard deviation magnitude.

Guidelines:
- > 0.85: Upper tail of recent regime.
- < 0.20: Lower tail of recent regime.

Combining ZScore & Percentile reduces false positives (e.g., high percentile but low ZScore if volatility just collapsed).

---
## 3. Trend & De‑Trending
### Trend
Log‑linear fitted trend to `log(Buffett)` over the full available sample:
`log(Buffett_t) ≈ a + b * t`  → `Trend_t = exp(a + b * t)`.

Purpose: Capture secular drift (structural valuation re‑rating) so we can look at relative deviation net of that drift.

### TrendResidual
`Buffett / Trend` (dimensionless ratio).

Interpretation:
- = 1.00: Exactly on fitted secular trend.
- > 1.00: Above long‑run trend; e.g., 1.15 means 15% above fitted trajectory.
- < 1.00: Below trend.

Heuristic Zones:
- > 1.15: Historically stretched vs secular path.
- < 0.90: Historically depressed vs secular path.

Use with caution: A simple linear trend may lag regime shifts (e.g., persistent change in interest-rate regime). For robustness you could refit periodically or include macro factors (rates, inflation expectations).

---
## 4. Putting It All Together
A possible decision *framework* (illustrative only — NOT investment advice):

| Condition | Interpretation | Possible Action Bias |
|-----------|---------------|----------------------|
| ZScore > +1 AND Percentile > 0.85 AND TrendResidual > 1.10 | Stretched within current + secular contexts | Consider defensive tilt / slower incremental allocation |
| ZScore < -1 AND Percentile < 0.20 AND TrendResidual < 0.95 | Depressed vs regime & trend | Consider gradual risk add / rebalance toward equities |
| Mixed signals | Ambiguous | Neutral stance |

Always corroborate with earnings yield vs bond yields, credit spreads, liquidity measures, and micro breadth.

---
## 5. Command-Line Usage
```
# Basic (default 1990 start, 60-quarter window)
python buffet.py

# Custom start date
python buffet.py --start 1960-01-01

# Longer context window (20 years = 80 quarters)
python buffet.py --start 1985-01-01 --window 80
```
Outputs: `buffett_indicator_enhanced.png`, `buffett_indicator_enhanced.csv`.

---
## 6. CSV Columns Cheat Sheet
| Column | Meaning |
|--------|---------|
| Buffett_pct_of_GDP | Core valuation ratio (%) |
| RollingMean | Rolling window average (%) |
| RollingStd | Rolling window standard deviation (%) |
| ZScore | Standardized deviation from rolling mean |
| Percentile | Rolling window percentile rank (0–1) |
| Trend | Log-linear fitted secular trajectory |
| TrendResidual | Actual / Trend ratio |

---
## 7. Limitations & Risks
- Data depend on Yahoo Finance (^W5000) and FRED (GDP); outages or revisions can alter history.
- Structural breaks (policy regime, tax law, globalization dynamics) can invalidate simple trend or rolling assumptions.
- Distributions are not i.i.d. normal; z-scores are heuristic, not strict probabilities.
- Market cap used here (Wilshire 5000 index level) is a proxy; true total market cap series (billions) would refine accuracy.

---
## 8. Possible Extensions
- Include interest-rate adjustment (e.g., regress on 10y yield or ERP proxy).
- Add ±1.5σ / ±2σ lines & shading layers.
- Provide expanding-window early-period z-score fallback.
- Swap denominator to GNP or Corporate Gross Value Added for alternative scaling.
- Download an actual Wilshire market cap series from FRED (e.g., `WILL5000INDFC`) if API access restored.

---
## 9. Minimal Requirements
Current script actually only needs:
```
pandas
pandas-datareader
matplotlib
yfinance
```
Extra libs in `requirements.txt` (requests, lxml, beautifulsoup4) were legacy (safe to remove if not scraping).

---
## 9. Manufacturing Sector Health Tracker

### Overview
`manufacturing_tracker.py` provides comprehensive analysis of US manufacturing sector health using official Federal Reserve data. This tracker monitors real economic activity rather than survey-based indices.

### Data Sources (All from FRED)
- **Industrial Production: Manufacturing (IPMAN)** - Current manufacturing output
- **Capacity Utilization: Manufacturing (MCUMFN)** - Manufacturing efficiency 
- **Manufacturers' New Orders: Total Manufacturing (AMTMNO)** - Forward-looking demand
- **Manufacturers' New Orders: Durable Goods (DGORDER)** - Core manufacturing demand
- **Industrial Production: Durable/Nondurable** - Sector breakdown

### Key Features
- **Manufacturing Health Score** (0-100): Weighted composite indicator
- **Multi-component Analysis**: Production, capacity, orders, efficiency
- **Trend Analysis**: 3, 6, and 12-month trends with volatility metrics
- **Classification System**: Very Strong → Strong → Stable → Weak → Contracting
- **Professional Visualizations**: 6-panel comprehensive charts
- **Data Export**: CSV format for further analysis

### Usage
```bash
# Basic analysis (last 10 years)
python manufacturing_tracker.py

# Custom date range with exports
python manufacturing_tracker.py --start 2015-01-01 --save-chart manufacturing.png --export-csv manufacturing.csv

# Minimum data requirements
python manufacturing_tracker.py --min-data-points 100
```

### Interpretation Guide
- **Health Score 75+**: Very strong manufacturing expansion
- **Health Score 60-75**: Strong manufacturing growth
- **Health Score 40-60**: Stable/balanced conditions
- **Health Score 25-40**: Manufacturing weakness
- **Health Score <25**: Manufacturing contraction

### Output Files
- `manufacturing_analysis.png` - Comprehensive 6-panel visualization
- `manufacturing_data.csv` - Complete time series with derived metrics

---
## 10. International Trade & Global Integration Tracker

### Overview
`international_trade_tracker.py` provides comprehensive analysis of US international trade health and global economic integration using official Federal Reserve and market data. This tracker monitors trade flows, currency dynamics, commodity markets, and international capital movements.

### Data Sources (All from FRED and Market Data)
- **Trade Balance (BOPGSTB)** - Net exports indicating trade competitiveness
- **US Dollar Index (DTWEXBGS)** - Currency strength impact on trade
- **Commodity Price Indices** - WTI Oil, Copper, Producer Price Index
- **Foreign Investment Flows** - Treasury holdings and debt metrics
- **Trade Volume Indicators** - Exports, imports, and trade flows

### Key Features
- **Trade Health Score** (0-100): Weighted composite indicator of trade conditions
- **Multi-component Analysis**: Trade balance, currency, commodities, capital flows
- **Trend Analysis**: 12-month rolling windows with volatility metrics
- **Classification System**: Very Strong → Strong → Above Average → Average → Below Average → Weak → Very Weak
- **Professional Visualizations**: 6-panel comprehensive charts with trade dynamics
- **Data Export**: CSV format for further analysis

### Usage
```bash
# Basic analysis (from 1990)
python international_trade_tracker.py

# Custom date range and analysis window
python international_trade_tracker.py --start 2000-01-01 --window 12

# Minimum data requirements
python international_trade_tracker.py --min-data-points 100
```

### Interpretation Guide
- **Very Strong (85+ percentile)**: Exceptional trade conditions with strong exports and competitive positioning
- **Strong (70-85 percentile)**: Favorable trade environment with healthy export growth
- **Above Average (55-70 percentile)**: Generally positive conditions above historical norms
- **Average (45-55 percentile)**: Balanced trade conditions with mixed signals
- **Below Average (30-45 percentile)**: Somewhat challenging environment with headwinds
- **Weak (15-30 percentile)**: Difficult conditions with multiple stress indicators
- **Very Weak (<15 percentile)**: Severely constrained trade environment

### Output Files
- `international_trade_analysis.png` - Comprehensive 6-panel visualization
- `international_trade_analysis.csv` - Complete time series with derived metrics

---
## 11. Disclaimer
This tool is for educational / research use only and does **not** constitute investment advice. Past relationships do not guarantee future outcomes.

---
## 12. Quick Interpretation Flow
1. Look at core series vs RollingMean & band: inside or outside?
2. Check ZScore: magnitude & persistence.
3. Confirm with Percentile: truly tail or just modestly elevated?
4. Inspect TrendResidual: is move just secular drift or an overshoot beyond trend?
5. Synthesize with macro context before acting.

---
## 12. Consumer Health Tracker

### Overview
The **Consumer Health Tracker** provides comprehensive monitoring of the consumer sector, which represents ~70% of US GDP and is critical for economic growth and recession prediction.

### Key Indicators Tracked
- **Retail Sales**: Month-over-month and year-over-year growth across categories
- **Consumer Confidence**: OECD and University of Michigan sentiment indicators  
- **Personal Consumption Expenditures (PCE)**: Real consumer spending (durable goods, non-durable goods, services)
- **Consumer Credit**: Household borrowing trends and credit growth
- **Personal Savings Rate**: Consumer financial health and spending capacity
- **Energy Costs**: Gas prices and energy commodities affecting purchasing power

### Consumer Health Classifications
- **Very Strong**: Robust spending, high confidence, healthy credit growth
- **Strong**: Above-average consumer activity and sentiment
- **Above Average**: Moderate consumer strength with positive trends
- **Average**: Balanced consumer conditions, neither strong nor weak
- **Below Average**: Some consumer weakness emerging
- **Weak**: Declining consumer activity and confidence
- **Very Weak**: Severe consumer retrenchment, recession risk

### Key Features
- **Consumer Health Score** (Z-Score): Weighted composite indicator of consumer conditions
- **Multi-component Analysis**: Spending (40%), Confidence (25%), Financial Health (20%), Income/Purchasing Power (15%)
- **Trend Analysis**: Rolling windows with volatility metrics and growth rates
- **Professional Visualizations**: 6-panel comprehensive charts with consumer dynamics
- **Data Export**: CSV format for further analysis

### Usage
```bash
# Basic analysis (from 1990)
python consumer_health_tracker.py

# Custom date range and analysis window
python consumer_health_tracker.py --start 2000-01-01 --window 12

# Minimum data requirements
python consumer_health_tracker.py --min-data-points 50
```

### Output Files
- `consumer_health_analysis.png` – Multi-panel consumer sector visualization
- `consumer_health_analysis.csv` – Time series with all indicators and composite metrics

---
Feel free to request additional metrics, alternative trend modeling, or an HTML interactive dashboard.
