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
## 10. Disclaimer
This tool is for educational / research use only and does **not** constitute investment advice. Past relationships do not guarantee future outcomes.

---
## 11. Quick Interpretation Flow
1. Look at core series vs RollingMean & band: inside or outside?
2. Check ZScore: magnitude & persistence.
3. Confirm with Percentile: truly tail or just modestly elevated?
4. Inspect TrendResidual: is move just secular drift or an overshoot beyond trend?
5. Synthesize with macro context before acting.

---
Feel free to request additional metrics, alternative trend modeling, or an HTML interactive dashboard.
