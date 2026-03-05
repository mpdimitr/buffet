#!/usr/bin/env python3
"""Shared utilities for streamlined single-metric economic trackers."""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf


NBER_RECESSIONS = [
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

STATUS_ELEVATED = "Elevated Risk"
STATUS_WATCH = "Watch Zone"
STATUS_NORMAL = "Normal"
STATUS_FAVORABLE = "Favorable"


def parse_common_args(description: str, default_start: str = "1990-01-01", default_window: int = 12, default_min_data_points: int = 24) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--start", type=str, default=default_start, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", type=str, default=dt.datetime.today().strftime("%Y-%m-%d"), help="End date in YYYY-MM-DD format")
    parser.add_argument("--window", type=int, default=default_window, help="Rolling window used for smoothing")
    parser.add_argument("--min-data-points", type=int, default=default_min_data_points, help="Minimum required observations")
    parser.add_argument("--describe", action="store_true", help="Print a one-paragraph explanation of the tracked metric and exit")
    return parser.parse_args()


def classify_current_status(series: pd.Series, risk_direction: str = "higher_is_risk") -> str:
    values = series.dropna()
    if values.empty:
        return STATUS_NORMAL

    latest = float(values.iloc[-1])

    if risk_direction == "higher_is_risk":
        q25 = float(values.quantile(0.25))
        q75 = float(values.quantile(0.75))
        q90 = float(values.quantile(0.90))
        if latest <= q25:
            return STATUS_FAVORABLE
        if latest <= q75:
            return STATUS_NORMAL
        if latest <= q90:
            return STATUS_WATCH
        return STATUS_ELEVATED

    if risk_direction == "lower_is_risk":
        q10 = float(values.quantile(0.10))
        q25 = float(values.quantile(0.25))
        q75 = float(values.quantile(0.75))
        if latest <= q10:
            return STATUS_ELEVATED
        if latest <= q25:
            return STATUS_WATCH
        if latest <= q75:
            return STATUS_NORMAL
        return STATUS_FAVORABLE

    median = float(values.median())
    abs_dev = (values - median).abs()
    d50 = float(abs_dev.quantile(0.50))
    d75 = float(abs_dev.quantile(0.75))
    d90 = float(abs_dev.quantile(0.90))
    latest_dev = abs(latest - median)
    if latest_dev <= d50:
        return STATUS_FAVORABLE
    if latest_dev <= d75:
        return STATUS_NORMAL
    if latest_dev <= d90:
        return STATUS_WATCH
    return STATUS_ELEVATED


def parse_start_date(start: str, fallback: str = "1990-01-01") -> pd.Timestamp:
    try:
        return pd.to_datetime(start)
    except Exception:
        return pd.to_datetime(fallback)


def fetch_fred_series(series_id: str, start_date: str) -> pd.Series:
    try:
        data = pdr.get_data_fred(series_id, start=start_date)
        if data.empty:
            return pd.Series(dtype=float)
        return data.iloc[:, 0].dropna().rename(series_id)
    except Exception:
        return pd.Series(dtype=float)


def fetch_yahoo_close(ticker: str, start_date: str) -> pd.Series:
    data = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
    if data.empty:
        return pd.Series(dtype=float)

    if "Close" in data.columns:
        close = data["Close"]
    elif ("Close", ticker) in data.columns:
        close = data[("Close", ticker)]
    else:
        close = data.iloc[:, 0]

    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 0:
            return pd.Series(dtype=float)
        close = close.iloc[:, 0]
    return close.dropna().rename(ticker)


def to_period_end(series: pd.Series, frequency: str) -> pd.Series:
    return series.resample(frequency).last().dropna()


def year_over_year_pct(series: pd.Series, periods: int = 12) -> pd.Series:
    return (series.pct_change(periods=periods) * 100).dropna()


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(3, window // 2)).mean()


def add_recession_shading(ax: plt.Axes, data_start: Optional[pd.Timestamp], data_end: Optional[pd.Timestamp], alpha: float = 0.12) -> None:
    if data_start is None or data_end is None:
        return
    for start_str, end_str in NBER_RECESSIONS:
        start_dt = pd.to_datetime(start_str)
        end_dt = pd.to_datetime(end_str)
        if end_dt < data_start or start_dt > data_end:
            continue
        ax.axvspan(start_dt, end_dt, alpha=alpha, color="gray")


def validate_series(series: pd.Series, min_data_points: int, name: str) -> pd.Series:
    clean = series.dropna()
    if len(clean) < min_data_points:
        raise ValueError(f"Insufficient data for {name}: {len(clean)} < {min_data_points}")
    return clean


def save_single_indicator_outputs(
    series: pd.Series,
    output_png: str,
    output_csv: str,
    chart_title: str,
    y_label: str,
    metric_name: str,
    window: int,
    add_recessions: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    major_interval_years: int = 2,
    risk_direction: str = "higher_is_risk",
) -> str:
    series = series.dropna().sort_index()
    smoothed = rolling_mean(series, window)

    if start_date is not None:
        x_start = pd.to_datetime(start_date)
    else:
        x_start = series.index.min()

    if end_date is not None:
        x_end = pd.to_datetime(end_date)
    else:
        x_end = pd.to_datetime(dt.datetime.today())

    hist_q10 = series.quantile(0.10)
    hist_q25 = series.quantile(0.25)
    hist_q75 = series.quantile(0.75)
    hist_q90 = series.quantile(0.90)
    hist_min = series.min()
    hist_max = series.max()

    rolling_std = series.rolling(window=window, min_periods=max(3, window // 2)).std()
    upper_1std = smoothed + rolling_std
    lower_1std = smoothed - rolling_std

    fig, ax = plt.subplots(figsize=(15, 7))

    if risk_direction == "higher_is_risk":
        ax.axhspan(hist_min, hist_q25, color="#B8E6C1", alpha=0.22, label="Favorable historical zone")
        ax.axhspan(hist_q25, hist_q75, color="#B0D8FF", alpha=0.16, label="Normal historical zone")
        ax.axhspan(hist_q75, hist_q90, color="#FFD8A8", alpha=0.20, label="Watch zone")
        ax.axhspan(hist_q90, hist_max, color="#F5B5B5", alpha=0.22, label="Elevated-risk zone")
    elif risk_direction == "lower_is_risk":
        ax.axhspan(hist_min, hist_q10, color="#F5B5B5", alpha=0.22, label="Elevated-risk zone")
        ax.axhspan(hist_q10, hist_q25, color="#FFD8A8", alpha=0.20, label="Watch zone")
        ax.axhspan(hist_q25, hist_q75, color="#B0D8FF", alpha=0.16, label="Normal historical zone")
        ax.axhspan(hist_q75, hist_max, color="#B8E6C1", alpha=0.22, label="Favorable historical zone")
    else:
        ax.axhspan(hist_q10, hist_q90, color="#B0D8FF", alpha=0.18, label="Historical 10-90% range")
        ax.axhspan(hist_q25, hist_q75, color="#74B3FF", alpha=0.22, label="Historical 25-75% range")

    if add_recessions:
        add_recession_shading(ax, x_start, x_end)

    ax.fill_between(series.index, lower_1std, upper_1std, color="#F6C46A", alpha=0.22, label=f"{window}-period ±1σ")

    ax.plot(series.index, series, color="tab:blue", linewidth=1.6, label=metric_name)
    ax.plot(smoothed.index, smoothed, color="tab:orange", linewidth=1.2, linestyle="--", label=f"{window}-period mean")
    ax.scatter(series.index[-1], series.iloc[-1], color="tab:blue", s=30, zorder=5)

    ax.set_title(chart_title, fontweight="bold")
    ax.set_ylabel(y_label)
    ax.set_xlabel("Date")
    ax.set_xlim(x_start, x_end)
    ax.xaxis.set_major_locator(mdates.YearLocator(major_interval_years))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for handle, label in zip(handles, labels):
        if label not in dedup:
            dedup[label] = handle
    ax.legend(dedup.values(), dedup.keys(), loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    out = pd.DataFrame({metric_name: series, f"{metric_name}_rolling_mean": smoothed})
    out.to_csv(output_csv, index_label="date")

    status = classify_current_status(series, risk_direction=risk_direction)
    print(f"Current status: {status}")
    return status


def log_tracker_header(name: str, start: str, window: int, min_data_points: int) -> None:
    print(f"📊 {name}")
    print("=" * 60)
    print(f"Start: {start} | Window: {window} | Min points: {min_data_points}")


def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")
