#!/usr/bin/env python3
"""Streamlined market indices tracker (S&P 500 + Nasdaq dual-axis chart)."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from streamline_core import (
    add_recession_shading,
    fetch_yahoo_close,
    log_tracker_header,
    parse_common_args,
    validate_series,
)

TRACKER_DESCRIPTION = (
    "Market indices tracker plots S&P 500 and Nasdaq Composite on separate y-axes to show broad equity trend "
    "and growth/tech sensitivity side-by-side. Recession bands are overlaid for cycle context, while status is "
    "derived from the pair's composite drawdown versus prior peaks."
)


OUTPUT_PNG = "market_indices_sp500_nasdaq.png"
OUTPUT_CSV = "market_indices_sp500_nasdaq.csv"


def classify_drawdown_status(drawdown_series: pd.Series) -> str:
    latest = float(drawdown_series.dropna().iloc[-1])
    if latest <= -0.25:
        return "Elevated Risk"
    if latest <= -0.12:
        return "Watch Zone"
    if latest <= -0.05:
        return "Normal"
    return "Favorable"


def main() -> None:
    args = parse_common_args(
        description="Market indices tracker",
        default_start="1990-01-01",
        default_window=12,
        default_min_data_points=36,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return

    log_tracker_header("MARKET INDICES TRACKER", args.start, args.window, args.min_data_points)

    sp500 = fetch_yahoo_close("^GSPC", args.start)
    nasdaq = fetch_yahoo_close("^IXIC", args.start)
    if sp500.empty or nasdaq.empty:
        raise RuntimeError("Could not retrieve required market index data (^GSPC and ^IXIC)")

    data = pd.concat([sp500.rename("SP500"), nasdaq.rename("NASDAQ")], axis=1).dropna().sort_index()
    start_date = pd.to_datetime(args.start)
    end_date = pd.to_datetime(args.end)
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    if data.empty:
        raise RuntimeError("No overlapping market-index data in selected date range")

    validate_series(data["SP500"], args.min_data_points, "S&P 500")
    validate_series(data["NASDAQ"], args.min_data_points, "Nasdaq Composite")

    sp500_drawdown = (data["SP500"] / data["SP500"].cummax()) - 1.0
    nasdaq_drawdown = (data["NASDAQ"] / data["NASDAQ"].cummax()) - 1.0
    composite_drawdown = ((sp500_drawdown + nasdaq_drawdown) / 2.0).rename("Composite_Drawdown")

    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax2 = ax1.twinx()

    add_recession_shading(ax1, data.index.min(), data.index.max(), alpha=0.14)

    line1, = ax1.plot(data.index, data["SP500"], color="tab:blue", linewidth=1.8, label="S&P 500")
    line2, = ax2.plot(data.index, data["NASDAQ"], color="tab:orange", linewidth=1.5, label="Nasdaq Composite")

    ax1.set_title("U.S. Equity Benchmarks: S&P 500 vs Nasdaq Composite", fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("S&P 500 Level", color="tab:blue")
    ax2.set_ylabel("Nasdaq Composite Level", color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_xlim(start_date, end_date)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_minor_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.grid(alpha=0.25)

    # Keep legend deterministic and clean across both axes.
    ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)

    out = data.copy()
    out["SP500_Drawdown"] = sp500_drawdown
    out["NASDAQ_Drawdown"] = nasdaq_drawdown
    out["Composite_Drawdown"] = composite_drawdown
    out.to_csv(OUTPUT_CSV, index_label="date")

    status = classify_drawdown_status(composite_drawdown)
    print(f"Latest S&P 500 close: {data['SP500'].iloc[-1]:.2f}")
    print(f"Latest Nasdaq close: {data['NASDAQ'].iloc[-1]:.2f}")
    print(f"Current status: {status}")
    print(f"Saved: {OUTPUT_PNG}, {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
