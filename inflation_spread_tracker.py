#!/usr/bin/env python3
"""Streamlined inflation spread tracker (single metric, single chart)."""

from streamline_core import (
    fetch_fred_series,
    log_tracker_header,
    parse_common_args,
    save_single_indicator_outputs,
    to_period_end,
    validate_series,
    year_over_year_pct,
)

TRACKER_DESCRIPTION = (
    "Inflation Spread tracks the gap between headline CPI and Core PCE inflation (both year-over-year). "
    "A wider positive spread often signals stronger energy and goods price pass-through, while narrowing spreads can indicate "
    "disinflation in the most volatile components and a more stable inflation backdrop."
)


def main() -> None:
    args = parse_common_args(
        description="Inflation spread tracker",
        default_start="1990-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return

    log_tracker_header("INFLATION SPREAD TRACKER", args.start, args.window, args.min_data_points)

    cpi = fetch_fred_series("CPIAUCSL", args.start)
    core_pce = fetch_fred_series("PCEPILFE", args.start)
    if cpi.empty or core_pce.empty:
        raise RuntimeError("Could not retrieve CPIAUCSL and/or PCEPILFE")

    cpi_yoy = year_over_year_pct(to_period_end(cpi, "ME"), periods=12)
    core_pce_yoy = year_over_year_pct(to_period_end(core_pce, "ME"), periods=12)

    spread = (cpi_yoy - core_pce_yoy).dropna().rename("CPI_minus_CorePCE_YoY")
    metric = validate_series(spread, args.min_data_points, "inflation spread")

    latest = metric.iloc[-1]
    print(f"Latest CPI-CorePCE spread: {latest:.2f} pp")

    save_single_indicator_outputs(
        series=metric,
        output_png="inflation_spread_analysis.png",
        output_csv="inflation_spread_analysis.csv",
        chart_title="Inflation Spread: CPI YoY minus Core PCE YoY",
        y_label="Percentage points",
        metric_name="CPI_minus_CorePCE_YoY",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="neutral",
    )
    print("Saved: inflation_spread_analysis.png, inflation_spread_analysis.csv")


if __name__ == "__main__":
    main()
