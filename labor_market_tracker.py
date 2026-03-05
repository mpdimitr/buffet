#!/usr/bin/env python3
"""Streamlined labor market tracker (single metric, single chart)."""

from streamline_core import (
    fetch_fred_series,
    log_tracker_header,
    parse_common_args,
    save_single_indicator_outputs,
    to_period_end,
    validate_series,
)

TRACKER_DESCRIPTION = (
    "Labor Market core signal tracks the U.S. unemployment rate. "
    "Lower unemployment generally reflects stronger labor demand and economic momentum, while sustained increases often indicate "
    "cooling activity and can be a lagging confirmation of downturn conditions."
)


def main() -> None:
    args = parse_common_args(
        description="Labor market tracker",
        default_start="1990-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return
    log_tracker_header("LABOR MARKET TRACKER", args.start, args.window, args.min_data_points)

    unrate = fetch_fred_series("UNRATE", args.start)
    if unrate.empty:
        raise RuntimeError("Could not retrieve UNRATE")

    metric = validate_series(to_period_end(unrate, "ME").rename("Unemployment_Rate"), args.min_data_points, "unemployment rate")
    latest = metric.iloc[-1]
    print(f"Latest unemployment rate: {latest:.2f}%")

    save_single_indicator_outputs(
        series=metric,
        output_png="labor_market_analysis.png",
        output_csv="labor_market_analysis.csv",
        chart_title="Labor Market Core Signal: Unemployment Rate",
        y_label="Percent",
        metric_name="Unemployment_Rate",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="higher_is_risk",
    )
    print("Saved: labor_market_analysis.png, labor_market_analysis.csv")


if __name__ == "__main__":
    main()
