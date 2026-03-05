#!/usr/bin/env python3
"""Streamlined inflation expectations tracker (single metric, single chart)."""

from streamline_core import (
    fetch_fred_series,
    log_tracker_header,
    parse_common_args,
    save_single_indicator_outputs,
    to_period_end,
    validate_series,
)

TRACKER_DESCRIPTION = (
    "Inflation Expectations tracks the 5-year, 5-year forward inflation expectation rate. "
    "Higher readings can signal concern that inflation may remain above target over the medium term, "
    "while lower readings suggest better anchoring of inflation expectations."
)


def main() -> None:
    args = parse_common_args(
        description="Inflation expectations tracker",
        default_start="2003-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return

    log_tracker_header("INFLATION EXPECTATIONS TRACKER", args.start, args.window, args.min_data_points)

    expectations = fetch_fred_series("T5YIFR", args.start)
    if expectations.empty:
        raise RuntimeError("Could not retrieve T5YIFR")

    monthly = to_period_end(expectations, "ME").rename("Inflation_Expectations_5Y5Y")
    metric = validate_series(monthly, args.min_data_points, "inflation expectations")

    latest = metric.iloc[-1]
    print(f"Latest 5Y5Y inflation expectation: {latest:.2f}%")

    save_single_indicator_outputs(
        series=metric,
        output_png="inflation_expectations_analysis.png",
        output_csv="inflation_expectations_analysis.csv",
        chart_title="Inflation Expectations: 5Y5Y Forward Rate",
        y_label="Percent",
        metric_name="Inflation_Expectations_5Y5Y",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="higher_is_risk",
    )
    print("Saved: inflation_expectations_analysis.png, inflation_expectations_analysis.csv")


if __name__ == "__main__":
    main()
