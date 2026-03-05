#!/usr/bin/env python3
"""Streamlined corporate earnings tracker (single metric, single chart)."""

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
    "Corporate Earnings core signal tracks year-over-year growth in corporate profits (CPATAX). "
    "Stronger profit growth often supports investment and hiring, while weakening or contracting profits can signal "
    "rising business-cycle stress."
)


def main() -> None:
    args = parse_common_args(
        description="Corporate earnings tracker",
        default_start="1990-01-01",
        default_window=8,
        default_min_data_points=16,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return
    log_tracker_header("CORPORATE EARNINGS TRACKER", args.start, args.window, args.min_data_points)

    profits = fetch_fred_series("CPATAX", args.start)
    if profits.empty:
        raise RuntimeError("Could not retrieve CPATAX")

    quarterly = to_period_end(profits, "QE")
    yoy = year_over_year_pct(quarterly, periods=4).rename("Corporate_Profits_YoY")
    metric = validate_series(yoy, args.min_data_points, "corporate profits YoY")

    latest = metric.iloc[-1]
    print(f"Latest corporate profits YoY: {latest:.2f}%")

    save_single_indicator_outputs(
        series=metric,
        output_png="corporate_earnings_analysis.png",
        output_csv="corporate_earnings_analysis.csv",
        chart_title="Corporate Core Signal: Corporate Profits Growth (YoY)",
        y_label="Year-over-year percent",
        metric_name="Corporate_Profits_YoY",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="lower_is_risk",
    )
    print("Saved: corporate_earnings_analysis.png, corporate_earnings_analysis.csv")


if __name__ == "__main__":
    main()
