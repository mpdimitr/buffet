#!/usr/bin/env python3
"""Streamlined manufacturing tracker (single metric, single chart)."""

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
    "Manufacturing core signal tracks year-over-year growth in Industrial Production: Manufacturing (IPMAN). "
    "Positive acceleration typically reflects stronger goods-sector demand, while persistent deceleration or contraction often "
    "signals cyclical slowdown pressure."
)


def main() -> None:
    args = parse_common_args(
        description="Manufacturing tracker",
        default_start="1990-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return
    log_tracker_header("MANUFACTURING TRACKER", args.start, args.window, args.min_data_points)

    ipman = fetch_fred_series("IPMAN", args.start)
    if ipman.empty:
        raise RuntimeError("Could not retrieve IPMAN")

    monthly = to_period_end(ipman, "ME")
    yoy = year_over_year_pct(monthly, periods=12).rename("Manufacturing_Output_YoY")
    metric = validate_series(yoy, args.min_data_points, "manufacturing output YoY")

    latest = metric.iloc[-1]
    print(f"Latest manufacturing output YoY: {latest:.2f}%")

    save_single_indicator_outputs(
        series=metric,
        output_png="manufacturing_analysis.png",
        output_csv="manufacturing_analysis.csv",
        chart_title="Manufacturing Core Signal: Industrial Production Growth (YoY)",
        y_label="Year-over-year percent",
        metric_name="Manufacturing_Output_YoY",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="lower_is_risk",
    )
    print("Saved: manufacturing_analysis.png, manufacturing_analysis.csv")


if __name__ == "__main__":
    main()
