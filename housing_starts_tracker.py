#!/usr/bin/env python3
"""Streamlined housing starts tracker (single metric, single chart)."""

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
    "Housing Starts tracks year-over-year growth in residential housing starts. "
    "Housing is a cyclical, rate-sensitive sector, so persistent weakness often signals broader demand deceleration, "
    "while strong growth indicates improving construction momentum."
)


def main() -> None:
    args = parse_common_args(
        description="Housing starts tracker",
        default_start="1990-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return

    log_tracker_header("HOUSING STARTS TRACKER", args.start, args.window, args.min_data_points)

    starts = fetch_fred_series("HOUST", args.start)
    if starts.empty:
        raise RuntimeError("Could not retrieve HOUST")

    starts_yoy = year_over_year_pct(to_period_end(starts, "ME"), periods=12).rename("Housing_Starts_YoY")
    metric = validate_series(starts_yoy, args.min_data_points, "housing starts YoY")

    latest = metric.iloc[-1]
    print(f"Latest housing starts YoY growth: {latest:.2f}%")

    save_single_indicator_outputs(
        series=metric,
        output_png="housing_starts_analysis.png",
        output_csv="housing_starts_analysis.csv",
        chart_title="Housing Activity: Starts Growth (YoY)",
        y_label="Year-over-year percent",
        metric_name="Housing_Starts_YoY",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="lower_is_risk",
    )
    print("Saved: housing_starts_analysis.png, housing_starts_analysis.csv")


if __name__ == "__main__":
    main()
