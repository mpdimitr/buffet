#!/usr/bin/env python3
"""Streamlined shipping tracker (single metric, single chart)."""

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
    "Shipping core signal tracks year-over-year growth in the Transportation Services Index: Freight (TSIFRGHT). "
    "Rising freight activity generally aligns with stronger real-economy goods movement, while weak or negative growth can "
    "indicate softening logistical and industrial demand."
)


def main() -> None:
    args = parse_common_args(
        description="Shipping tracker",
        default_start="2000-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return
    log_tracker_header("SHIPPING TRACKER", args.start, args.window, args.min_data_points)

    freight = fetch_fred_series("TSIFRGHT", args.start)
    if freight.empty:
        raise RuntimeError("Could not retrieve TSIFRGHT")

    monthly = to_period_end(freight, "ME")
    yoy = year_over_year_pct(monthly, periods=12).rename("Freight_Activity_YoY")
    metric = validate_series(yoy, args.min_data_points, "freight activity YoY")

    latest = metric.iloc[-1]
    print(f"Latest freight activity YoY: {latest:.2f}%")

    save_single_indicator_outputs(
        series=metric,
        output_png="primary_shipping_tracker.png",
        output_csv="primary_shipping_tracker.csv",
        chart_title="Shipping Core Signal: Freight Activity Growth (YoY)",
        y_label="Year-over-year percent",
        metric_name="Freight_Activity_YoY",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="lower_is_risk",
    )
    print("Saved: primary_shipping_tracker.png, primary_shipping_tracker.csv")


if __name__ == "__main__":
    main()
