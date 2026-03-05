#!/usr/bin/env python3
"""Streamlined consumer health tracker (single metric, single chart)."""

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
    "Consumer Health core signal tracks year-over-year growth in retail sales. "
    "Higher growth typically implies stronger household demand, while persistent slowing or negative growth can indicate "
    "consumer fatigue and weaker near-term growth momentum."
)


def main() -> None:
    args = parse_common_args(
        description="Consumer health tracker",
        default_start="1990-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return
    log_tracker_header("CONSUMER HEALTH TRACKER", args.start, args.window, args.min_data_points)

    retail_sales = fetch_fred_series("RSAFS", args.start)
    if retail_sales.empty:
        raise RuntimeError("Could not retrieve RSAFS")

    monthly = to_period_end(retail_sales, "ME")
    yoy = year_over_year_pct(monthly, periods=12).rename("Retail_Sales_YoY")
    metric = validate_series(yoy, args.min_data_points, "retail sales YoY")

    latest = metric.iloc[-1]
    print(f"Latest retail sales YoY: {latest:.2f}%")

    save_single_indicator_outputs(
        series=metric,
        output_png="consumer_health_analysis.png",
        output_csv="consumer_health_analysis.csv",
        chart_title="Consumer Core Signal: Retail Sales Growth (YoY)",
        y_label="Year-over-year percent",
        metric_name="Retail_Sales_YoY",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="lower_is_risk",
    )
    print("Saved: consumer_health_analysis.png, consumer_health_analysis.csv")


if __name__ == "__main__":
    main()
