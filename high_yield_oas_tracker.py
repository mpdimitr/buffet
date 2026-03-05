#!/usr/bin/env python3
"""Streamlined high-yield credit spread tracker (single metric, single chart)."""

from streamline_core import (
    fetch_fred_series,
    log_tracker_header,
    parse_common_args,
    save_single_indicator_outputs,
    to_period_end,
    validate_series,
)

TRACKER_DESCRIPTION = (
    "High-Yield OAS tracks option-adjusted spreads on below-investment-grade corporate bonds. "
    "Widening spreads indicate rising default risk and tighter financing conditions, while tighter spreads generally align "
    "with stronger risk appetite and easier credit transmission."
)


def main() -> None:
    args = parse_common_args(
        description="High-yield OAS tracker",
        default_start="1997-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return

    log_tracker_header("HIGH-YIELD OAS TRACKER", args.start, args.window, args.min_data_points)

    hy_oas = fetch_fred_series("BAMLH0A0HYM2", args.start)
    if hy_oas.empty:
        raise RuntimeError("Could not retrieve BAMLH0A0HYM2")

    monthly = to_period_end(hy_oas, "ME").rename("High_Yield_OAS")
    metric = validate_series(monthly, args.min_data_points, "high-yield OAS")

    latest = metric.iloc[-1]
    print(f"Latest HY OAS: {latest:.2f} pp")

    save_single_indicator_outputs(
        series=metric,
        output_png="high_yield_oas_analysis.png",
        output_csv="high_yield_oas_analysis.csv",
        chart_title="Credit Conditions: High-Yield OAS",
        y_label="Percentage points",
        metric_name="High_Yield_OAS",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="higher_is_risk",
    )
    print("Saved: high_yield_oas_analysis.png, high_yield_oas_analysis.csv")


if __name__ == "__main__":
    main()
