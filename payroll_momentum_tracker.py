#!/usr/bin/env python3
"""Streamlined payroll momentum tracker (single metric, single chart)."""

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
    "Payroll Momentum tracks year-over-year growth in nonfarm payroll employment. "
    "Higher and stable payroll growth reflects resilient labor demand, while sustained deceleration typically signals "
    "late-cycle softening and rising macro downside risk."
)


def main() -> None:
    args = parse_common_args(
        description="Payroll momentum tracker",
        default_start="1990-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return

    log_tracker_header("PAYROLL MOMENTUM TRACKER", args.start, args.window, args.min_data_points)

    payrolls = fetch_fred_series("PAYEMS", args.start)
    if payrolls.empty:
        raise RuntimeError("Could not retrieve PAYEMS")

    payroll_yoy = year_over_year_pct(to_period_end(payrolls, "ME"), periods=12).rename("Payrolls_YoY")
    metric = validate_series(payroll_yoy, args.min_data_points, "payroll momentum")

    latest = metric.iloc[-1]
    print(f"Latest payroll YoY growth: {latest:.2f}%")

    save_single_indicator_outputs(
        series=metric,
        output_png="payroll_momentum_analysis.png",
        output_csv="payroll_momentum_analysis.csv",
        chart_title="Labor Depth: Nonfarm Payroll Growth (YoY)",
        y_label="Year-over-year percent",
        metric_name="Payrolls_YoY",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="lower_is_risk",
    )
    print("Saved: payroll_momentum_analysis.png, payroll_momentum_analysis.csv")


if __name__ == "__main__":
    main()
