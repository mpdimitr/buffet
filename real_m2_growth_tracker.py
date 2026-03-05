#!/usr/bin/env python3
"""Streamlined real M2 growth tracker (single metric, single chart)."""

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
    "Real M2 Growth tracks broad money supply growth adjusted for CPI inflation (M2 YoY minus CPI YoY). "
    "Higher readings usually imply stronger liquidity support for nominal activity, while weak or negative readings "
    "indicate tighter real liquidity conditions."
)


def main() -> None:
    args = parse_common_args(
        description="Real M2 growth tracker",
        default_start="1990-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return

    log_tracker_header("REAL M2 GROWTH TRACKER", args.start, args.window, args.min_data_points)

    m2 = fetch_fred_series("M2SL", args.start)
    cpi = fetch_fred_series("CPIAUCSL", args.start)
    if m2.empty or cpi.empty:
        raise RuntimeError("Could not retrieve M2SL and/or CPIAUCSL")

    m2_yoy = year_over_year_pct(to_period_end(m2, "ME"), periods=12)
    cpi_yoy = year_over_year_pct(to_period_end(cpi, "ME"), periods=12)
    real_m2_growth = (m2_yoy - cpi_yoy).dropna().rename("Real_M2_Growth")
    metric = validate_series(real_m2_growth, args.min_data_points, "real M2 growth")

    latest = metric.iloc[-1]
    print(f"Latest real M2 growth: {latest:.2f}%")

    save_single_indicator_outputs(
        series=metric,
        output_png="real_m2_growth_analysis.png",
        output_csv="real_m2_growth_analysis.csv",
        chart_title="Liquidity Pulse: Real M2 Growth",
        y_label="Year-over-year percent",
        metric_name="Real_M2_Growth",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="lower_is_risk",
    )
    print("Saved: real_m2_growth_analysis.png, real_m2_growth_analysis.csv")


if __name__ == "__main__":
    main()
