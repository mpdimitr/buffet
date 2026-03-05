#!/usr/bin/env python3
"""Streamlined real policy rate tracker (single metric, single chart)."""

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
    "Real Policy Rate tracks the effective federal funds rate minus core PCE inflation (year-over-year). "
    "Rising real rates generally imply tighter policy and more growth headwinds, while deeply negative real rates imply easier "
    "financial conditions that can support demand but may reheat inflation pressure."
)


def main() -> None:
    args = parse_common_args(
        description="Real policy rate tracker",
        default_start="1990-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return

    log_tracker_header("REAL POLICY RATE TRACKER", args.start, args.window, args.min_data_points)

    fed_funds = fetch_fred_series("FEDFUNDS", args.start)
    core_pce = fetch_fred_series("PCEPILFE", args.start)
    if fed_funds.empty or core_pce.empty:
        raise RuntimeError("Could not retrieve FEDFUNDS and/or PCEPILFE")

    ff_monthly = to_period_end(fed_funds, "ME")
    core_pce_yoy = year_over_year_pct(to_period_end(core_pce, "ME"), periods=12)

    real_rate = (ff_monthly - core_pce_yoy).dropna().rename("Real_Policy_Rate")
    metric = validate_series(real_rate, args.min_data_points, "real policy rate")

    latest = metric.iloc[-1]
    print(f"Latest real policy rate: {latest:.2f}%")

    save_single_indicator_outputs(
        series=metric,
        output_png="real_policy_rate_analysis.png",
        output_csv="real_policy_rate_analysis.csv",
        chart_title="Policy Stance: Real Fed Funds Rate",
        y_label="Percent",
        metric_name="Real_Policy_Rate",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="neutral",
    )
    print("Saved: real_policy_rate_analysis.png, real_policy_rate_analysis.csv")


if __name__ == "__main__":
    main()
