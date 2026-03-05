#!/usr/bin/env python3
"""Streamlined credit conditions tracker (single metric, single chart)."""

import pandas as pd

from streamline_core import (
    fetch_fred_series,
    log_tracker_header,
    parse_common_args,
    save_single_indicator_outputs,
    to_period_end,
    validate_series,
)

TRACKER_DESCRIPTION = (
    "Credit Conditions core signal is the BAA minus 10Y Treasury spread. "
    "Wider spreads usually indicate tighter financial conditions and higher perceived credit risk, "
    "while narrower spreads suggest easier credit transmission to the real economy."
)


def main() -> None:
    args = parse_common_args(
        description="Credit conditions tracker",
        default_start="1990-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return
    log_tracker_header("CREDIT CONDITIONS TRACKER", args.start, args.window, args.min_data_points)

    baa = fetch_fred_series("BAA", args.start)
    treasury_10y = fetch_fred_series("DGS10", args.start)
    if baa.empty or treasury_10y.empty:
        raise RuntimeError("Could not retrieve BAA or DGS10")

    monthly = pd.concat(
        [to_period_end(baa, "ME").rename("BAA"), to_period_end(treasury_10y, "ME").rename("DGS10")],
        axis=1,
    ).dropna()
    spread = (monthly["BAA"] - monthly["DGS10"]).rename("BAA_Treasury_Spread")
    metric = validate_series(spread, args.min_data_points, "BAA-Treasury spread")

    latest = metric.iloc[-1]
    print(f"Latest BAA-Treasury spread: {latest:.2f}%")

    save_single_indicator_outputs(
        series=metric,
        output_png="credit_conditions_analysis.png",
        output_csv="credit_conditions_analysis.csv",
        chart_title="Credit Conditions Core Signal: BAA - 10Y Treasury Spread",
        y_label="Spread (percentage points)",
        metric_name="BAA_Treasury_Spread",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="higher_is_risk",
    )
    print("Saved: credit_conditions_analysis.png, credit_conditions_analysis.csv")


if __name__ == "__main__":
    main()
