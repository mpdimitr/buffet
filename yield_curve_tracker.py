#!/usr/bin/env python3
"""Streamlined yield curve tracker (single metric, single chart)."""

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
    "Yield Curve core signal uses the 10Y-3M Treasury spread. "
    "When the spread compresses toward zero or turns negative (inversion), recession risk is generally interpreted as rising; "
    "positive, wider spreads usually indicate less immediate cyclical stress."
)


def main() -> None:
    args = parse_common_args(
        description="Yield curve tracker",
        default_start="1990-01-01",
        default_window=12,
        default_min_data_points=36,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return
    log_tracker_header("YIELD CURVE TRACKER", args.start, args.window, args.min_data_points)

    ten_year = fetch_fred_series("DGS10", args.start)
    three_month = fetch_fred_series("DGS3MO", args.start)
    if ten_year.empty or three_month.empty:
        raise RuntimeError("Could not retrieve DGS10 or DGS3MO from FRED")

    monthly = pd.concat(
        [to_period_end(ten_year, "ME").rename("DGS10"), to_period_end(three_month, "ME").rename("DGS3MO")],
        axis=1,
    ).dropna()
    spread = (monthly["DGS10"] - monthly["DGS3MO"]).rename("Yield_Spread_10Y_3M")
    metric = validate_series(spread, args.min_data_points, "10Y-3M spread")

    latest = metric.iloc[-1]
    regime = "INVERTED" if latest < 0 else "NORMAL"
    print(f"Latest 10Y-3M spread: {latest:.2f}% ({regime})")

    save_single_indicator_outputs(
        series=metric,
        output_png="yield_curve_analysis.png",
        output_csv="yield_curve_analysis.csv",
        chart_title="Yield Curve Core Signal: 10Y - 3M Spread",
        y_label="Spread (percentage points)",
        metric_name="Yield_Spread_10Y_3M",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="lower_is_risk",
    )
    print("Saved: yield_curve_analysis.png, yield_curve_analysis.csv")


if __name__ == "__main__":
    main()
