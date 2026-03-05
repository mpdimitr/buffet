#!/usr/bin/env python3
"""Streamlined international trade tracker (single metric, single chart)."""

from streamline_core import (
    fetch_fred_series,
    log_tracker_header,
    parse_common_args,
    save_single_indicator_outputs,
    to_period_end,
    validate_series,
)

TRACKER_DESCRIPTION = (
    "International Trade core signal uses the U.S. trade balance level. "
    "A less negative or improving balance can indicate stronger external demand/competitiveness dynamics, "
    "while a widening deficit can reflect relative domestic demand strength or external softness depending on context."
)


def main() -> None:
    args = parse_common_args(
        description="International trade tracker",
        default_start="1990-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return
    log_tracker_header("INTERNATIONAL TRADE TRACKER", args.start, args.window, args.min_data_points)

    trade_balance = fetch_fred_series("BOPGSTB", args.start)
    if trade_balance.empty:
        raise RuntimeError("Could not retrieve BOPGSTB")

    metric = validate_series(to_period_end(trade_balance, "ME").rename("Trade_Balance"), args.min_data_points, "trade balance")
    latest = metric.iloc[-1]
    print(f"Latest trade balance: {latest:.2f} (billions of USD)")

    save_single_indicator_outputs(
        series=metric,
        output_png="international_trade_analysis.png",
        output_csv="international_trade_analysis.csv",
        chart_title="International Trade Core Signal: US Trade Balance",
        y_label="Billions of USD",
        metric_name="Trade_Balance",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="lower_is_risk",
    )
    print("Saved: international_trade_analysis.png, international_trade_analysis.csv")


if __name__ == "__main__":
    main()
