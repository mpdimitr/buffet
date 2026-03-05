#!/usr/bin/env python3
"""Streamlined Buffett Indicator tracker (single metric, single chart)."""

import pandas as pd

from streamline_core import (
    fetch_fred_series,
    fetch_yahoo_close,
    log_tracker_header,
    parse_common_args,
    save_single_indicator_outputs,
    to_period_end,
    validate_series,
)

TRACKER_DESCRIPTION = (
    "Buffett Indicator tracks total U.S. equity market value relative to U.S. GDP. "
    "Higher values indicate richer aggregate equity valuation versus the size of the economy, "
    "which is typically interpreted as higher long-term valuation risk rather than a short-term timing signal."
)


def main() -> None:
    args = parse_common_args(
        description="Buffett Indicator tracker",
        default_start="1990-01-01",
        default_window=20,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return
    log_tracker_header("BUFFETT INDICATOR", args.start, args.window, args.min_data_points)

    wilshire = fetch_yahoo_close("^W5000", args.start)
    gdp = fetch_fred_series("GDP", args.start)

    if wilshire.empty or gdp.empty:
        raise RuntimeError("Could not retrieve required data (Wilshire 5000 and GDP)")

    wilshire_q = to_period_end(wilshire, "QE")
    gdp_q = to_period_end(gdp, "QE")
    aligned = pd.concat([wilshire_q.rename("Wilshire"), gdp_q.rename("GDP")], axis=1).dropna()
    buffett_pct = (aligned["Wilshire"] / aligned["GDP"]) * 100

    metric = validate_series(buffett_pct.rename("Buffett_pct_of_GDP"), args.min_data_points, "Buffett indicator")
    latest = metric.iloc[-1]
    print(f"Latest Buffett % of GDP: {latest:.2f}%")

    save_single_indicator_outputs(
        series=metric,
        output_png="buffett_indicator_enhanced.png",
        output_csv="buffett_indicator_enhanced.csv",
        chart_title="Buffett Indicator (% of GDP)",
        y_label="Percent of GDP",
        metric_name="Buffett_pct_of_GDP",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="higher_is_risk",
    )
    print("Saved: buffett_indicator_enhanced.png, buffett_indicator_enhanced.csv")


if __name__ == "__main__":
    main()
