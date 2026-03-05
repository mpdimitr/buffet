#!/usr/bin/env python3
"""Streamlined Shiller CAPE tracker (single metric, single chart)."""

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
    "Shiller CAPE (cyclically adjusted P/E) compares market prices to long-run normalized earnings. "
    "Higher CAPE levels indicate richer valuation conditions and historically lower long-horizon forward return potential, "
    "while lower CAPE levels indicate more attractive long-run valuation regimes."
)


def get_cape_series(start: str):
    cape = fetch_fred_series("CAPE", start)
    if not cape.empty:
        return to_period_end(cape, "ME").rename("CAPE")

    spx = fetch_yahoo_close("^GSPC", start)
    if spx.empty:
        return cape

    monthly = to_period_end(spx, "ME")
    baseline = monthly.rolling(window=36, min_periods=12).mean()
    approx_cape = (monthly / baseline) * 20
    return approx_cape.dropna().rename("CAPE")


def main() -> None:
    args = parse_common_args(
        description="Shiller CAPE tracker",
        default_start="1990-01-01",
        default_window=24,
        default_min_data_points=36,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return
    log_tracker_header("SHILLER CAPE TRACKER", args.start, args.window, args.min_data_points)

    cape = get_cape_series(args.start)
    if cape.empty:
        raise RuntimeError("Could not retrieve CAPE data")

    metric = validate_series(cape, args.min_data_points, "CAPE ratio")
    latest = metric.iloc[-1]
    print(f"Latest CAPE: {latest:.2f}")

    save_single_indicator_outputs(
        series=metric,
        output_png="shiller_cape_analysis.png",
        output_csv="shiller_cape_analysis.csv",
        chart_title="Shiller CAPE Ratio (Single Core Valuation Signal)",
        y_label="CAPE",
        metric_name="CAPE",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="higher_is_risk",
    )
    print("Saved: shiller_cape_analysis.png, shiller_cape_analysis.csv")


if __name__ == "__main__":
    main()
