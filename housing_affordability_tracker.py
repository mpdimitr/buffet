#!/usr/bin/env python3
"""Streamlined housing affordability pressure tracker (single metric, single chart)."""

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
    "Housing Affordability Pressure combines 30-year mortgage rates with home price inflation pressure. "
    "Higher values indicate tighter affordability conditions for marginal buyers, which often precede weaker housing turnover "
    "and broader interest-rate sensitive slowdown."
)


def main() -> None:
    args = parse_common_args(
        description="Housing affordability pressure tracker",
        default_start="1990-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return

    log_tracker_header("HOUSING AFFORDABILITY TRACKER", args.start, args.window, args.min_data_points)

    mortgage_rate = fetch_fred_series("MORTGAGE30US", args.start)
    home_prices = fetch_fred_series("CSUSHPINSA", args.start)
    if mortgage_rate.empty or home_prices.empty:
        raise RuntimeError("Could not retrieve MORTGAGE30US and/or CSUSHPINSA")

    mortgage_m = to_period_end(mortgage_rate, "ME")
    home_price_yoy = year_over_year_pct(to_period_end(home_prices, "ME"), periods=12)

    pressure = (mortgage_m + (home_price_yoy / 10.0)).dropna().rename("Housing_Affordability_Pressure")
    metric = validate_series(pressure, args.min_data_points, "housing affordability pressure")

    latest = metric.iloc[-1]
    print(f"Latest housing affordability pressure: {latest:.2f}")

    save_single_indicator_outputs(
        series=metric,
        output_png="housing_affordability_analysis.png",
        output_csv="housing_affordability_analysis.csv",
        chart_title="Housing Cycle: Affordability Pressure Proxy",
        y_label="Composite level",
        metric_name="Housing_Affordability_Pressure",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="higher_is_risk",
    )
    print("Saved: housing_affordability_analysis.png, housing_affordability_analysis.csv")


if __name__ == "__main__":
    main()
