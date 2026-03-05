#!/usr/bin/env python3
"""Streamlined initial claims tracker (single metric, single chart)."""

from streamline_core import (
    fetch_fred_series,
    log_tracker_header,
    parse_common_args,
    rolling_mean,
    save_single_indicator_outputs,
    to_period_end,
    validate_series,
)

TRACKER_DESCRIPTION = (
    "Initial Claims tracks the 4-week moving average of new unemployment insurance claims. "
    "Rising claims are typically an early warning of labor market stress and slowing growth momentum, "
    "while low and stable claims suggest resilient employment conditions."
)


def main() -> None:
    args = parse_common_args(
        description="Initial claims tracker",
        default_start="1990-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return

    log_tracker_header("INITIAL CLAIMS TRACKER", args.start, args.window, args.min_data_points)

    claims = fetch_fred_series("ICSA", args.start)
    if claims.empty:
        raise RuntimeError("Could not retrieve ICSA")

    weekly_avg4 = rolling_mean(claims, window=4)
    monthly = to_period_end(weekly_avg4, "ME").rename("Initial_Claims_4WkAvg")
    metric = validate_series(monthly, args.min_data_points, "initial claims")

    latest = metric.iloc[-1]
    print(f"Latest initial claims (4-week avg): {latest:,.0f}")

    save_single_indicator_outputs(
        series=metric,
        output_png="initial_claims_analysis.png",
        output_csv="initial_claims_analysis.csv",
        chart_title="Labor Stress Early Signal: Initial Claims (4-Week Avg)",
        y_label="Claims",
        metric_name="Initial_Claims_4WkAvg",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="higher_is_risk",
    )
    print("Saved: initial_claims_analysis.png, initial_claims_analysis.csv")


if __name__ == "__main__":
    main()
