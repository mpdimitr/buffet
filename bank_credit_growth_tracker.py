#!/usr/bin/env python3
"""Streamlined bank credit growth tracker (single metric, single chart)."""

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
    "Bank Credit Growth tracks year-over-year growth in total bank loans and leases. "
    "Stronger growth indicates healthier credit transmission into the real economy, while persistent slowdown can signal "
    "tighter lending and weaker forward demand."
)


def main() -> None:
    args = parse_common_args(
        description="Bank credit growth tracker",
        default_start="1990-01-01",
        default_window=12,
        default_min_data_points=24,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return

    log_tracker_header("BANK CREDIT GROWTH TRACKER", args.start, args.window, args.min_data_points)

    bank_credit = fetch_fred_series("TOTLL", args.start)
    if bank_credit.empty:
        raise RuntimeError("Could not retrieve TOTLL")

    yoy = year_over_year_pct(to_period_end(bank_credit, "ME"), periods=12).rename("Bank_Credit_Growth_YoY")
    metric = validate_series(yoy, args.min_data_points, "bank credit growth")

    latest = metric.iloc[-1]
    print(f"Latest bank credit YoY growth: {latest:.2f}%")

    save_single_indicator_outputs(
        series=metric,
        output_png="bank_credit_growth_analysis.png",
        output_csv="bank_credit_growth_analysis.csv",
        chart_title="Credit Supply: Bank Loans & Leases Growth (YoY)",
        y_label="Year-over-year percent",
        metric_name="Bank_Credit_Growth_YoY",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="lower_is_risk",
    )
    print("Saved: bank_credit_growth_analysis.png, bank_credit_growth_analysis.csv")


if __name__ == "__main__":
    main()
