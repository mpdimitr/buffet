#!/usr/bin/env python3
"""Streamlined household balance tracker (single metric, single chart)."""

from streamline_core import (
    fetch_fred_series,
    log_tracker_header,
    parse_common_args,
    save_single_indicator_outputs,
    to_period_end,
    validate_series,
)

TRACKER_DESCRIPTION = (
    "Household Balance Buffer tracks the spread between the personal saving rate and household debt-service burden. "
    "A larger positive buffer suggests stronger household shock absorption capacity, while compression indicates less flexibility "
    "to absorb income or financing stress."
)


def main() -> None:
    args = parse_common_args(
        description="Household balance buffer tracker",
        default_start="1990-01-01",
        default_window=8,
        default_min_data_points=12,
    )
    if args.describe:
        print(TRACKER_DESCRIPTION)
        return

    log_tracker_header("HOUSEHOLD BALANCE TRACKER", args.start, args.window, args.min_data_points)

    saving_rate = fetch_fred_series("PSAVERT", args.start)
    debt_service = fetch_fred_series("TDSP", args.start)
    if saving_rate.empty or debt_service.empty:
        raise RuntimeError("Could not retrieve PSAVERT and/or TDSP")

    saving_q = to_period_end(saving_rate, "QE")
    debt_q = to_period_end(debt_service, "QE")

    buffer = (saving_q - debt_q).dropna().rename("Household_Buffer")
    metric = validate_series(buffer, args.min_data_points, "household balance buffer")

    latest = metric.iloc[-1]
    print(f"Latest household buffer: {latest:.2f} pp")

    save_single_indicator_outputs(
        series=metric,
        output_png="household_balance_analysis.png",
        output_csv="household_balance_analysis.csv",
        chart_title="Household Sector: Savings minus Debt-Service Buffer",
        y_label="Percentage points",
        metric_name="Household_Buffer",
        window=args.window,
        start_date=args.start,
        end_date=args.end,
        risk_direction="lower_is_risk",
        major_interval_years=2,
    )
    print("Saved: household_balance_analysis.png, household_balance_analysis.csv")


if __name__ == "__main__":
    main()
