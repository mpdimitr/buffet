#!/usr/bin/env python3
"""Run all present trackers on one shared interval and build a vertically aligned comparison PDF."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import textwrap
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages


TRACKERS = [
    {"script": "buffet_tracker.py", "chart": "buffett_indicator_enhanced.png", "min_points": 8},
    {"script": "inflation_spread_tracker.py", "chart": "inflation_spread_analysis.png", "min_points": 24},
    {"script": "real_policy_rate_tracker.py", "chart": "real_policy_rate_analysis.png", "min_points": 24},
    {"script": "inflation_expectations_tracker.py", "chart": "inflation_expectations_analysis.png", "min_points": 24},
    {"script": "real_m2_growth_tracker.py", "chart": "real_m2_growth_analysis.png", "min_points": 24},
    {"script": "yield_curve_tracker.py", "chart": "yield_curve_analysis.png", "min_points": 24},
    {"script": "shiller_cape_tracker.py", "chart": "shiller_cape_analysis.png", "min_points": 24},
    {"script": "labor_market_tracker.py", "chart": "labor_market_analysis.png", "min_points": 24},
    {"script": "payroll_momentum_tracker.py", "chart": "payroll_momentum_analysis.png", "min_points": 24},
    {"script": "initial_claims_tracker.py", "chart": "initial_claims_analysis.png", "min_points": 24},
    {"script": "credit_conditions_tracker.py", "chart": "credit_conditions_analysis.png", "min_points": 24},
    {"script": "high_yield_oas_tracker.py", "chart": "high_yield_oas_analysis.png", "min_points": 24},
    {"script": "bank_credit_growth_tracker.py", "chart": "bank_credit_growth_analysis.png", "min_points": 24},
    {"script": "consumer_health_tracker.py", "chart": "consumer_health_analysis.png", "min_points": 24},
    {"script": "household_balance_tracker.py", "chart": "household_balance_analysis.png", "min_points": 12},
    {"script": "corporate_earnings_tracker.py", "chart": "corporate_earnings_analysis.png", "min_points": 8},
    {"script": "international_trade_tracker.py", "chart": "international_trade_analysis.png", "min_points": 24},
    {"script": "housing_affordability_tracker.py", "chart": "housing_affordability_analysis.png", "min_points": 24},
    {"script": "housing_starts_tracker.py", "chart": "housing_starts_analysis.png", "min_points": 24},
    {"script": "manufacturing_tracker.py", "chart": "manufacturing_analysis.png", "min_points": 24},
    {"script": "shipping_tracker_complete.py", "chart": "primary_shipping_tracker.png", "min_points": 24},
]

STATUS_COLORS = {
    "Elevated Risk": "#F5B5B5",
    "Watch Zone": "#FFD8A8",
    "Normal": "#B0D8FF",
    "Favorable": "#B8E6C1",
}


def default_start_20y() -> str:
    today = date.today()
    try:
        return today.replace(year=today.year - 20).strftime("%Y-%m-%d")
    except ValueError:
        return today.replace(month=2, day=28, year=today.year - 20).strftime("%Y-%m-%d")


def run_tracker(script: str, start: str, end: str, min_points: int) -> str:
    cmd = [
        sys.executable,
        script,
        "--start",
        start,
        "--end",
        end,
        "--min-data-points",
        str(min_points),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    if result.returncode != 0:
        raise RuntimeError(f"Tracker failed: {script}\n{result.stderr.strip()}")
    return result.stdout


def extract_tracker_status(stdout_text: str, script: str) -> str:
    match = re.search(r"Current status:\s*(.+)", stdout_text)
    if not match:
        raise RuntimeError(f"Tracker did not report current status: {script}")
    status = match.group(1).strip()
    if status not in STATUS_COLORS:
        raise RuntimeError(f"Tracker returned unknown status '{status}': {script}")
    return status


def get_tracker_description(script: str) -> str:
    cmd = [sys.executable, script, "--describe"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    if result.returncode != 0:
        raise RuntimeError(f"Description fetch failed: {script}\n{result.stderr.strip()}")
    text = result.stdout.strip()
    if not text:
        raise RuntimeError(f"Tracker did not return description text: {script}")
    return text


def pretty_tracker_name(script_name: str) -> str:
    name = script_name.replace("_tracker_complete.py", "").replace("_tracker.py", "")
    words = [part.capitalize() for part in name.split("_") if part]
    return " ".join(words)


def add_summary_page(pdf: PdfPages, rows: list[dict[str, str]], start: str, end: str) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")

    fig.suptitle("Executive Summary Dashboard", fontsize=16, fontweight="bold", y=0.97)
    ax.text(0.5, 0.925, f"Status Snapshot ({start} to {end})", ha="center", va="center", fontsize=10)

    total = len(rows)
    cols = 3
    if total > 12:
        cols = 4
    if total > 24:
        cols = 5

    rows_count = (len(rows) + cols - 1) // cols
    left_margin = 0.035
    right_margin = 0.035
    x_gap = 0.015
    usable_w = 1.0 - left_margin - right_margin - (cols - 1) * x_gap
    card_w = usable_w / cols

    y_start = 0.865
    y_gap = 0.01
    available_h = 0.72
    card_h = (available_h - y_gap * max(rows_count - 1, 0)) / max(rows_count, 1)

    name_fs = 8
    status_fs = 9
    if rows_count >= 6:
        name_fs = 7
        status_fs = 8
    if rows_count >= 8:
        name_fs = 6
        status_fs = 7

    for idx, row in enumerate(rows):
        c = idx % cols
        r = idx // cols
        x = left_margin + c * (card_w + x_gap)
        y = y_start - r * (card_h + y_gap) - card_h

        rect = patches.FancyBboxPatch(
            (x, y),
            card_w,
            card_h,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            facecolor=STATUS_COLORS[row["status"]],
            edgecolor="#444444",
            linewidth=0.7,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        wrapped_name = textwrap.fill(row["display_name"], width=16)
        ax.text(x + 0.008, y + card_h - 0.012, wrapped_name, transform=ax.transAxes, ha="left", va="top", fontsize=name_fs, fontweight="bold")
        ax.text(x + 0.008, y + 0.009, row["status"], transform=ax.transAxes, ha="left", va="bottom", fontsize=status_fs)

    legend_y = 0.03
    legend_items = ["Elevated Risk", "Watch Zone", "Normal", "Favorable"]
    for idx, label in enumerate(legend_items):
        x = 0.05 + idx * 0.235
        ax.add_patch(patches.Rectangle((x, legend_y), 0.024, 0.018, transform=ax.transAxes, facecolor=STATUS_COLORS[label], edgecolor="#555555", linewidth=0.7))
        ax.text(x + 0.032, legend_y + 0.009, label, transform=ax.transAxes, va="center", ha="left", fontsize=8)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf(rows: list[dict[str, str]], output_pdf: str, start: str, end: str) -> None:
    with PdfPages(output_pdf) as pdf:
        add_summary_page(pdf, rows, start, end)
        for row in rows:
            fig = plt.figure(figsize=(11, 8.5))
            gs = fig.add_gridspec(2, 1, height_ratios=[6.0, 1.4], hspace=0.08)

            ax_img = fig.add_subplot(gs[0, 0])
            image = plt.imread(row["chart"])
            ax_img.imshow(image)
            ax_img.axis("off")
            ax_img.set_title(row["script"], loc="left", fontsize=10, fontweight="bold")

            ax_txt = fig.add_subplot(gs[1, 0])
            ax_txt.axis("off")
            wrapped = textwrap.fill(row["description"], width=140)
            ax_txt.text(0.0, 0.95, wrapped, va="top", ha="left", fontsize=9)

            fig.suptitle(
                f"Tracker Comparison ({start} to {end})",
                fontsize=14,
                fontweight="bold",
                y=0.98,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trackers on one interval and build aligned comparison PDF")
    parser.add_argument("--start", type=str, default=default_start_20y(), help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=date.today().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    parser.add_argument("--output", type=str, default="tracker_comparison_aligned.pdf", help="Output PDF filename")
    args = parser.parse_args()

    present = [tracker for tracker in TRACKERS if os.path.exists(tracker["script"])]
    if not present:
        raise RuntimeError("No tracker scripts found.")

    rows: list[dict[str, str]] = []
    for tracker in present:
        stdout_text = run_tracker(tracker["script"], args.start, args.end, tracker["min_points"])
        if not os.path.exists(tracker["chart"]):
            raise RuntimeError(f"Expected chart not found after run: {tracker['chart']}")
        description = get_tracker_description(tracker["script"])
        status = extract_tracker_status(stdout_text, tracker["script"])
        rows.append({
            "script": tracker["script"],
            "display_name": pretty_tracker_name(tracker["script"]),
            "chart": tracker["chart"],
            "description": description,
            "status": status,
        })

    build_pdf(rows, args.output, args.start, args.end)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
