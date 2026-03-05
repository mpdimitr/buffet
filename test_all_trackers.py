#!/usr/bin/env python3
"""Run all present trackers on one shared interval and build a vertically aligned comparison PDF."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from datetime import date

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


TRACKERS = [
    {"script": "buffet_tracker.py", "chart": "buffett_indicator_enhanced.png", "min_points": 8},
    {"script": "yield_curve_tracker.py", "chart": "yield_curve_analysis.png", "min_points": 24},
    {"script": "shiller_cape_tracker.py", "chart": "shiller_cape_analysis.png", "min_points": 24},
    {"script": "labor_market_tracker.py", "chart": "labor_market_analysis.png", "min_points": 24},
    {"script": "credit_conditions_tracker.py", "chart": "credit_conditions_analysis.png", "min_points": 24},
    {"script": "consumer_health_tracker.py", "chart": "consumer_health_analysis.png", "min_points": 24},
    {"script": "corporate_earnings_tracker.py", "chart": "corporate_earnings_analysis.png", "min_points": 8},
    {"script": "international_trade_tracker.py", "chart": "international_trade_analysis.png", "min_points": 24},
    {"script": "manufacturing_tracker.py", "chart": "manufacturing_analysis.png", "min_points": 24},
    {"script": "shipping_tracker_complete.py", "chart": "primary_shipping_tracker.png", "min_points": 24},
]


def default_start_20y() -> str:
    today = date.today()
    try:
        return today.replace(year=today.year - 20).strftime("%Y-%m-%d")
    except ValueError:
        return today.replace(month=2, day=28, year=today.year - 20).strftime("%Y-%m-%d")


def run_tracker(script: str, start: str, end: str, min_points: int) -> None:
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


def get_tracker_description(script: str) -> str:
    cmd = [sys.executable, script, "--describe"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    if result.returncode != 0:
        raise RuntimeError(f"Description fetch failed: {script}\n{result.stderr.strip()}")
    text = result.stdout.strip()
    if not text:
        raise RuntimeError(f"Tracker did not return description text: {script}")
    return text


def build_pdf(rows: list[dict[str, str]], output_pdf: str, start: str, end: str) -> None:
    height = max(3.2 * len(rows), 8)
    with PdfPages(output_pdf) as pdf:
        fig = plt.figure(figsize=(16, height))
        gs = fig.add_gridspec(len(rows), 2, width_ratios=[3.4, 2.0], wspace=0.15, hspace=0.32)

        for idx, row in enumerate(rows):
            ax_img = fig.add_subplot(gs[idx, 0])
            image = plt.imread(row["chart"])
            ax_img.imshow(image)
            ax_img.axis("off")
            ax_img.set_title(row["script"], loc="left", fontsize=10, fontweight="bold")

            ax_txt = fig.add_subplot(gs[idx, 1])
            ax_txt.axis("off")
            wrapped = textwrap.fill(row["description"], width=58)
            ax_txt.text(0.0, 0.98, wrapped, va="top", ha="left", fontsize=9)

        fig.suptitle(
            f"Tracker Comparison ({start} to {end})",
            fontsize=14,
            fontweight="bold",
            y=0.995,
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
        run_tracker(tracker["script"], args.start, args.end, tracker["min_points"])
        if not os.path.exists(tracker["chart"]):
            raise RuntimeError(f"Expected chart not found after run: {tracker['chart']}")
        description = get_tracker_description(tracker["script"])
        rows.append({
            "script": tracker["script"],
            "chart": tracker["chart"],
            "description": description,
        })

    build_pdf(rows, args.output, args.start, args.end)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
