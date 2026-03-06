#!/usr/bin/env python3
"""Generate historical dashboard PDFs for pre-recession peaks and recession bottoms."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DashboardEvent:
    name: str
    event_type: str
    event_date: str


EVENTS = [
    DashboardEvent("pre_gulf_war_peak", "pre-recession peak", "1990-07-16"),
    DashboardEvent("recession_bottom_1991", "recession bottom", "1991-03-01"),
    DashboardEvent("pre_dotcom_peak", "pre-recession peak", "2000-03-24"),
    DashboardEvent("recession_bottom_2001", "recession bottom", "2001-11-01"),
    DashboardEvent("pre_gfc_peak", "pre-recession peak", "2007-10-09"),
    DashboardEvent("recession_bottom_2009", "recession bottom", "2009-06-01"),
    DashboardEvent("pre_covid_peak", "pre-recession peak", "2020-02-19"),
    DashboardEvent("recession_bottom_2020", "recession bottom", "2020-04-01"),
]


def subtract_years_safe(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, month=2, day=28)


def build_output_name(event: DashboardEvent) -> str:
    slug = event.name.replace(" ", "_").lower()
    return f"dashboard_{event.event_date}_{slug}.pdf"


def run_event(event: DashboardEvent, output_dir: str) -> str:
    end_day = date.fromisoformat(event.event_date)
    start_day = subtract_years_safe(end_day, 20)
    output_file = os.path.join(output_dir, build_output_name(event))

    cmd = [
        sys.executable,
        "test_all_trackers.py",
        "--start",
        start_day.isoformat(),
        "--end",
        end_day.isoformat(),
        "--output",
        output_file,
        "--allow-partial",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed for {event.event_type} ({event.event_date}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return output_file


def main() -> None:
    output_dir = os.path.join(os.getcwd(), "historical_dashboards")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating historical dashboard reports...")
    generated: list[str] = []
    for event in EVENTS:
        print(f"- {event.event_type.title()} @ {event.event_date}")
        path = run_event(event, output_dir)
        generated.append(path)

    print("\nGenerated reports:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
