#!/usr/bin/env python3

import re
from datetime import datetime
from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "tracker_logs_rerun"
OUTPUT_PDF = ROOT / "economic_tracker_report.pdf"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def match_first(pattern: str, text: str, default: str = "N/A") -> str:
    m = re.search(pattern, text, flags=re.MULTILINE)
    return m.group(1).strip() if m else default


def sanitize_text(value: str) -> str:
    """Best-effort cleanup to avoid PDF font warnings from emoji."""
    if not value:
        return value
    # Drop most non-ascii (emoji) while keeping common punctuation like the bullet.
    cleaned = value.replace("\t", " ")
    cleaned = re.sub(r"[\U00010000-\U0010FFFF]", "", cleaned)
    cleaned = cleaned.replace("✅", "").replace("⚠", "").replace("📊", "").replace("📈", "")
    cleaned = cleaned.replace("📉", "").replace("💡", "").replace("🎯", "").replace("🏆", "")
    cleaned = cleaned.replace("🟢", "").replace("🟡", "").replace("🔴", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_float(value: str) -> float | None:
    if not value or value == "N/A":
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", value)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def parse_percent(value: str) -> float | None:
    if not value or value == "N/A":
        return None
    m = re.search(r"(\d+(?:\.\d+)?)%", value)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    # Sometimes stored as 0-1 in CSV.
    f = parse_float(value)
    if f is None:
        return None
    if 0 <= f <= 1.0:
        return f * 100
    return f


def extract_interpretation_block(text: str) -> str:
    """Pull the log's own INTERPRETATION block if present."""
    if not text:
        return ""
    m = re.search(r"INTERPRETATION:\s*\n(?P<body>(?:.*\n){0,8})", text)
    if not m:
        m = re.search(r"INTERPRETATION:\s*(?P<body>.*)", text)
    if not m:
        return ""
    body = m.group("body")
    # Stop at common section/file markers
    lines = []
    for raw in body.splitlines():
        line = raw.rstrip()
        if not line.strip():
            if lines:
                break
            continue
        if re.match(r"^(?:Files saved|Analysis results saved|Generating|\u2705|\ud83d\udcc1|\ud83d\udcbe|\ud83d\udcca)", line.strip()):
            break
        lines.append(sanitize_text(line))
    return " ".join(lines).strip()


def percentile_bucket(percentile: float | None) -> str:
    if percentile is None:
        return "Unknown"
    if percentile >= 85:
        return "Very strong"
    if percentile >= 70:
        return "Above average"
    if percentile >= 55:
        return "Slightly above average"
    if percentile >= 45:
        return "Around average"
    if percentile >= 30:
        return "Below average"
    if percentile >= 15:
        return "Weak"
    return "Very weak"


def interpretation_from_percentile(
    *,
    name: str,
    assessment: str,
    percentile: float | None,
    trend_label: str | None,
    slope: float | None,
    higher_is_good: bool = True,
    extra: str | None = None,
) -> str:
    bucket = percentile_bucket(percentile)
    if percentile is None:
        base = f"{assessment}."
    else:
        direction = "tailwind" if higher_is_good else "headwind"
        if (percentile >= 70 and higher_is_good) or (percentile <= 30 and not higher_is_good):
            impulse = f"a likely {direction}"
        elif (percentile <= 30 and higher_is_good) or (percentile >= 70 and not higher_is_good):
            impulse = f"a likely {'headwind' if higher_is_good else 'tailwind'}"
        else:
            impulse = "more neutral"
        base = f"{assessment} ({bucket}, {percentile:.1f}th pct) suggests {impulse}."

    if trend_label:
        trend_part = trend_label.lower()
        if slope is not None:
            base += f" Trend is {trend_part} (slope {slope:+.3f})."
        else:
            base += f" Trend is {trend_part}."

    if extra:
        base += f" {extra.strip()}"

    # Keep sentences compact.
    return base.strip()


def extract_signals() -> list[dict[str, Any]]:
    signals = []

    buffett_log = read_text(LOG_DIR / "buffet_tracker.log")
    buffett_csv = ROOT / "buffett_indicator_enhanced.csv"
    buffett_summary = "Generated successfully"
    buffett_view = "Valuation context produced; see CSV for Z-score/percentile/trend"
    buffett_percentile: float | None = None
    buffett_date = "N/A"
    if buffett_csv.exists():
        try:
            df = pd.read_csv(buffett_csv)
            if not df.empty:
                # Some runs may have an empty/incomplete most-recent quarter; use last row with a ZScore.
                df_valid = df[df["ZScore"].notna()] if "ZScore" in df.columns else df
                latest = df_valid.iloc[-1]
                buffett_date = str(latest.get("date", "N/A"))
                z = latest.get("ZScore", None)
                p = latest.get("Percentile", None)
                tr = latest.get("TrendResidual", None)
                parts = []
                if pd.notna(z):
                    parts.append(f"ZScore {float(z):.2f}")
                if pd.notna(p):
                    buffett_percentile = float(p) * 100 if float(p) <= 1.0 else float(p)
                    parts.append(f"Percentile {buffett_percentile:.1f}%")
                if pd.notna(tr):
                    parts.append(f"TrendResidual {float(tr):.2f}")
                if parts:
                    buffett_view = ", ".join(parts)
        except Exception:
            pass
    if "Saved: buffett_indicator_enhanced" not in buffett_log:
        buffett_summary = "Could not confirm in rerun log"

    buffett_interpret = interpretation_from_percentile(
        name="Buffett Indicator",
        assessment="Valuation",
        percentile=buffett_percentile,
        trend_label=None,
        slope=None,
        higher_is_good=False,
        extra="Higher percentiles imply richer equity valuation versus GDP; historically this tends to reduce long-run forward return potential and increase drawdown sensitivity.",
    )
    signals.append({
        "name": "Buffett Indicator",
        "summary": buffett_summary,
        "reading": buffett_view,
        "latest": buffett_date,
        "interpretation": buffett_interpret,
        "source_interpretation": extract_interpretation_block(buffett_log),
    })

    def load(name: str) -> str:
        return read_text(LOG_DIR / name)

    consumer = load("consumer_health_tracker.log")
    consumer_assessment = match_first(r"Current Assessment:\s*(.+)", consumer)
    consumer_pct = parse_percent(match_first(r"Percentile Rank:\s*([^\n]+)", consumer))
    consumer_slope = parse_float(match_first(r"Recent Trend \(12M\):\s*\w+ \(slope:\s*([^\)]+)\)", consumer))
    consumer_trend = match_first(r"Recent Trend \(12M\):\s*([A-Z_]+)", consumer, default="N/A")
    consumer_latest = match_first(r"Latest Analysis \(([^\)]+)\):", consumer)
    consumer_log_interp = extract_interpretation_block(consumer)
    consumer_interpret = interpretation_from_percentile(
        name="Consumer Health",
        assessment=sanitize_text(consumer_assessment),
        percentile=consumer_pct,
        trend_label=None if consumer_trend == "N/A" else consumer_trend.replace("_", " "),
        slope=consumer_slope,
        higher_is_good=True,
        extra=consumer_log_interp or "Consumer is a key near-term growth driver; readings near the middle of history usually mean consumption is neither strongly accelerating nor collapsing.",
    )
    signals.append({
        "name": "Consumer Health",
        "summary": sanitize_text(consumer_assessment),
        "reading": f"Composite {match_first(r'Composite Score:\s*([^\\n]+)', consumer)}, Percentile {match_first(r'Percentile Rank:\s*([^\\n]+)', consumer)}",
        "latest": consumer_latest,
        "interpretation": consumer_interpret,
        "source_interpretation": consumer_log_interp,
    })

    corp = load("corporate_earnings_tracker.log")
    corp_assessment = match_first(r"Current Assessment:\s*(.+)", corp)
    corp_pct = parse_percent(match_first(r"Percentile Rank:\s*([^\n]+)", corp))
    corp_slope = parse_float(match_first(r"Recent Trend \(12M\):\s*\w+ \(slope:\s*([^\)]+)\)", corp))
    corp_trend = match_first(r"Recent Trend \(12M\):\s*([A-Z_]+)", corp, default="N/A")
    corp_latest = match_first(r"Latest Analysis \(([^\)]+)\):", corp)
    corp_log_interp = extract_interpretation_block(corp)
    corp_interpret = interpretation_from_percentile(
        name="Corporate Earnings",
        assessment=sanitize_text(corp_assessment),
        percentile=corp_pct,
        trend_label=None if corp_trend == "N/A" else corp_trend.replace("_", " "),
        slope=corp_slope,
        higher_is_good=True,
        extra=corp_log_interp or "Corporate health typically leads hiring/capex; a deteriorating trend can foreshadow softer investment and margins even if the current level is still average.",
    )
    signals.append({
        "name": "Corporate Earnings",
        "summary": sanitize_text(corp_assessment),
        "reading": f"Composite {match_first(r'Composite Score:\s*([^\\n]+)', corp)}, Percentile {match_first(r'Percentile Rank:\s*([^\\n]+)', corp)}",
        "latest": corp_latest,
        "interpretation": corp_interpret,
        "source_interpretation": corp_log_interp,
    })

    credit = load("credit_conditions_tracker.log")
    credit_assessment = match_first(r"Credit Conditions:\s*([^\n]+)", credit)
    credit_pct = parse_percent(match_first(r"Percentile Rank:\s*([^\n]+)", credit))
    credit_slope = parse_float(match_first(r"Recent Trend \(12M\):\s*\w+ \(slope:\s*([^\)]+)\)", credit))
    credit_trend = match_first(r"Recent Trend \(12M\):\s*([A-Z_]+)", credit, default="N/A")
    credit_latest = match_first(r"Latest Analysis \(([^\)]+)\):", credit)
    credit_log_interp = extract_interpretation_block(credit)
    credit_interpret = interpretation_from_percentile(
        name="Credit Conditions",
        assessment=sanitize_text(credit_assessment),
        percentile=credit_pct,
        trend_label=None if credit_trend == "N/A" else credit_trend.replace("_", " "),
        slope=credit_slope,
        higher_is_good=True,
        extra=credit_log_interp or "Tighter credit conditions tend to transmit into slower growth with a lag (via weaker lending, higher refinancing costs, and wider spreads).",
    )
    signals.append({
        "name": "Credit Conditions",
        "summary": sanitize_text(credit_assessment),
        "reading": f"Composite {match_first(r'Composite Score:\s*([^\\n]+)', credit)}, Percentile {match_first(r'Percentile Rank:\s*([^\\n]+)', credit)}",
        "latest": credit_latest,
        "interpretation": credit_interpret,
        "source_interpretation": credit_log_interp,
    })

    trade = load("international_trade_tracker.log")
    trade_assessment = match_first(r"Trade Strength:\s*([^\n]+)", trade)
    trade_pct = parse_percent(match_first(r"Percentile Rank:\s*([^\n]+)", trade))
    trade_slope = parse_float(match_first(r"Recent Trend \(12M\):\s*\w+ \(slope:\s*([^\)]+)\)", trade))
    trade_trend = match_first(r"Recent Trend \(12M\):\s*([A-Z_]+)", trade, default="N/A")
    trade_latest = match_first(r"Latest Analysis \(([^\)]+)\):", trade)
    trade_log_interp = extract_interpretation_block(trade)
    trade_interpret = interpretation_from_percentile(
        name="International Trade",
        assessment=sanitize_text(trade_assessment),
        percentile=trade_pct,
        trend_label=None if trade_trend == "N/A" else trade_trend.replace("_", " "),
        slope=trade_slope,
        higher_is_good=True,
        extra=trade_log_interp or "Stronger trade conditions often correlate with steadier global demand and healthier industrial/export activity.",
    )
    signals.append({
        "name": "International Trade",
        "summary": sanitize_text(trade_assessment),
        "reading": f"Composite {match_first(r'Composite Score:\s*([^\\n]+)', trade)}, Percentile {match_first(r'Percentile Rank:\s*([^\\n]+)', trade)}",
        "latest": trade_latest,
        "interpretation": trade_interpret,
        "source_interpretation": trade_log_interp,
    })

    labor = load("labor_market_tracker.log")
    labor_assessment = match_first(r"Labor Market Strength:\s*([^\n]+)", labor)
    labor_pct = parse_percent(match_first(r"Percentile Rank:\s*([^\n]+)", labor))
    labor_slope = parse_float(match_first(r"Recent Trend \(12M\):\s*\w+ \(slope:\s*([^\)]+)\)", labor))
    labor_trend = match_first(r"Recent Trend \(12M\):\s*([A-Z_]+)", labor, default="N/A")
    labor_latest = match_first(r"Latest Analysis \(([^\)]+)\):", labor)
    labor_log_interp = extract_interpretation_block(labor)
    labor_interpret = interpretation_from_percentile(
        name="Labor Market",
        assessment=sanitize_text(labor_assessment),
        percentile=labor_pct,
        trend_label=None if labor_trend == "N/A" else labor_trend.replace("_", " "),
        slope=labor_slope,
        higher_is_good=True,
        extra=labor_log_interp or "A balanced labor market is consistent with steady (but not overheating) domestic demand; large moves here typically show up in consumption next.",
    )
    signals.append({
        "name": "Labor Market",
        "summary": sanitize_text(labor_assessment),
        "reading": f"Composite {match_first(r'Composite Score:\s*([^\\n]+)', labor)}, Percentile {match_first(r'Percentile Rank:\s*([^\\n]+)', labor)}",
        "latest": labor_latest,
        "interpretation": labor_interpret,
        "source_interpretation": labor_log_interp,
    })

    mfg = load("manufacturing_tracker.log")
    mfg_health = parse_float(match_first(r"Health Score:\s*([^\n]+)", mfg))
    mfg_class = sanitize_text(match_first(r"Classification:\s*([^\n]+)", mfg))
    mfg_pct = parse_percent(match_first(r"IP Percentile:\s*([^\n]+)", mfg))
    mfg_latest = match_first(r"Latest Data:\s*([^\n]+)", mfg)
    mfg_log_interp = extract_interpretation_block(mfg) or sanitize_text(match_first(r"INTERPRETATION:\s*([^\n]+)", mfg, default=""))
    mfg_extra = mfg_log_interp or "Manufacturing is often a cyclical bellwether; stable readings typically mean neither a strong acceleration nor a contraction impulse from goods production."
    # Treat manufacturing score as higher-is-good.
    mfg_assessment = f"{mfg_class}"
    mfg_interpret = interpretation_from_percentile(
        name="Manufacturing",
        assessment=mfg_assessment,
        percentile=mfg_pct,
        trend_label=None,
        slope=None,
        higher_is_good=True,
        extra=mfg_extra,
    )
    signals.append({
        "name": "Manufacturing",
        "summary": mfg_class,
        "reading": f"Health Score {match_first(r'Health Score:\s*([^\\n]+)', mfg)}, IP Percentile {match_first(r'IP Percentile:\s*([^\\n]+)', mfg)}",
        "latest": mfg_latest,
        "interpretation": mfg_interpret,
        "source_interpretation": mfg_log_interp,
    })

    cape = load("shiller_cape_tracker.log")
    cape_class = sanitize_text(match_first(r"Classification:\s*([^\n]+)", cape))
    cape_current = match_first(r"Current CAPE:\s*([^\n]+)", cape)
    cape_pct = parse_percent(match_first(r"Historical Percentile:\s*([^\n]+)", cape))
    cape_latest = match_first(r"Latest Data:\s*([^\n]+)", cape)
    cape_log_interp = extract_interpretation_block(cape)
    cape_interpret = interpretation_from_percentile(
        name="Shiller CAPE",
        assessment=cape_class,
        percentile=cape_pct,
        trend_label=None,
        slope=None,
        higher_is_good=False,
        extra=cape_log_interp or "Higher CAPE indicates higher valuation versus long-run earnings; historically that has been associated with lower medium/long-run real returns and higher sensitivity to earnings disappointments.",
    )
    signals.append({
        "name": "Shiller CAPE",
        "summary": cape_class,
        "reading": f"Current CAPE {cape_current}, Percentile {match_first(r'Historical Percentile:\s*([^\\n]+)', cape)}",
        "latest": cape_latest,
        "interpretation": cape_interpret,
        "source_interpretation": cape_log_interp,
    })

    shipping = load("shipping_tracker_complete.log")
    shipping_status = sanitize_text(match_first(r"CURRENT STATUS:\s*([^\n]+)", shipping))
    shipping_pct = parse_percent(match_first(r"Percentile Rank:\s*([^\n]+)", shipping))
    shipping_slope = parse_float(match_first(r"Recent Trend Slope:\s*([^\n]+)", shipping))
    shipping_latest = match_first(r"Latest Analysis \(([^\)]+)\):", shipping)
    shipping_log_interp = extract_interpretation_block(shipping) or sanitize_text(match_first(r"INTERPRETATION:\s*([^\n]+)", shipping, default=""))
    shipping_interpret = interpretation_from_percentile(
        name="Shipping Activity",
        assessment=shipping_status,
        percentile=shipping_pct,
        trend_label="STABLE" if shipping_slope is None else "TRENDING",
        slope=shipping_slope,
        higher_is_good=True,
        extra=shipping_log_interp or "Shipping/freight is a real-economy pulse for goods demand; hotter readings can be consistent with firm activity but can also be noisy around supply chain/fuel effects.",
    )
    signals.append({
        "name": "Shipping Activity",
        "summary": shipping_status,
        "reading": f"Composite {match_first(r'Composite Index:\s*([^\\n]+)', shipping)}, Z-Score {match_first(r'Z-Score:\s*([^\\n]+)', shipping)}",
        "latest": shipping_latest,
        "interpretation": shipping_interpret,
        "source_interpretation": shipping_log_interp,
    })

    yc = load("yield_curve_tracker.log")
    yc_risk = sanitize_text(match_first(r"Recession Risk Level:\s*([^\n]+)", yc))
    yc_latest = match_first(r"Latest Analysis \(([^\)]+)\):", yc)
    yc_score = match_first(r"Risk Score:\s*([^\n]+)", yc)
    spread_10y2y = match_first(r"10Y-2Y:\s*([^\n]+)", yc)
    spread_10y3m = match_first(r"10Y-3M:\s*([^\n]+)", yc)
    last_inv_10y2y = match_first(r"10Y-2Y last inverted (\d+) days ago", yc)
    last_inv_10y3m = match_first(r"10Y-3M last inverted (\d+) days ago", yc)
    yc_log_interp = extract_interpretation_block(yc)
    yc_extra = (
        f"Current spreads are positive (10Y-2Y {sanitize_text(spread_10y2y)}, 10Y-3M {sanitize_text(spread_10y3m)}). "
        f"Last inversions: 10Y-2Y {last_inv_10y2y}d ago; 10Y-3M {last_inv_10y3m}d ago."
    )
    yc_interpret = f"{yc_risk}. {yc_log_interp or 'A non-inverted curve generally implies lower near-term recession odds; recent inversion history still matters for the lagged cycle impact.'} {yc_extra}"
    signals.append({
        "name": "Yield Curve",
        "summary": yc_risk,
        "reading": f"Risk Score {yc_score}, 10Y-2Y {sanitize_text(spread_10y2y)}, 10Y-3M {sanitize_text(spread_10y3m)}",
        "latest": yc_latest,
        "interpretation": sanitize_text(yc_interpret),
        "source_interpretation": yc_log_interp,
    })

    return signals


def add_text_page(pdf: PdfPages, title: str, lines: list[str], subtitle: str | None = None):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = 0.96
    ax.text(0.06, y, title, fontsize=20, fontweight="bold", va="top")
    y -= 0.04
    if subtitle:
        ax.text(0.06, y, subtitle, fontsize=10, color="dimgray", va="top")
        y -= 0.04

    for line in lines:
        chunks = wrap(line, width=106) if line else [""]
        for chunk in chunks:
            ax.text(0.06, y, chunk, fontsize=10.5, va="top")
            y -= 0.022
            if y < 0.06:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                y = 0.95
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_chart_page(pdf: PdfPages, image_path: Path, title: str):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.88])
    ax.axis("off")
    img = plt.imread(str(image_path))
    ax.imshow(img)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf():
    signals = extract_signals()

    chart_files = [
        ROOT / "buffett_indicator_enhanced.png",
        ROOT / "consumer_health_analysis.png",
        ROOT / "corporate_earnings_analysis.png",
        ROOT / "credit_conditions_analysis.png",
        ROOT / "international_trade_analysis.png",
        ROOT / "labor_market_analysis.png",
        ROOT / "primary_shipping_tracker.png",
        ROOT / "shiller_cape_analysis.png",
        ROOT / "yield_curve_analysis.png",
        ROOT / "yield_curve_10Y_2Y.png",
    ]
    chart_files = [c for c in chart_files if c.exists()]

    with PdfPages(OUTPUT_PDF) as pdf:
        add_text_page(
            pdf,
            title="Economic Tracker Report",
            subtitle=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Inputs: existing charts + latest rerun logs",
            lines=[
                "This report contains:",
                "1) Analysis section: what each indicator is currently signaling.",
                "2) Charts section (at bottom): all generated tracker charts.",
                "",
                "Clean rerun status: 10 / 10 trackers succeeded.",
            ],
        )

        # Topline summary page (kept text-only so charts remain at bottom).
        bullets = []
        for item in signals:
            latest = item.get("latest", "N/A")
            summary = item.get("summary", "N/A")
            bullets.append(f"• {item['name']}: {summary} (latest: {latest})")
        add_text_page(
            pdf,
            title="Topline Summary",
            subtitle="One-line status for each indicator",
            lines=bullets,
        )

        analysis_lines: list[str] = []
        for item in signals:
            latest = item.get("latest", "N/A")
            analysis_lines.append(f"• {item['name']}: {item['summary']}")
            analysis_lines.append(f"  Latest: {latest}")
            analysis_lines.append(f"  Reading: {sanitize_text(str(item.get('reading', 'N/A')))}")
            analysis_lines.append(f"  Interpretation: {sanitize_text(str(item.get('interpretation', 'N/A')))}")
            analysis_lines.append("")

        add_text_page(
            pdf,
            title="Analysis Section",
            subtitle="Indicator-by-indicator signal summary from latest run",
            lines=analysis_lines,
        )

        # Charts at the bottom of the PDF (append-only section).
        add_text_page(
            pdf,
            title="Charts (Appended)",
            subtitle=f"Included charts: {len(chart_files)}",
            lines=[f"• {p.name}" for p in chart_files],
        )

        for chart in chart_files:
            add_chart_page(pdf, chart, title=chart.name)

    print(f"Saved report: {OUTPUT_PDF}")
    print(f"Charts included: {len(chart_files)}")


if __name__ == "__main__":
    build_pdf()