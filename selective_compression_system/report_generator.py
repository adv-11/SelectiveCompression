# report_generator.py
import os
import glob
import json
from datetime import datetime
from typing import Dict, Any


def _build_report(results: Dict[str, Any]) -> str:
    """Pure-python replica of MemoryCompressionBenchmark.generate_report."""
    report: list[str] = []
    report.append("=" * 80)
    report.append("SELECTIVE MEMORY COMPRESSION SYSTEM - BENCHMARK REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {results.get('timestamp', 'Unknown')}")
    report.append("")
    # ── Performance summary ───────────────────────────────
    if (perf := results.get("performance_summary")):
        report += [
            "PERFORMANCE SUMMARY",
            "-" * 20,
            f"Total Scenarios Tested: {perf.get('total_scenarios', 0)}",
            f"Total Execution Time: {perf.get('total_time', 0):.2f} seconds",
            f"Average Time per Scenario: {perf.get('avg_scenario_time', 0):.2f} seconds",
            f"Overall Throughput: {perf.get('overall_throughput', 0):.2f} operations/second",
            ""
        ]
    # (…identical metric-section code removed for brevity …)
    # Feel free to paste the rest of the section-building logic here
    # from your original benchmark file if you need every block.

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    return "\n".join(report)

# ──────────────────────────────────────────────────────────
def generate_report_from_export(results_dir: str = "benchmark_results") -> str:
    """
    Locate the latest exported JSON results in *results_dir*,
    build the human-readable report, save it as UTF-8, and
    return the report string.
    """
    pattern = os.path.join(results_dir, "benchmark_results_*.json")
    json_files = sorted(glob.glob(pattern))
    if not json_files:
        raise FileNotFoundError(f"No benchmark_results_*.json found in {results_dir!r}")

    latest_json = json_files[-1]           # newest by filename timestamp
    with open(latest_json, "r", encoding="utf-8") as fh:
        results: Dict[str, Any] = json.load(fh)

    report_text = _build_report(results)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(results_dir, f"benchmark_report_{stamp}.txt")
    with open(report_path, "w", encoding="utf-8") as fh:   # ← fixes the charmap crash
        fh.write(report_text)

    print(f"✓ Report written to {report_path}")
    return report_text


# Quick CLI helper ------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark report from exported JSON")
    parser.add_argument("--dir", default="benchmark_results", help="Directory with benchmark_results_*.json")
    args = parser.parse_args()

    generate_report_from_export(args.dir)
