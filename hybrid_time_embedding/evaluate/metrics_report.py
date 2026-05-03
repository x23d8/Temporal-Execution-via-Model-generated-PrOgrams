"""
metrics_report.py — Generates human-readable + machine-readable evaluation reports.
Prints a formatted table to stdout and saves JSON + CSV artifacts.
"""

import csv
import json
import os
from datetime import datetime
from typing import Dict


def generate_report(
    results: Dict,
    output_dir: str,
    phase: str = "test",
    save: bool = True,
) -> str:
    """
    Print and optionally save evaluation results.

    Args:
        results: Dict from TemporalEvaluator.evaluate() — keys: arithmetic, duration, overall.
        output_dir: Directory to save JSON and CSV files.
        phase: Label for the report (e.g. "phase2", "test").
        save: Whether to write files to disk.

    Returns:
        Formatted report string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = [f"\n{'='*60}", f"  Evaluation Report — {phase.upper()} — {timestamp}", f"{'='*60}"]

    for subtask in ["arithmetic", "duration", "overall"]:
        m = results.get(subtask, {})
        if not m:
            continue
        lines.append(f"\n  [{subtask.upper()}]")
        lines.append(f"    MAE:              {m.get('mae', 0):.4f}")
        lines.append(f"    Exact Match:      {m.get('exact_match', 0)*100:.2f}%")
        lines.append(f"    Within 1yr:       {m.get('within_1yr', 0)*100:.2f}%")
        lines.append(f"    Within 5yr:       {m.get('within_5yr', 0)*100:.2f}%")
        lines.append(f"    Consistency Rate: {m.get('consistency_rate', 0)*100:.2f}%")
        lines.append(f"    Median AE:        {m.get('median_ae', 0):.4f}")
        lines.append(f"    P90 AE:           {m.get('p90_ae', 0):.4f}")

    lines.append(f"\n{'='*60}\n")
    report = "\n".join(lines)
    print(report)

    if save:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"results_{phase}_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump({k: v for k, v in results.items() if k != "raw"}, f, indent=2)

        raw = results.get("raw", {})
        csv_path = os.path.join(output_dir, f"predictions_{phase}_{timestamp}.csv")
        if raw:
            n = len(raw.get("arith_preds", []))
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "idx", "arith_pred", "arith_truth", "dur_pred", "dur_truth", "start_time"
                ])
                writer.writeheader()
                for i in range(n):
                    writer.writerow({
                        "idx": i,
                        "arith_pred": raw["arith_preds"][i],
                        "arith_truth": raw["arith_truths"][i],
                        "dur_pred": raw["dur_preds"][i],
                        "dur_truth": raw["dur_truths"][i],
                        "start_time": raw["start_times"][i],
                    })
        print(f"Saved: {json_path}")
        if raw:
            print(f"Saved: {csv_path}")

    return report
