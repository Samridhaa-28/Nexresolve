import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "evaluation" / "reports"
REPORT_JSON = REPORT_DIR / "summarizer_eval.json"
REPORT_TXT = REPORT_DIR / "summarizer_eval.txt"


def evaluate_summaries(issues_path: str = "data/final/cleaned_issues.csv") -> Dict[str, Any]:
    if not os.path.exists(issues_path):
        print(f"Error: CSV not found at {issues_path}")
        return {}

    df = pd.read_csv(issues_path)

    if "summary" not in df.columns:
        print("Error: Summary column missing. Please run nlp/summarizer.py --mode generate first.")
        return {}

    # Extract clean text
    df["full_text"] = df.get("clean_title", pd.Series([""] * len(df))).fillna("") + " " + df.get("clean_body", pd.Series([""] * len(df))).fillna("")
    
    # Fill NAs
    df["summary"] = df["summary"].fillna("")
    df["full_text"] = df["full_text"].str.strip()
    df["summary"] = df["summary"].str.strip()

    # Compute lengths
    df["orig_len"] = df["full_text"].apply(lambda x: len(x))
    df["summ_len"] = df["summary"].apply(lambda x: len(x))
    df["orig_words"] = df["full_text"].apply(lambda x: len(x.split()))
    df["summ_words"] = df["summary"].apply(lambda x: len(x.split()))

    # Basic stats
    avg_orig_len = df["orig_len"].mean()
    avg_summ_len = df["summ_len"].mean()
    avg_orig_words = df["orig_words"].mean()
    avg_summ_words = df["summ_words"].mean()
    
    compression_ratio_char = df["summ_len"].sum() / max(1, df["orig_len"].sum())
    compression_ratio_words = df["summ_words"].sum() / max(1, df["orig_words"].sum())

    # Quality control
    too_short = df[df["summ_len"] < 10].shape[0]
    
    identical_mask = df["summary"] == df["full_text"]
    identical_total = df[identical_mask].shape[0]
    
    # Identify identical due to short source text (less than 30 words)
    identical_short = df[identical_mask & (df["orig_words"] <= 30)].shape[0]
    identical_long = identical_total - identical_short
    
    empty = df[df["summ_len"] == 0].shape[0]

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_analyzed": len(df),
        "averages": {
            "original_length_chars": round(avg_orig_len, 2),
            "summary_length_chars": round(avg_summ_len, 2),
            "original_words": round(avg_orig_words, 2),
            "summary_words": round(avg_summ_words, 2),
            "compression_ratio_chars": round(compression_ratio_char, 4),
            "compression_ratio_words": round(compression_ratio_words, 4),
        },
        "quality_flags": {
            "too_short_less_than_10_chars": too_short,
            "identical_total": identical_total,
            "identical_due_to_short_text": identical_short,
            "identical_algorithm_fallback": identical_long,
            "empty_summaries": empty,
        }
    }
    return report


def print_report(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 50)
    lines.append(" NEXRESOLVE SUMMARIZER EVALUATION REPORT")
    lines.append("=" * 50)
    lines.append(f"Generated At: {report.get('generated_at')}")
    lines.append(f"Total Tickets Analyzed: {report.get('total_analyzed')}")
    lines.append("")
    
    lines.append("--- Averages & Compression ---")
    avg = report.get("averages", {})
    lines.append(f"  Original Length (Chars): {avg.get('original_length_chars')}")
    lines.append(f"  Summary Length (Chars):  {avg.get('summary_length_chars')}")
    lines.append(f"  Compression Ratio:       {avg.get('compression_ratio_chars')}")
    lines.append("")

    lines.append("--- Quality Control Flags ---")
    flags = report.get("quality_flags", {})
    lines.append(f"  Too Short Less Than 10 Chars: {flags.get('too_short_less_than_10_chars')}")
    lines.append(f"  Empty Summaries:              {flags.get('empty_summaries')}")
    lines.append(f"  Identical Total:              {flags.get('identical_total')}")
    lines.append(f"    -> Due to Short Source Text:{flags.get('identical_due_to_short_text')}")
    lines.append(f"    -> Algorithm Fallbacks:     {flags.get('identical_algorithm_fallback')}")
    lines.append("")

    lines.append("=" * 50)
    return "\n".join(lines)


def run(issues_path: str = "data/final/cleaned_issues.csv") -> None:
    print("Generating summarizer evaluation report...")
    report = evaluate_summaries(issues_path)

    if not report:
        return

    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    txt_content = print_report(report)
    print(txt_content)

    with open(REPORT_TXT, "w", encoding="utf-8") as fh:
        fh.write(txt_content)

    print(f"\nReports saved:")
    print(f"  {REPORT_JSON}")
    print(f"  {REPORT_TXT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate output of Summarizer")
    parser.add_argument("--issues_path", default="data/final/cleaned_issues.csv")
    args = parser.parse_args()
    
    run(issues_path=args.issues_path)
