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
REPORT_JSON = REPORT_DIR / "sentiment_eval.json"
REPORT_TXT = REPORT_DIR / "sentiment_eval.txt"


def evaluate_sentiment(issues_path: str = "data/final/cleaned_issues.csv") -> Dict[str, Any]:
    if not os.path.exists(issues_path):
        return {"error": "Issues CSV not found"}

    df = pd.read_csv(issues_path)

    if "sentiment_score" not in df.columns or "frustration_level" not in df.columns:
        print("Error: Sentiment columns missing. Please run nlp/sentiment_analyzer.py --mode generate first.")
        return {}

    # 1. Label distribution
    label_dist = df["sentiment_label"].value_counts(normalize=True).to_dict()
    label_dist = {k: round(v * 100, 2) for k, v in label_dist.items()}

    # 2. Average frustration sliced by urgency if available
    frust_by_urgency = {}
    if "urgent_flag" in df.columns:
        frust_by_urgency = df.groupby("urgent_flag")["frustration_level"].mean().to_dict()
        frust_by_urgency = {str(k): round(float(v), 4) for k, v in frust_by_urgency.items()}

    # 3. Average frustration sliced by intent
    frust_by_intent = {}
    if "intent_group" in df.columns:
        frust_by_intent = df.groupby("intent_group")["frustration_level"].mean().to_dict()
        frust_by_intent = {str(k): round(float(v), 4) for k, v in frust_by_intent.items()}

    # 4. Global averages
    avg_frustration = round(df["frustration_level"].mean(), 4)
    avg_sentiment = round(df["sentiment_score"].mean(), 4)

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_analyzed": len(df),
        "global_averages": {
            "average_frustration": avg_frustration,
            "average_sentiment": avg_sentiment,
        },
        "label_distribution_pct": label_dist,
        "frustration_by_urgency": frust_by_urgency,
        "frustration_by_intent": frust_by_intent,
    }
    return report


def print_report(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 50)
    lines.append(" NEXRESOLVE SENTIMENT EVALUATION REPORT")
    lines.append("=" * 50)
    lines.append(f"Generated At: {report.get('generated_at')}")
    lines.append(f"Total Tickets Analyzed: {report.get('total_analyzed')}")
    lines.append("")
    
    lines.append("--- Global Averages ---")
    glob = report.get("global_averages", {})
    lines.append(f"  Average Sentiment:   {glob.get('average_sentiment')}")
    lines.append(f"  Average Frustration: {glob.get('average_frustration')}")
    lines.append("")

    lines.append("--- Label Distribution ---")
    for label, pct in report.get("label_distribution_pct", {}).items():
        lines.append(f"  {label.capitalize()}: {pct}%")
    lines.append("")

    lines.append("--- Frustration by Urgency ---")
    for urg, val in report.get("frustration_by_urgency", {}).items():
        lines.append(f"  {urg.capitalize()}: {val}")
    lines.append("")

    lines.append("--- Frustration by Intent ---")
    for intent, val in report.get("frustration_by_intent", {}).items():
        lines.append(f"  {intent.capitalize()}: {val}")
    lines.append("")

    lines.append("=" * 50)
    return "\n".join(lines)


def run(issues_path: str = "data/final/cleaned_issues.csv") -> None:
    print("Generating sentiment evaluation report...")
    report = evaluate_sentiment(issues_path)

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
    parser = argparse.ArgumentParser(description="Evaluate output of Sentiment Analyzer")
    parser.add_argument("--issues_path", default="data/final/cleaned_issues.csv")
    args = parser.parse_args()
    
    run(issues_path=args.issues_path)
