import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class SentimentAnalyzer:
    def __init__(self, model: str = "vader"):
        self.model = model
        if self.model == "vader":
            self.analyzer = SentimentIntensityAnalyzer()
        elif self.model == "textblob":
            pass
        else:
            raise ValueError(f"Unknown model: {model}")

    def analyze_text(self, text: str) -> float:
        """Returns a sentiment score between -1.0 and 1.0"""
        if not isinstance(text, str) or not text.strip():
            return 0.0

        if self.model == "vader":
            scores = self.analyzer.polarity_scores(text)
            return scores["compound"]
        
        elif self.model == "textblob":
            blob = TextBlob(text)
            return blob.sentiment.polarity
            
        return 0.0

    def get_label(self, score: float) -> str:
        if score <= -0.05:
            return "negative"
        elif score >= 0.05:
            return "positive"
        return "neutral"

    def compute_frustration(
        self, 
        sentiment_score: float, 
        urgency: str, 
        resolution_time_days: float
    ) -> float:
        """Computes frustration level clamped to [0.0, 1.0]"""
        # Base frustration is reverse of sentiment scaled to 0-1
        # Sentiment -1 -> Frustration 1
        # Sentiment 1  -> Frustration 0
        frustration = (-sentiment_score + 1) / 2.0
        
        if urgency and str(urgency).lower() == "high":
            frustration += 0.2
            
        if pd.notna(resolution_time_days) and resolution_time_days > 7:
            frustration += 0.1
            
        return max(0.0, min(1.0, frustration))


def generate_features(
    issues_path: str = "data/final/cleaned_issues.csv",
    rl_path: str = "data/final/final_rl_dataset.csv",
    model: str = "vader"
) -> None:
    """Generates sentiment features and upates datasets."""
    if not os.path.exists(issues_path) or not os.path.exists(rl_path):
        logging.error("Source CSV files not found.")
        return

    issues_df = pd.read_csv(issues_path)
    rl_df = pd.read_csv(rl_path)

    analyzer = SentimentAnalyzer(model=model)

    logging.info(f"Generating sentiment features using {model}...")

    # Calculate sentiment from description
    texts = issues_df.get("clean_title", pd.Series([""] * len(issues_df))).fillna("") + " " + \
            issues_df.get("clean_body", pd.Series([""] * len(issues_df))).fillna("")
    
    sentiment_scores = texts.apply(analyzer.analyze_text)
    sentiment_labels = sentiment_scores.apply(analyzer.get_label)

    # Calculate frustration
    # Requires urgency, resolution_time_days if available
    urgencies = issues_df.get("urgency", pd.Series(["medium"] * len(issues_df)))
    res_times = issues_df.get("resolution_time_days", pd.Series([0.0] * len(issues_df)))
    
    frustration_levels = [
        analyzer.compute_frustration(score, urg, res)
        for score, urg, res in zip(sentiment_scores, urgencies, res_times)
    ]

    # Remove old frustration_score and sentiment columns before merging if they exist
    cols_to_drop = [
        "frustration_score", 
        "sentiment_score", 
        "sentiment_label", 
        "frustration_level"
    ]
    issues_df.drop(columns=[c for c in cols_to_drop if c in issues_df.columns], inplace=True)
    rl_df.drop(columns=[c for c in cols_to_drop if c in rl_df.columns], inplace=True)

    # Update DataFrames with new sentiment data
    issues_df["sentiment_score"] = sentiment_scores
    issues_df["sentiment_label"] = sentiment_labels
    issues_df["frustration_level"] = frustration_levels

    # Update RL Dataset
    if "issue_id" in rl_df.columns and "issue_id" in issues_df.columns:
        rl_df = rl_df.merge(
            issues_df[["issue_id", "sentiment_score", "sentiment_label", "frustration_level"]],
            on="issue_id",
            how="left"
        )
    else:
        rl_df["sentiment_score"] = sentiment_scores
        rl_df["sentiment_label"] = sentiment_labels
        rl_df["frustration_level"] = frustration_levels

    issues_df.to_csv(issues_path, index=False)
    rl_df.to_csv(rl_path, index=False)
    
    logging.info(f"Updated {issues_path} and {rl_path} with sentiment features")


def predict(text: str, urgency: str = "medium", resolution_hours: float = 0.0, model: str = "vader") -> Dict[str, Any]:
    """Predicts sentiment features for a single text input."""
    analyzer = SentimentAnalyzer(model=model)
    score = analyzer.analyze_text(text)
    label = analyzer.get_label(score)
    frustration = analyzer.compute_frustration(score, urgency, resolution_hours / 24.0)
    
    return {
        "text": text,
        "sentiment_score": round(score, 4),
        "sentiment_label": label,
        "frustration_level": round(frustration, 4)
    }


def analyse(issues_path: str = "data/final/cleaned_issues.csv") -> Dict[str, Any]:
    """Provides basic stats on sentiment distributions."""
    if not os.path.exists(issues_path):
        return {"error": "Issues CSV not found"}
        
    df = pd.read_csv(issues_path)
    
    if "sentiment_score" not in df.columns:
        return {"error": "Sentiment columns not generated yet."}
        
    label_dist = df["sentiment_label"].value_counts().to_dict()
    avg_frustration = df["frustration_level"].mean()
    
    return {
        "label_distribution": label_dist,
        "average_frustration": round(avg_frustration, 3),
        "total_analyzed": len(df)
    }


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analyzer for NexResolve")
    parser.add_argument("--issues_path", default="data/final/cleaned_issues.csv")
    parser.add_argument("--rl_path", default="data/final/final_rl_dataset.csv")
    parser.add_argument("--model", default="vader", choices=["vader", "textblob"])
    parser.add_argument("--mode", default="generate", choices=["generate", "analyse", "predict"])
    parser.add_argument("--text", default="The system crashed and I lost my data! Unacceptable.", help="Text for prediction")
    parser.add_argument("--urgency", default="high", help="Urgency for prediction")
    parser.add_argument("--resolution_hours", type=float, default=168.0, help="Resolution time in hours for prediction")
    
    args = parser.parse_args()

    if args.mode == "generate":
        generate_features(args.issues_path, args.rl_path, args.model)
        print("\n✓ Sentiment detection complete.")
        print("  Next steps:")
        print("    python evaluation/sentiment_eval.py")
        
    elif args.mode == "analyse":
        result = analyse(args.issues_path)
        print(json.dumps(result, indent=2))
        
    elif args.mode == "predict":
        result = predict(
            text=args.text, 
            urgency=args.urgency, 
            resolution_hours=args.resolution_hours, 
            model=args.model
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
