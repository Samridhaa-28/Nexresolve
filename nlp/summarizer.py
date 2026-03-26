import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.warning("Spacy model 'en_core_web_sm' not found. It will be downloaded on first prediction.")
    nlp = None


class Summarizer:
    def __init__(self, model: str = "textrank"):
        self.model = model
        self.bart_pipeline = None
        global nlp

        if self.model == "bart":
            try:
                from transformers import pipeline
                self.bart_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
            except Exception as e:
                logging.error(f"Failed to load BART model: {e}")
                self.model = "textrank"
                
        if self.model == "textrank":
            if nlp is None:
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                nlp = spacy.load("en_core_web_sm")

    def summarize(self, text: str, ratio: float = 0.3, max_length: int = 150) -> str:
        """Returns a summarized version of the input text."""
        if not isinstance(text, str) or not text.strip():
            return ""

        if self.model == "bart" and self.bart_pipeline:
            # BART generation
            # Limit input length to roughly fit in 1024 tokens if text is very long
            truncated_text = text[:3000]
            try:
                res = self.bart_pipeline(truncated_text, max_length=max_length, min_length=15, do_sample=False)
                return res[0]["summary_text"]
            except Exception as e:
                logging.warning(f"BART summarization failed: {e}. Falling back to TextRank.")
        
        # TextRank (Default/Fallback)
        return self._text_rank_summarize(text, ratio)

    def _text_rank_summarize(self, text: str, ratio: float) -> str:
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        
        if len(sentences) <= 2:
            return text

        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            return text
            
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        num_sentences = max(1, int(len(sentences) * ratio))
        
        # Sort back to original order to form summary
        original_order = {s: i for i, s in enumerate(sentences)}
        selected = [s for _, s in ranked_sentences[:num_sentences]]
        selected.sort(key=lambda s: original_order[s])
        
        return " ".join(selected)


def generate_features(
    issues_path: str = "data/final/cleaned_issues.csv",
    rl_path: str = "data/final/final_rl_dataset.csv",
    model: str = "textrank"
) -> None:
    """Generates summary features and updates datasets."""
    if not os.path.exists(issues_path) or not os.path.exists(rl_path):
        logging.error("Source CSV files not found.")
        return

    issues_df = pd.read_csv(issues_path)
    rl_df = pd.read_csv(rl_path)

    summarizer = Summarizer(model=model)

    logging.info(f"Generating summaries using {model}...")

    texts = issues_df.get("clean_title", pd.Series([""] * len(issues_df))).fillna("") + " " + \
            issues_df.get("clean_body", pd.Series([""] * len(issues_df))).fillna("")
            
    # For large datasets limit using BART
    if model == "bart" and len(texts) > 500:
        logging.warning("BART summarization is slow on CPU. This may take a while.")

    # Using tqdm if it exists
    summaries = []
    
    try:
        from tqdm import tqdm
        for text in tqdm(texts, desc="Summarizing"):
            summaries.append(summarizer.summarize(text))
    except ImportError:
        for i, text in enumerate(texts):
            summaries.append(summarizer.summarize(text))
            if (i + 1) % 100 == 0:
                logging.info(f"Processed {i + 1} / {len(texts)}")

    issues_df["summary"] = summaries

    if "summary" in rl_df.columns:
        rl_df.drop(columns=["summary"], inplace=True)
        
    if "issue_id" in rl_df.columns and "issue_id" in issues_df.columns:
        rl_df = rl_df.merge(
            issues_df[["issue_id", "summary"]],
            on="issue_id",
            how="left"
        )
    else:
        rl_df["summary"] = summaries

    issues_df.to_csv(issues_path, index=False)
    rl_df.to_csv(rl_path, index=False)
    
    logging.info(f"Updated {issues_path} and {rl_path} with summary features")


def predict(text: str, model: str = "textrank") -> Dict[str, Any]:
    """Predicts a summary for a single text input."""
    summarizer = Summarizer(model=model)
    summary = summarizer.summarize(text)
    
    return {
        "text_length": len(text),
        "summary_length": len(summary),
        "summary": summary
    }


def analyse(issues_path: str = "data/final/cleaned_issues.csv") -> Dict[str, Any]:
    """Provides basic stats on summary lengths."""
    if not os.path.exists(issues_path):
        return {"error": "Issues CSV not found"}
        
    df = pd.read_csv(issues_path)
    
    if "summary" not in df.columns:
        return {"error": "Summary column not generated yet."}
        
    df["full_text"] = df.get("clean_title", "").fillna("") + " " + df.get("clean_body", "").fillna("")
    df["orig_len"] = df["full_text"].apply(lambda x: len(str(x)))
    df["summ_len"] = df["summary"].apply(lambda x: len(str(x)))
    
    compression_ratio = df["summ_len"].sum() / df["orig_len"].sum()
    
    return {
        "total_analyzed": len(df),
        "avg_original_length": round(df["orig_len"].mean(), 2),
        "avg_summary_length": round(df["summ_len"].mean(), 2),
        "compression_ratio": round(compression_ratio, 4)
    }


def main():
    parser = argparse.ArgumentParser(description="Summarizer for NexResolve")
    parser.add_argument("--issues_path", default="data/final/cleaned_issues.csv")
    parser.add_argument("--rl_path", default="data/final/final_rl_dataset.csv")
    parser.add_argument("--model", default="textrank", choices=["textrank", "bart"])
    parser.add_argument("--mode", default="generate", choices=["generate", "analyse", "predict"])
    parser.add_argument("--text", default="I was running the server when it crashed immediately. I tried restarting it multiple times but it always fails with a memory leak error. Please fix this critical bug immediately.", help="Text for prediction")
    
    args = parser.parse_args()

    if args.mode == "generate":
        generate_features(args.issues_path, args.rl_path, args.model)
        print("\n[Done] Summarization complete.")
        print("  Next steps:")
        print("    python evaluation/summarizer_eval.py")
        
    elif args.mode == "analyse":
        result = analyse(args.issues_path)
        print(json.dumps(result, indent=2))
        
    elif args.mode == "predict":
        result = predict(text=args.text, model=args.model)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
