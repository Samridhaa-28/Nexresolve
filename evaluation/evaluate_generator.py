import os
import sys
import pandas as pd
import numpy as np
import json
import torch
from tqdm import tqdm
from rouge_score import rouge_scorer
try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None

# Ensure root dir is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generation.generator import ResponseGenerator

# Config
SEED = 42
SAMPLE_SIZE = 100
REPORT_DIR = "evaluation/reports"
BEST_GEN_PATH = "models/rl/best_generator.txt"

def compute_structure_score(text):
    """
    Score based on presence of lists/steps/bullets (standard for solutions).
    """
    score = 0
    if any(marker in text for marker in ["1.", "2.", "3.", "-", "* ", "Step"]):
        score += 0.5
    if len(text.split('\n')) > 2:
        score += 0.5
    return score

def compute_grounding_score(generated, retrieved):
    """
    Simple word overlap as a proxy for factual grounding if bert-score fails.
    """
    gen_words = set(generated.lower().split())
    ret_words = set(retrieved.lower().split())
    if not ret_words: return 1.0
    overlap = gen_words.intersection(ret_words)
    return len(overlap) / len(ret_words) if len(ret_words) > 0 else 0

def main():
    np.random.seed(SEED)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    print("Loading datasets...")
    rl_df = pd.read_csv("data/final/rl_ready_dataset.csv")
    final_df = pd.read_csv("data/final/final_dataset.csv")
    
    # Merge on issue_number
    df = rl_df.merge(final_df[["issue_number", "solution_comments", "labels_normalised"]], on="issue_number", how="inner")
    
    # Clean and Sample
    df = df.dropna(subset=["solution_comments"])
    df = df[df["solution_comments"].str.strip() != ""]
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)
    
    print(f"Evaluating on {len(df)} samples...")
    
    models = ["FLAN", "BART"]
    results = {}
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    for m_name in models:
        print(f"  Testing {m_name}...")
        gen = ResponseGenerator(model_type=m_name)
        
        m_metrics = {
            "rougeL": [],
            "bert_score": [],
            "structure": [],
            "grounding": []
        }
        generated_outputs = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Prepare inputs
            # Reconstruct entities from clean_text if needed, or use dummy
            # Generator.generate(ticket_summary, intent, entities, retrieved_solution)
            summary = row["clean_text"][:500] # Proxy for summary
            intent = row["primary_label"]
            retrieved = row.get("rl_recommendation", "No specific solution found.")
            target = row["solution_comments"]
            
            # Generate
            try:
                # We'll pass empty entities list as default
                output = gen.generate(summary, intent, [], retrieved)
                generated_outputs.append(output)
            except Exception as e:
                print(f"Generation failed for {m_name}: {e}")
                output = ""
            
            # Scores
            r_score = scorer.score(target, output)['rougeL'].fmeasure
            m_metrics["rougeL"].append(r_score)
            m_metrics["structure"].append(compute_structure_score(output))
            m_metrics["grounding"].append(compute_grounding_score(output, retrieved))
            
            # BERTScore (Optional)
            if bert_score:
                # We'll batch this for efficiency if possible, but for 100 samples it's okay individually
                # Or we can do it after the loop
                pass
        
        # Batch BERTScore if available
        if bert_score:
            print(f"    Computing BERTScore for {m_name}...")
            texts = df["solution_comments"].tolist()
            #preds = [gen.generate(row["clean_text"][:500], row["primary_label"], [], row.get("rl_recommendation", "")) for _, row in df.iterrows()]
            P, R, F1 = bert_score(generated_outputs, texts, lang="en", verbose=False)
            m_metrics["bert_score"] = F1.tolist()
        else:
            m_metrics["bert_score"] = [0] * len(df)
            
        results[m_name] = {
            "avg_rougeL": np.mean(m_metrics["rougeL"]),
            "avg_bert_score": np.mean(m_metrics["bert_score"]),
            "avg_structure": np.mean(m_metrics["structure"]),
            "avg_grounding": np.mean(m_metrics["grounding"]),
            "composite": (np.mean(m_metrics["rougeL"]) + np.mean(m_metrics["bert_score"]) + np.mean(m_metrics["structure"])) / 3
        }
        
        # Free memory
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    with open(os.path.join(REPORT_DIR, "generator_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    # Determine winner
    winner = max(results, key=lambda k: results[k]["composite"])
    print(f"\nWinner: {winner}")
    
    with open(BEST_GEN_PATH, "w") as f:
        f.write(winner)
        
    print(f"Results saved to {REPORT_DIR}")
    print(f"Best model record saved to {BEST_GEN_PATH}")

if __name__ == "__main__":
    main()
