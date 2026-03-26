import pandas as pd
import numpy as np
import json
import os

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/final/rl_ready_dataset.csv")

output_dir = "evaluation/reports"
os.makedirs(output_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. SIMILARITY DISTRIBUTION
# ─────────────────────────────────────────────────────────────
sim_stats = {
    "mean": float(df["max_sim"].mean()),
    "std": float(df["max_sim"].std()),
    "min": float(df["max_sim"].min()),
    "max": float(df["max_sim"].max()),
}

# ─────────────────────────────────────────────────────────────
# 2. BUCKET ANALYSIS
# ─────────────────────────────────────────────────────────────
bins = [0, 0.3, 0.5, 0.7, 1.0]
labels = ["low", "mid_low", "mid_high", "high"]

df["sim_bucket"] = pd.cut(df["max_sim"], bins=bins, labels=labels)

bucket_stats = df.groupby("sim_bucket", observed=True).agg(
    avg_success=("resolution_success", "mean"),
    avg_reopen=("reopen_count", "mean"),
    count=("sim_bucket", "count")
).reset_index()

# ─────────────────────────────────────────────────────────────
# 3. GAP VALIDATION
# ─────────────────────────────────────────────────────────────
gap_1 = df[df["knowledge_gap_flag"] == 1]
gap_0 = df[df["knowledge_gap_flag"] == 0]

gap_analysis = {
    "gap1_success": float(gap_1["resolution_success"].mean()),
    "gap0_success": float(gap_0["resolution_success"].mean()),
    "gap1_reopen": float(gap_1["reopen_count"].mean()),
    "gap0_reopen": float(gap_0["reopen_count"].mean()),
    "gap_ratio": float(len(gap_1) / len(df))
}

# ─────────────────────────────────────────────────────────────
# 4. HIGH CONF VALIDATION
# ─────────────────────────────────────────────────────────────
df["high_conf"] = (df["max_sim"] >= 0.85).astype(int)

hc_1 = df[df["high_conf"] == 1]
hc_0 = df[df["high_conf"] == 0]

high_conf_analysis = {
    "hc_success": float(hc_1["resolution_success"].mean()),
    "non_hc_success": float(hc_0["resolution_success"].mean()),
    "hc_reopen": float(hc_1["reopen_count"].mean()),
    "non_hc_reopen": float(hc_0["reopen_count"].mean()),
    "hc_ratio": float(len(hc_1) / len(df))
}

# ─────────────────────────────────────────────────────────────
# 5. INTENT MATCH (Top-K proxy)
# ─────────────────────────────────────────────────────────────
# Proxy: if max_sim > threshold → assume correct retrieval

df["retrieval_correct"] = (df["max_sim"] > 0.6).astype(int)

intent_match_rate = float(df["retrieval_correct"].mean())

# ─────────────────────────────────────────────────────────────
# 6. CORRELATIONS
# ─────────────────────────────────────────────────────────────
correlations = {
    "sim_vs_success": float(df["max_sim"].corr(df["resolution_success"])),
    "sim_vs_reopen": float(df["max_sim"].corr(df["reopen_count"])),
    "gap_vs_success": float(df["knowledge_gap_flag"].corr(df["resolution_success"]))
}

# ─────────────────────────────────────────────────────────────
# SAVE JSON
# ─────────────────────────────────────────────────────────────
json_path = os.path.join(output_dir, "rl_rag_eval.json")

results = {
    "similarity_stats": sim_stats,
    "bucket_analysis": bucket_stats.to_dict(orient="records"),
    "gap_analysis": gap_analysis,
    "high_conf_analysis": high_conf_analysis,
    "intent_match_rate": intent_match_rate,
    "correlations": correlations
}

with open(json_path, "w") as f:
    json.dump(results, f, indent=4)

# ─────────────────────────────────────────────────────────────
# SAVE TEXT REPORT
# ─────────────────────────────────────────────────────────────
txt_path = os.path.join(output_dir, "rl_rag_eval.txt")

with open(txt_path, "w") as f:
    f.write("RAG + RL EVALUATION REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write("SIMILARITY DISTRIBUTION\n")
    f.write(str(sim_stats) + "\n\n")

    f.write("BUCKET ANALYSIS\n")
    f.write(bucket_stats.to_string(index=False) + "\n\n")

    f.write("GAP ANALYSIS\n")
    f.write(str(gap_analysis) + "\n\n")

    f.write("HIGH CONF ANALYSIS\n")
    f.write(str(high_conf_analysis) + "\n\n")

    f.write("INTENT MATCH RATE\n")
    f.write(str(intent_match_rate) + "\n\n")

    f.write("CORRELATIONS\n")
    f.write(str(correlations) + "\n\n")

    f.write("INTERPRETATION\n")
    f.write(
        "High similarity should correlate with higher success and lower reopen.\n"
        "Knowledge gap should correlate with lower success.\n"
        "High confidence should show strong success signal.\n"
    )

print("Evaluation complete")
print(f"Saved:\n{json_path}\n{txt_path}")