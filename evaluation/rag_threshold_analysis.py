import pandas as pd
import numpy as np
import json
import os

# ── Load dataset ─────────────────────────────────────────────
df = pd.read_csv("data/final/rl_ready_dataset.csv")

# ── Thresholds to test ───────────────────────────────────────
gap_thresholds = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7]
high_conf_thresholds = [0.7, 0.75, 0.8, 0.85]

results = []

# ── Analysis loop ────────────────────────────────────────────
for gap_t in gap_thresholds:
    for high_t in high_conf_thresholds:

        df["gap_flag_test"] = (df["max_sim"] < gap_t).astype(int)
        df["high_conf_flag"] = (df["max_sim"] >= high_t).astype(int)

        # ── GAP ANALYSIS ─────────────────────────────────────
        gap_1 = df[df["gap_flag_test"] == 1]
        gap_0 = df[df["gap_flag_test"] == 0]

        reopen_gap1 = gap_1["reopen_count"].mean() if len(gap_1) else 0
        reopen_gap0 = gap_0["reopen_count"].mean() if len(gap_0) else 0

        success_gap1 = gap_1["resolution_success"].mean() if len(gap_1) else 0
        success_gap0 = gap_0["resolution_success"].mean() if len(gap_0) else 0

        gap_ratio = len(gap_1) / len(df)

        gap_score = (reopen_gap1 - reopen_gap0) + (success_gap0 - success_gap1)

        # 🔥 NEW: penalty for extreme ratios
        balance_penalty = abs(gap_ratio - 0.35)  # ideal ~35%
        adjusted_gap_score = gap_score - balance_penalty

        # ── HIGH CONF ANALYSIS ───────────────────────────────
        hc_1 = df[df["high_conf_flag"] == 1]
        hc_0 = df[df["high_conf_flag"] == 0]

        success_hc1 = hc_1["resolution_success"].mean() if len(hc_1) else 0
        success_hc0 = hc_0["resolution_success"].mean() if len(hc_0) else 0

        reopen_hc1 = hc_1["reopen_count"].mean() if len(hc_1) else 0
        reopen_hc0 = hc_0["reopen_count"].mean() if len(hc_0) else 0

        hc_ratio = len(hc_1) / len(df)

        high_conf_score = (success_hc1 - success_hc0) + (reopen_hc0 - reopen_hc1)

        results.append({
            "gap_threshold": gap_t,
            "high_conf_threshold": high_t,

            "gap_ratio": round(gap_ratio, 4),
            "gap_score": round(gap_score, 4),
            "adjusted_gap_score": round(adjusted_gap_score, 4),

            "gap1_success": round(success_gap1, 4),
            "gap0_success": round(success_gap0, 4),

            "hc_ratio": round(hc_ratio, 4),
            "high_conf_score": round(high_conf_score, 4),

            "hc1_success": round(success_hc1, 4),
            "hc0_success": round(success_hc0, 4),
        })

# ── Convert to DataFrame ─────────────────────────────────────
res_df = pd.DataFrame(results)

# ── Sort for visibility ──────────────────────────────────────
res_df_sorted_gap = res_df.sort_values("adjusted_gap_score", ascending=False)
res_df_sorted_hc = res_df.sort_values("high_conf_score", ascending=False)

# ── Select best thresholds ───────────────────────────────────
best_gap_row = res_df_sorted_gap.iloc[0]
best_hc_row = res_df_sorted_hc.iloc[0]

best_gap_threshold = best_gap_row["gap_threshold"]
best_high_conf_threshold = best_hc_row["high_conf_threshold"]

# ── Prepare output directory ─────────────────────────────────
output_dir = "evaluation/reports"
os.makedirs(output_dir, exist_ok=True)

# ── Save JSON ────────────────────────────────────────────────
json_path = os.path.join(output_dir, "rag_threshold_analysis.json")

with open(json_path, "w") as f:
    json.dump({
        "best_gap_threshold": float(best_gap_threshold),
        "best_high_conf_threshold": float(best_high_conf_threshold),
        "results": results
    }, f, indent=4)

# ── Save TXT report ──────────────────────────────────────────
txt_path = os.path.join(output_dir, "rag_threshold_analysis.txt")

with open(txt_path, "w") as f:
    f.write("RAG THRESHOLD ANALYSIS REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write("FULL RESULTS (Top 10 by adjusted gap score):\n")
    f.write(res_df_sorted_gap.head(10).to_string(index=False))
    f.write("\n\n")

    f.write("TOP HIGH CONFIDENCE SETTINGS:\n")
    f.write(res_df_sorted_hc.head(10).to_string(index=False))
    f.write("\n\n")

    f.write(f"Selected GAP threshold: {best_gap_threshold}\n")
    f.write(f"Selected HIGH CONF threshold: {best_high_conf_threshold}\n\n")

    f.write("Interpretation:\n")
    f.write(
        "Gap threshold is selected using adjusted score balancing separation and distribution.\n"
        "High confidence threshold is selected based on strong success and low reopen rates.\n"
        "This ensures both reliability and realistic system behavior.\n"
    )

# ── Print summary ────────────────────────────────────────────
print("\n=== FULL RESULTS (Top 10 GAP) ===")
print(res_df_sorted_gap.head(10))

print("\n=== TOP HIGH CONF SETTINGS ===")
print(res_df_sorted_hc.head(10))

print(f"\nSelected GAP threshold: {best_gap_threshold}")
print(f"Selected HIGH CONF threshold: {best_high_conf_threshold}")

print(f"\nSaved to:\n- {json_path}\n- {txt_path}")