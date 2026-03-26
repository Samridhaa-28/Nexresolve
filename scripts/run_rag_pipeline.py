import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.state_builder import build_state_vectors, get_state_column_names


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full RAG pipeline and build RL-ready state matrix."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/final/final_rl_dataset.csv",
        help="Path to final_rl_dataset.csv.",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="data/retrieval/kb_index",
        help="Path prefix for FAISS index files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/final/rl_ready_dataset.csv",
        help="Where to write the RAG-augmented CSV.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-k retrieved issues per ticket.",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save output CSV (dry run).",
    )
    return parser.parse_args()


def print_sample_states(df, state_matrix, n=3):
    """Print a few sample rows to verify the pipeline worked."""
    print(f"\n{'='*62}")
    print(f"  Sample state vectors (first {n} tickets):")
    print(f"{'='*62}")
    cols = get_state_column_names()
    for i in range(min(n, len(df))):
        issue = df["issue_number"].iloc[i] if "issue_number" in df.columns else i
        print(f"\n  Ticket #{issue}")
        rag_cols = ["max_sim", "avg_sim", "sim_spread", "knowledge_gap_flag"]
        for col in rag_cols:
            idx = cols.index(col)
            print(f"    {col:<28} = {state_matrix[i, idx]:.4f}")


def main():
    args = parse_args()

    print("=" * 62)
    print("  NexResolve — RAG Pipeline + RL State Builder")
    print("=" * 62)
    print(f"  Dataset      : {args.dataset_path}")
    print(f"  Index        : {args.index_path}")
    print(f"  Output       : {args.output_path}")
    print(f"  Top-K        : {args.top_k}")
    print("=" * 62 + "\n")

    t0 = time.time()

    df_aug, state_matrix = build_state_vectors(
        dataset_path = args.dataset_path,
        index_path   = args.index_path,
        output_path  = args.output_path,
        top_k        = args.top_k,
        save         = not args.no_save,
    )

    elapsed = time.time() - t0

    # ── Summary stats ──────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  Pipeline complete!")
    print(f"{'='*62}")
    print(f"  Processed      : {len(df_aug)} tickets")
    print(f"  State dim      : {state_matrix.shape[1]} features")
    print(f"  Elapsed        : {elapsed:.1f}s  ({elapsed/len(df_aug)*1000:.1f} ms/ticket)")
    print(f"  NaN in matrix  : {np.isnan(state_matrix).sum()}")

    # ── RAG signal stats ───────────────────────────────────────────────────
    print(f"\n  RAG Signal Summary:")
    for col in ["max_sim", "avg_sim", "sim_spread", "knowledge_gap_flag"]:
        if col in df_aug.columns:
            s = df_aug[col]
            print(f"    {col:<28} mean={s.mean():.4f}  std={s.std():.4f}  "
                  f"min={s.min():.4f}  max={s.max():.4f}")

    # ── Knowledge gap breakdown ────────────────────────────────────────────
    if "knowledge_gap_flag" in df_aug.columns:
        gap_count     = (df_aug["knowledge_gap_flag"] == 1).sum()
        no_gap_count  = (df_aug["knowledge_gap_flag"] == 0).sum()
        print(f"\n  Knowledge Gap Breakdown:")
        print(f"    Gap (flag=1) : {gap_count:>4}  ({100*gap_count/len(df_aug):.1f}%)")
        print(f"    No Gap (0)   : {no_gap_count:>4}  ({100*no_gap_count/len(df_aug):.1f}%)")

    print_sample_states(df_aug, state_matrix)

    if not args.no_save:
        print(f"\n  Saved → {args.output_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
