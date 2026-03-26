
import argparse
import sys
import os

# Ensure project root is on path when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.faiss_index import build_and_save_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and save FAISS knowledge-base index for NexResolve RAG."
    )
    parser.add_argument(
        "--kb_path",
        type=str,
        default="data/splits/knowledge_base.csv",
        help="Path to knowledge_base.csv (resolved issues).",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="data/retrieval/kb_index",
        help="Output path prefix for .faiss and .meta.json files.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Embedding batch size (lower if running out of RAM).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  NexResolve — Knowledge Base Index Builder")
    print("=" * 60)
    print(f"  KB path    : {args.kb_path}")
    print(f"  Index path : {args.index_path}")
    print(f"  Batch size : {args.batch_size}")
    print("=" * 60 + "\n")

    index, metadata = build_and_save_index(
        kb_path    = args.kb_path,
        index_path = args.index_path,
        batch_size = args.batch_size,
    )

    print("\n" + "=" * 60)
    print("  Index build complete!")
    print(f"  Total vectors  : {index.ntotal}")
    print(f"  Metadata rows  : {len(metadata)}")
    print(f"  Saved to       : {args.index_path}.faiss")
    print(f"                   {args.index_path}.meta.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
