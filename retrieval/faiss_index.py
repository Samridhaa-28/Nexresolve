import os
import json
import numpy as np
import pandas as pd
import faiss

from retrieval.embedder import generate_embeddings

# ---------------------------------------------------------------------------
# Default paths (relative to project root; override via arguments)
# ---------------------------------------------------------------------------
DEFAULT_KB_PATH = "data/splits/knowledge_base.csv"
DEFAULT_INDEX_PATH = "data/retrieval/kb_index"   # saved as .faiss + .meta.json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Create parent directories if they do not exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------------------------------------------------------------------------
# Prepare knowledge base
# ---------------------------------------------------------------------------

def load_knowledge_base(kb_path: str = DEFAULT_KB_PATH) -> pd.DataFrame:
   
    df = pd.read_csv(kb_path)

    # --- drop rows with missing text ---
    df = df[df["clean_text"].notna() & (df["clean_text"].str.strip() != "")]

    # --- drop rows with no useful solution ---
    df = df[
        df["solution_comments"].notna()
        & (~df["solution_comments"].isin(["NO_COMMENTS", ""]))
        & (df["solution_comments"].str.strip() != "")
    ]

    # --- deduplicate ---
    df = df.drop_duplicates(subset=["clean_text", "solution_comments"])
    df = df.reset_index(drop=True)

    print(f"[FAISS Index] KB loaded — {len(df)} usable rows after cleaning.")
    return df


def prepare_kb(df: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """
    Extract texts to embed + metadata to keep alongside the index.

    Returns
    -------
    texts    : list of strings (clean_text), one per KB row
    metadata : list of dicts with keys:
                 issue_number, solution_comments, primary_label, clean_text
    """
    texts = df["clean_text"].tolist()
    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            "issue_number":      int(row["issue_number"]),
            "solution_comments": str(row["solution_comments"]),
            "primary_label":     str(row.get("primary_label", "unknown")),
            "clean_text":        str(row["clean_text"]),
            "kb_quality_tier": str(row.get("kb_quality_tier", "tier1_verified"))
        })
    return texts, metadata


# ---------------------------------------------------------------------------
# Build FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(
    embeddings: np.ndarray,
) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP (inner-product / cosine-similarity index).

    Parameters
    ----------
    embeddings : (n_rows, embedding_dim) float32 array, must be L2-normalised

    Returns
    -------
    FAISS IndexFlatIP with all rows added
    """
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)   # inner-product = cosine for normed vecs
    index.add(embeddings)
    print(f"[FAISS Index] Index built — {index.ntotal} vectors, dim={embedding_dim}.")
    return index


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_index(
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    index_path: str = DEFAULT_INDEX_PATH,
) -> None:
    """Save FAISS index binary + JSON metadata side-car."""
    _ensure_dir(index_path)
    faiss_path = index_path + ".faiss"
    meta_path  = index_path + ".meta.json"

    faiss.write_index(index, faiss_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[FAISS Index] Saved → {faiss_path}")
    print(f"[FAISS Index] Saved → {meta_path}")


def load_index(
    index_path: str = DEFAULT_INDEX_PATH,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Load a previously saved FAISS index + metadata.

    Returns
    -------
    (index, metadata)  — same types as returned by build_faiss_index + prepare_kb
    """
    faiss_path = index_path + ".faiss"
    meta_path  = index_path + ".meta.json"

    if not os.path.exists(faiss_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{faiss_path}'. "
            "Run build_and_save_index() first."
        )

    index    = faiss.read_index(faiss_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"[FAISS Index] Loaded — {index.ntotal} vectors from '{faiss_path}'.")
    return index, metadata


# ---------------------------------------------------------------------------
# Pipeline convenience — build everything in one call
# ---------------------------------------------------------------------------

def build_and_save_index(
    kb_path:    str = DEFAULT_KB_PATH,
    index_path: str = DEFAULT_INDEX_PATH,
    batch_size: int = 64,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Full pipeline:
      1. Load & clean KB
      2. Embed clean_text   (NOT solution_comments)
      3. Build FAISS index
      4. Save to disk
      5. Return (index, metadata)
    """
    df = load_knowledge_base(kb_path)
    texts, metadata = prepare_kb(df)

    print(f"[FAISS Index] Embedding {len(texts)} KB texts …")
    embeddings = generate_embeddings(texts, batch_size=batch_size, show_progress=True)

    index = build_faiss_index(embeddings)
    save_index(index, metadata, index_path)
    return index, metadata


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os, sys
    # Run from project root: python -m retrieval.faiss_index
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    index, meta = build_and_save_index()
    print(f"\nIndex total vectors : {index.ntotal}")
    print(f"Metadata entries    : {len(meta)}")
    print("Sample metadata[0]  :")
    import pprint; pprint.pprint(meta[0])
