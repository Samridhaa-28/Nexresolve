import re
import numpy as np

from retrieval.embedder   import generate_embeddings
from retrieval.faiss_index import load_index, DEFAULT_INDEX_PATH
from nlp.summarizer import Summarizer

summarizer = Summarizer(model="textrank")

# ---------------------------------------------------------------------------
# Text preprocessing — keeps consistent with KB embedding style
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Minimal normalisation:
      - lowercase
      - collapse whitespace
    (mirrors the KB clean_text preprocessing)
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_text_for_embedding(text: str, max_words: int = 150) -> str:
    words = text.split()

    if len(words) > 150:
       return summarizer.summarize(text)  # summarizer
    return text

def combine_ticket_text(title: str = "", body: str = "") -> str:
    """
    Combine ticket title and body into a single retrieval text.
    If only one field is provided the other is treated as empty.
    """
    parts = [p.strip() for p in [title, body] if p and p.strip()]
    combined = " ".join(parts)
    return _clean_text(combined)


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def retrieve_similar_issues(
    query_text: str,
    index,                     # faiss.IndexFlatIP
    metadata: list[dict],      # list returned by faiss_index.prepare_kb
    top_k: int = 5,
    query_issue_number = None 
) -> list[dict]:
    """
    Find the top-k most similar resolved issues for a given ticket text.

    Parameters
    ----------
    query_text : cleaned text of the incoming ticket
    index      : loaded FAISS IndexFlatIP
    metadata   : list of dicts (position i → KB row info)
    top_k      : number of candidates to retrieve

    Returns
    -------
    List of dicts, sorted by similarity descending. Each dict:
        {
          "rank"              : int  (1 = best match),
          "issue_number"      : int,
          "primary_label"     : str,
          "similarity_score"  : float  (cosine similarity, 0–1),
          "solution_comments" : str,
          "clean_text"        : str    (the KB ticket text),
        }
    """
    if not query_text.strip():
        return []

    # Embed query — returns (1, dim) array
    query_text = prepare_text_for_embedding(query_text)

    # Embed query — returns (1, dim) array
    query_emb = generate_embeddings([query_text], show_progress=False)

    # FAISS search — distances here ARE cosine similarities (inner product on normed vecs)
    actual_k = min(top_k, index.ntotal)
    distances, indices = index.search(query_emb, actual_k)

    results = []
    for rank_idx, (dist, pos) in enumerate(zip(distances[0], indices[0])):
        if pos < 0:           # FAISS sentinel for "no result"
            continue
        row = metadata[pos]
        

        if query_issue_number is not None and int(row["issue_number"]) == int(query_issue_number):
            continue
        results.append({
            "rank"              : rank_idx + 1,
            "issue_number"      : row["issue_number"],
            "primary_label"     : row["primary_label"],
            "similarity_score"  : float(np.clip(dist, 0.0, 1.0)),  # cosine ∈ [0,1]
            "solution_comments" : row["solution_comments"],
            "clean_text"        : row["clean_text"],
            "kb_quality_tier": row.get("kb_quality_tier", "tier1_verified")
        })

    # Sort descending by similarity (usually already sorted by FAISS, but be safe)
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Convenience: one-shot retrieve from freshly loaded index
# ---------------------------------------------------------------------------

def retrieve(
    query_text:  str,
    top_k:       int = 5,
    index_path:  str = DEFAULT_INDEX_PATH,
    index=None,
    metadata=None,
    query_issue_number: int = None 
) -> list[dict]:
    """
    High-level helper. Loads the index if not already supplied in memory.

    Parameters
    ----------
    query_text  : incoming ticket text (pre-cleaned or raw; will be cleaned here)
    top_k       : number of results to return
    index_path  : where the .faiss and .meta.json files live
    index       : pre-loaded FAISS index (skip disk load if provided)
    metadata    : pre-loaded metadata list (skip disk load if provided)

    Returns
    -------
    
    """
    if index is None or metadata is None:
        index, metadata = load_index(index_path)

    query_text = _clean_text(query_text)
    return retrieve_similar_issues(query_text, index, metadata, top_k=top_k,query_issue_number=query_issue_number)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os, sys, pprint
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    sample_query = combine_ticket_text(
        title="Cannot login after resetting password",
        body="Login keeps failing with invalid session error after I changed my password.",
    )
    print(f"Query: {sample_query!r}\n")

    results = retrieve(sample_query, top_k=5)
    print(f"Top-{len(results)} retrieved issues:")
    for r in results:
        print(f"  Rank {r['rank']}: issue #{r['issue_number']} | "
              f"label={r['primary_label']} | sim={r['similarity_score']:.4f}")
        print(f"    Solution snippet: {r['solution_comments'][:120]}…")
