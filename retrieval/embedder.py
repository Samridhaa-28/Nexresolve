import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Module-level singleton — load once, reuse everywhere
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance (lazy-load)."""
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {MODEL_NAME} …")
        _model = SentenceTransformer(MODEL_NAME,device="cpu")
        print("[Embedder] Model loaded successfully.")
    return _model


# ---------------------------------------------------------------------------
# Core embedding function
# ---------------------------------------------------------------------------

def generate_embeddings(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    import torch
    torch.cuda.empty_cache()
    model = get_model()

    # Encode – sentence-transformers returns a float32 numpy array
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalise in-place
    )

    return embeddings.astype(np.float32)


def embed_single(text: str) -> np.ndarray:
    """
    Convenience wrapper for embedding a single string.

    Returns
    -------
    np.ndarray of shape (embedding_dim,), dtype float32, L2-normed
    """
    emb = generate_embeddings([text], show_progress=False)
    return emb[0]


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "Cannot login after resetting password",
        "Jupyter notebook crashes on startup after VS Code update",
        "Rate limiting error even though quota is not exceeded",
    ]
    print("=== Embedder Demo ===")
    vecs = generate_embeddings(sample_texts, show_progress=True)
    print(f"Shape: {vecs.shape}")          # e.g. (3, 384)
    print(f"Dtype: {vecs.dtype}")          # float32
    # Spot-check: each row should have unit norm after normalisation
    norms = np.linalg.norm(vecs, axis=1)
    print(f"L2 norms (should all be ~1.0): {norms}")
