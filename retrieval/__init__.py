

from retrieval.embedder             import generate_embeddings, embed_single
from retrieval.faiss_index          import build_and_save_index, load_index
from retrieval.retriever            import retrieve, combine_ticket_text
from retrieval.retrieval_evaluator  import evaluate_retrieval
from retrieval.retrieval_optimizer  import optimize_retrieval
from retrieval.similarity_features  import compute_similarity_features
from retrieval.knowledge_gap        import compute_knowledge_gap_flag, compute_retrieval_confidence

__all__ = [
    # Embedding
    "generate_embeddings",
    "embed_single",
    # Index management
    "build_and_save_index",
    "load_index",
    # Retrieval
    "retrieve",
    "combine_ticket_text",
    # Post-processing
    "evaluate_retrieval",
    "optimize_retrieval",
    # RL signals
    "compute_similarity_features",
    "compute_knowledge_gap_flag",
    "compute_retrieval_confidence",
]
