"""Modular reranker component with LangChain integration."""

from typing import List, Dict, Any, Optional, Union, Tuple

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


class Reranker:
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda",
        max_length: int = 512
    ):
        """Initialize reranker with cross-encoder model.

        Args:
            model_name: HuggingFace cross-encoder model name
            device: Device for inference (cuda/cpu)
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        self._model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length
        )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top documents to return (None = all)
            return_scores: Whether to include scores in output

        Returns:
            List of dicts with 'content', 'score', and 'rank'
        """
        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            result = {
                "content": documents[idx],
                "rank": rank,
                "original_index": idx
            }
            if return_scores:
                result["score"] = float(scores[idx])
            results.append(result)

        return results

    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        content_key: str = "content",
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank documents while preserving metadata.

        Args:
            query: Search query
            documents: List of dicts containing document content and metadata
            content_key: Key for document text in each dict
            top_k: Number of top documents to return

        Returns:
            Reranked documents with scores and original metadata
        """
        if not documents:
            return []

        texts = [doc[content_key] for doc in documents]
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            result = dict(documents[idx])
            result["score"] = float(scores[idx])
            result["rank"] = rank
            results.append(result)

        return results

    def rerank_langchain_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """Rerank LangChain Document objects.

        Args:
            query: Search query
            documents: List of LangChain Document objects
            top_k: Number of top documents to return

        Returns:
            List of (Document, score) tuples
        """
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        return [(documents[idx], float(scores[idx])) for idx in ranked_indices]

    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair.

        Args:
            query: Search query
            document: Document text

        Returns:
            Relevance score
        """
        return float(self._model.predict([(query, document)])[0])

    def score_pairs(
        self,
        pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """Score multiple query-document pairs.

        Args:
            pairs: List of (query, document) tuples

        Returns:
            List of relevance scores
        """
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


def create_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "cuda",
    max_length: int = 512
) -> Reranker:
    """Factory function to create a reranker.

    Args:
        model_name: Cross-encoder model name
        device: Device for inference
        max_length: Maximum sequence length

    Returns:
        Configured Reranker instance
    """
    return Reranker(
        model_name=model_name,
        device=device,
        max_length=max_length
    )


# Common model presets
MODELS = {
    "tiny": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "small": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "medium": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "large": "cross-encoder/ms-marco-electra-base",
    "bge-base": "BAAI/bge-reranker-base",
    "bge-large": "BAAI/bge-reranker-large",
    "bge-v2": "BAAI/bge-reranker-v2-m3",  # Best quality, multilingual
}


def get_reranker(preset: str = "small", device: str = "cuda") -> Reranker:
    """Get a reranker by preset name.

    Args:
        preset: Model preset (tiny, small, medium, large)
        device: Device for inference

    Returns:
        Configured Reranker instance
    """
    if preset not in MODELS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(MODELS.keys())}")

    return create_reranker(model_name=MODELS[preset], device=device)
