"""Modular retriever component with LangChain integration."""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class Retriever:
    """Dense retriever using HuggingFace embeddings and FAISS."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        normalize_embeddings: bool = False
    ):
        """Initialize retriever with embedding model.

        Args:
            model_name: HuggingFace model name for embeddings
            device: Device for inference (cuda/cpu)
            normalize_embeddings: Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings

        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings}
        )
        self._vector_store: Optional[LangChainFAISS] = None
        self._documents: List[Document] = []

    def build_index(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Build FAISS index from texts.

        Args:
            texts: List of text passages to index
            metadatas: Optional metadata for each passage
            ids: Optional IDs for each passage
        """
        if metadatas is None:
            metadatas = [{"id": str(i)} for i in range(len(texts))]
        if ids is None:
            ids = [str(i) for i in range(len(texts))]

        self._documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]

        self._vector_store = LangChainFAISS.from_documents(
            documents=self._documents,
            embedding=self._embeddings,
            ids=ids
        )

    def save_index(self, path: str) -> None:
        """Save index and metadata to disk.

        Args:
            path: Directory path to save index
        """
        if self._vector_store is None:
            raise RuntimeError("No index to save. Call build_index first.")

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        self._vector_store.save_local(str(save_path))

        metadata_path = save_path / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "model_name": self.model_name,
                "device": self.device,
                "normalize_embeddings": self.normalize_embeddings,
                "num_documents": len(self._documents)
            }, f)

    def load_index(self, path: str) -> None:
        """Load index from disk.

        Args:
            path: Directory path containing saved index
        """
        self._vector_store = LangChainFAISS.load_local(
            path,
            self._embeddings,
            allow_dangerous_deserialization=True
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score (optional)

        Returns:
            List of dicts with 'content', 'metadata', and 'score'
        """
        if self._vector_store is None:
            raise RuntimeError("No index loaded. Call build_index or load_index first.")

        if score_threshold is not None:
            results = self._vector_store.similarity_search_with_relevance_scores(
                query, k=top_k, score_threshold=score_threshold
            )
        else:
            results = self._vector_store.similarity_search_with_score(query, k=top_k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            }
            for doc, score in results
        ]

    def as_langchain_retriever(self, search_kwargs: Optional[Dict] = None):
        """Get LangChain retriever interface.

        Args:
            search_kwargs: Search parameters (k, score_threshold, etc.)

        Returns:
            LangChain retriever object
        """
        if self._vector_store is None:
            raise RuntimeError("No index loaded.")

        kwargs = search_kwargs or {"k": 5}
        return self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=kwargs
        )

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Get the embedding model."""
        return self._embeddings


def build_index_from_texts(
    texts: List[str],
    output_path: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda",
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None
) -> Retriever:
    """Build and save a FAISS index from texts.

    Args:
        texts: List of text passages
        output_path: Directory to save the index
        model_name: Embedding model name
        device: Device for inference
        metadatas: Optional metadata for each passage
        ids: Optional IDs for each passage

    Returns:
        Configured Retriever instance
    """
    retriever = Retriever(model_name=model_name, device=device)
    retriever.build_index(texts, metadatas=metadatas, ids=ids)
    retriever.save_index(output_path)
    return retriever


def build_index_from_documents(
    documents: List[Document],
    output_path: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda"
) -> Retriever:
    """Build and save a FAISS index from LangChain documents.

    Args:
        documents: List of LangChain Document objects
        output_path: Directory to save the index
        model_name: Embedding model name
        device: Device for inference

    Returns:
        Configured Retriever instance
    """
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    ids = [doc.metadata.get("id", str(i)) for i, doc in enumerate(documents)]

    return build_index_from_texts(
        texts=texts,
        output_path=output_path,
        model_name=model_name,
        device=device,
        metadatas=metadatas,
        ids=ids
    )


def load_retriever(
    index_path: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda"
) -> Retriever:
    """Load a retriever from saved index.

    Args:
        index_path: Path to saved index directory
        model_name: Embedding model name (must match saved index)
        device: Device for inference

    Returns:
        Configured Retriever instance
    """
    retriever = Retriever(model_name=model_name, device=device)
    retriever.load_index(index_path)
    return retriever
