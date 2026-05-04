"""RAG configuration space for COMPASS-V.

The 234-config (before constraints) RAG space evaluated in the paper:
    6 generators * 5 retriever_k * 4 reranker_k * 3 reranker_models
    with the constraint reranker_k <= retriever_k.
"""

from typing import Any, Dict, List

from compass.search.parameter_space import NormType, ParameterSpace


# Best-first ordering by expected accuracy (largest / strongest first).
LLM_MODELS: List[str] = [
    "gemma3:12b",
    "llama3.1:8b",
    "gemma3:4b",
    "llama3.2:3b",
    "gemma3:1b",
    "llama3.2:1b",
]

RETRIEVER_K_VALUES: List[int] = [20, 10, 5, 3]
RERANKER_K_VALUES: List[int] = [10, 5, 3, 1]
RERANKER_MODELS: List[str] = ["bge-v2", "bge-base", "ms-marco"]


def _reranker_k_le_retriever_k(config: Dict[str, Any]) -> bool:
    return config["reranker_k"] <= config["retriever_k"]


def load_dataset(path: str = "data/squad_questions.json", n: int = 100) -> List[Dict]:
    """Load the first `n` answerable SQuAD2 questions."""
    import json
    with open(path) as f:
        all_q = json.load(f)
    return [q for q in all_q if not q.get("is_impossible", False)][:n]


def parameter_space() -> ParameterSpace:
    """Construct the RAG parameter space."""
    return ParameterSpace(
        params={
            "generator_model": LLM_MODELS,
            "retriever_k": RETRIEVER_K_VALUES,
            "reranker_k": RERANKER_K_VALUES,
            "reranker_model": RERANKER_MODELS,
        },
        norm_types={
            "generator_model": NormType.CATEGORICAL,
            "retriever_k": NormType.LOG,
            "reranker_k": NormType.LOG,
            "reranker_model": NormType.CATEGORICAL,
        },
        constraints=[_reranker_k_le_retriever_k],
    )
