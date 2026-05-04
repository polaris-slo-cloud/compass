"""RAG evaluator: F1 over SQuAD questions for COMPASS-V."""

from typing import Any, Callable, Dict, List, Optional

from compass.search.evaluator import Evaluator
from workflows.rag.generator import PROMPTS
from workflows.rag.utils import compute_f1
from workflows.rag.workflow import (
    GeneratorComponent,
    RerankerComponent,
    RetrieverComponent,
    Workflow,
)


def _build_workflow(index_path: str) -> Workflow:
    workflow = Workflow()
    workflow.add_node("retriever", RetrieverComponent(index_path=index_path))
    workflow.add_node(
        "reranker",
        RerankerComponent(),
        input_mapping={"documents": "retriever.documents", "question": "question"},
    )
    workflow.add_node(
        "generator",
        GeneratorComponent(),
        input_mapping={"context": "reranker.context", "question": "question"},
    )
    workflow.add_edge("retriever", "reranker")
    workflow.add_edge("reranker", "generator")
    return workflow


class RagEvaluator(Evaluator):
    """Evaluates RAG configurations against SQuAD questions using F1."""

    def __init__(
        self,
        dataset: List[Dict],
        index_path: str = "data/squad_index",
        metric_fn: Optional[Callable[[str, List[str]], float]] = None,
    ):
        self.dataset = dataset
        self.metric_fn = metric_fn or (
            lambda pred, truths: max(compute_f1(pred, gt) for gt in truths)
        )
        self.prompt = PROMPTS["qa_short"]
        self.workflow = _build_workflow(index_path)

    @property
    def n_samples(self) -> int:
        return len(self.dataset)

    def _configure(self, config: Dict[str, Any]) -> None:
        for name, node in self.workflow.nodes.items():
            if name == "retriever":
                node.component.configure(retriever_k=config.get("retriever_k", 5))
            elif name == "reranker":
                node.component.configure(
                    reranker_k=config.get("reranker_k", 5),
                    reranker_model=config.get("reranker_model", "ms-marco"),
                )
            elif name == "generator":
                node.component.configure(
                    model_id=config.get("generator_model", "llama3.1:8b"),
                    temperature=config.get("generator_temperature", 0.0),
                    max_tokens=config.get("generator_max_tokens", 100),
                    prompt_template=self.prompt,
                )

    def _score_one(self, sample: Dict) -> float:
        try:
            result = self.workflow.execute({"question": sample["question"], "context": ""})
            pred = result.get("answer", "")
        except Exception:
            pred = ""
        return self.metric_fn(pred, sample["answers"])

    def evaluate(self, config: Dict[str, Any]) -> float:
        self._configure(config)
        scores = [self._score_one(s) for s in self.dataset]
        return sum(scores) / len(scores) if scores else 0.0

    def evaluate_partial(
        self, config: Dict[str, Any], indices: List[int]
    ) -> List[float]:
        self._configure(config)
        return [self._score_one(self.dataset[i % len(self.dataset)]) for i in indices]
