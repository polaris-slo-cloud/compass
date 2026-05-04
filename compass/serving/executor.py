"""Workflow executor: RAG pipeline driver for the serving layer.

The executor wraps a configurable workflow so the controller can switch its
parameters without rebuilding the DAG. The default builds the RAG pipeline
from `workflows.rag`; supply a custom builder for other workflows.
"""

from typing import Any, Callable, Dict, Optional

from workflows.rag.generator import PROMPTS
from workflows.rag.workflow import (
    GeneratorComponent,
    RerankerComponent,
    RetrieverComponent,
    Workflow,
)


def _build_rag_workflow(index_path: str, keep_models_loaded: bool) -> Workflow:
    workflow = Workflow()
    workflow.add_node("retriever", RetrieverComponent(index_path=index_path))
    workflow.add_node(
        "reranker",
        RerankerComponent(keep_models_loaded=keep_models_loaded),
        input_mapping={"documents": "retriever.documents", "question": "question"},
    )
    workflow.add_node(
        "generator",
        GeneratorComponent(keep_models_loaded=keep_models_loaded),
        input_mapping={"context": "reranker.context", "question": "question"},
    )
    workflow.add_edge("retriever", "reranker")
    workflow.add_edge("reranker", "generator")
    return workflow


class WorkflowExecutor:
    """Configurable RAG workflow executor."""

    def __init__(
        self,
        index_path: str = "data/squad_index",
        keep_models_loaded: bool = False,
        builder: Optional[Callable[[str, bool], Workflow]] = None,
    ):
        self.keep_models_loaded = keep_models_loaded
        self.workflow = (builder or _build_rag_workflow)(index_path, keep_models_loaded)
        self.prompt = PROMPTS["qa_short"]
        self._current_config: Dict[str, Any] = {}

    def configure(self, config: Dict[str, Any]) -> None:
        if config == self._current_config:
            return
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
        self._current_config = config.copy()

    def execute(self, question: str) -> str:
        try:
            result = self.workflow.execute({"question": question, "context": ""})
            return result.get("answer", "")
        except Exception as e:
            print(f"Execution error: {e}")
            return ""
