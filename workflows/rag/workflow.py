from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from collections import deque

from workflows.rag.generator import Generator
from workflows.rag.retriever import Retriever
from workflows.rag.reranker import Reranker


class Component(ABC):
    component_type: str = "base"

    @abstractmethod
    def configure(self, model_id: str, **hyperparameters) -> None:
        pass

    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass


class GeneratorComponent(Component):
    component_type = "generator"

    def __init__(self, base_url: str = "http://localhost:11434", keep_models_loaded: bool = False):
        self.base_url = base_url
        self.model_id: Optional[str] = None
        self.hyperparameters: Dict[str, Any] = {}
        self._generator: Optional[Generator] = None
        self._current_model_id: Optional[str] = None
        self.keep_models_loaded = keep_models_loaded  # If True, don't unload models on switch

    def _unload_ollama_model(self, model_name: str) -> None:
        """Unload a model from Ollama to free GPU memory."""
        import requests
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=5
            )
        except Exception:
            pass  # Best effort

    def configure(self, model_id: str, **hyperparameters) -> None:
        # Unload previous model if switching (unless keep_models_loaded is True)
        if (not self.keep_models_loaded and
            self._current_model_id is not None and
            self._current_model_id != model_id):
            self._unload_ollama_model(self._current_model_id)

        self.model_id = model_id
        self._current_model_id = model_id
        self.hyperparameters = hyperparameters
        self._generator = Generator(
            model_name=model_id,
            base_url=self.base_url,
            temperature=hyperparameters.get("temperature", 0.0),
            max_tokens=hyperparameters.get("max_tokens", 50),
            prompt_template=hyperparameters.get("prompt_template")
        )

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._generator is None:
            raise RuntimeError("Component not configured")
        question = inputs.get("question", "")
        context = inputs.get("context", "")
        answer = self._generator.generate(question=question, context=context)
        return {"answer": answer}

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "hyperparameters": self.hyperparameters.copy()
        }


class RetrieverComponent(Component):
    component_type = "retriever"

    def __init__(self, index_path: str,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.index_path = index_path
        self.model_name = model_name
        self._retriever: Optional[Retriever] = None
        self.top_k = 5
        self.hyperparameters: Dict[str, Any] = {}

    def configure(self, model_id: str = None, **hyperparameters) -> None:
        self.top_k = hyperparameters.get("retriever_k", 5)
        self.hyperparameters = hyperparameters
        if self._retriever is None:
            self._retriever = Retriever(model_name=self.model_name)
            self._retriever.load_index(self.index_path)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._retriever is None:
            raise RuntimeError("Component not configured")
        # k=0 means skip retrieval (for influence estimation)
        if self.top_k == 0:
            return {"documents": [], "context": ""}
        question = inputs.get("question", "")
        results = self._retriever.retrieve(question, top_k=self.top_k)
        return {
            "documents": results,
            "context": "\n\n".join([r["content"] for r in results])
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_name,
            "hyperparameters": self.hyperparameters.copy()
        }


class RerankerComponent(Component):
    component_type = "reranker"

    # Mapping from short names to HuggingFace model names
    MODEL_MAPPING = {
        "ms-marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "bge-base": "BAAI/bge-reranker-base",
        "bge-v2": "BAAI/bge-reranker-v2-m3",
    }

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", keep_models_loaded: bool = False):
        self.model_name = model_name
        self._reranker: Optional[Reranker] = None
        self._current_model_name: Optional[str] = None
        self.top_k = 5
        self.hyperparameters: Dict[str, Any] = {}
        self.keep_models_loaded = keep_models_loaded  # If True, don't unload models on switch

    def configure(self, model_id: str = None, **hyperparameters) -> None:
        self.top_k = hyperparameters.get("reranker_k", 5)
        self.hyperparameters = hyperparameters

        # Handle reranker_model parameter
        reranker_model = hyperparameters.get("reranker_model", "bge-v2")
        target_model_name = self.MODEL_MAPPING.get(reranker_model, reranker_model)

        # Only recreate reranker if model changed
        if self._reranker is None or self._current_model_name != target_model_name:
            # Free old model first to avoid CUDA OOM (unless keep_models_loaded)
            if self._reranker is not None and not self.keep_models_loaded:
                del self._reranker
                self._reranker = None
                # Force garbage collection and clear CUDA cache
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            self._reranker = Reranker(model_name=target_model_name)
            self._current_model_name = target_model_name

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._reranker is None:
            raise RuntimeError("Component not configured")
        # k=0 means skip reranking - pass through documents unchanged
        if self.top_k == 0:
            docs = inputs.get("documents", [])
            return {
                "documents": docs,
                "context": "\n\n".join([d["content"] for d in docs])
            }
        question = inputs.get("question", "")
        documents = inputs.get("documents", [])
        if not documents:
            return {"documents": [], "context": ""}
        reranked = self._reranker.rerank_with_metadata(
            question, documents, content_key="content", top_k=self.top_k
        )
        return {
            "documents": reranked,
            "context": "\n\n".join([r["content"] for r in reranked])
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_name,
            "hyperparameters": self.hyperparameters.copy()
        }


class Node:
    def __init__(self, name: str, component: Component,
                 input_mapping: Optional[Dict[str, str]] = None):
        self.name = name
        self.component = component
        self.input_mapping = input_mapping or {}


class Workflow:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[tuple] = []
        self._adjacency: Dict[str, List[str]] = {}
        self._reverse_adjacency: Dict[str, List[str]] = {}

    def add_node(self, name: str, component: Component,
                 input_mapping: Optional[Dict[str, str]] = None) -> None:
        self.nodes[name] = Node(name, component, input_mapping)
        if name not in self._adjacency:
            self._adjacency[name] = []
        if name not in self._reverse_adjacency:
            self._reverse_adjacency[name] = []

    def add_edge(self, from_node: str, to_node: str) -> None:
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError(f"Invalid edge: {from_node} -> {to_node}")
        self.edges.append((from_node, to_node))
        self._adjacency[from_node].append(to_node)
        self._reverse_adjacency[to_node].append(from_node)

    def get_execution_order(self) -> List[str]:
        in_degree = {name: 0 for name in self.nodes}
        for _, to_node in self.edges:
            in_degree[to_node] += 1

        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self._adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            raise ValueError("Workflow contains a cycle")
        return order

    def configure(self, config: "Configuration") -> None:
        from optimizer import Configuration
        for node_name, comp_config in config.component_configs.items():
            if node_name in self.nodes:
                self.nodes[node_name].component.configure(
                    model_id=comp_config.model_id,
                    **comp_config.hyperparameters
                )

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        order = self.get_execution_order()
        node_outputs: Dict[str, Dict[str, Any]] = {}

        for node_name in order:
            node = self.nodes[node_name]
            node_inputs = inputs.copy()

            for input_key, source in node.input_mapping.items():
                if "." in source:
                    src_node, src_key = source.split(".", 1)
                    if src_node in node_outputs:
                        node_inputs[input_key] = node_outputs[src_node].get(src_key)
                else:
                    node_inputs[input_key] = inputs.get(source)

            node_outputs[node_name] = node.component.execute(node_inputs)

        if order:
            return node_outputs[order[-1]]
        return {}

    def get_node_names(self) -> List[str]:
        return list(self.nodes.keys())
