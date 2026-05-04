"""Modular generator component with LangChain integration."""

from typing import List, Dict, Any, Optional, Union

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel


class Generator:
    """LLM generator using Ollama or other LangChain-compatible models."""

    DEFAULT_TEMPLATE = """Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        prompt_template: Optional[str] = None
    ):
        """Initialize generator with LLM model.

        Args:
            model_name: Ollama model name
            base_url: Ollama server URL
            temperature: Generation temperature (0 = deterministic)
            max_tokens: Maximum tokens to generate
            prompt_template: Custom prompt template with {context} and {question}
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens
        )

        template = prompt_template or self.DEFAULT_TEMPLATE
        self._prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def generate(
        self,
        question: str,
        context: Union[str, List[str]],
        prompt_template: Optional[str] = None
    ) -> str:
        """Generate answer from question and context.

        Args:
            question: User question
            context: Context string or list of passages
            prompt_template: Override prompt template for this call

        Returns:
            Generated answer string
        """
        if isinstance(context, list):
            context = "\n\n".join(
                f"[{i+1}] {passage}" for i, passage in enumerate(context)
            )

        if prompt_template:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        else:
            prompt = self._prompt

        formatted = prompt.format(context=context, question=question)
        response = self._llm.invoke(formatted)
        return response.content.strip()

    def generate_with_metadata(
        self,
        question: str,
        context: Union[str, List[str]],
        prompt_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate answer with additional metadata.

        Args:
            question: User question
            context: Context string or list of passages
            prompt_template: Override prompt template

        Returns:
            Dict with 'answer', 'prompt', 'model'
        """
        if isinstance(context, list):
            context_str = "\n\n".join(
                f"[{i+1}] {passage}" for i, passage in enumerate(context)
            )
        else:
            context_str = context

        if prompt_template:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        else:
            prompt = self._prompt

        formatted = prompt.format(context=context_str, question=question)
        response = self._llm.invoke(formatted)

        return {
            "answer": response.content.strip(),
            "prompt": formatted,
            "model": self.model_name,
            "context_length": len(context_str),
            "response_metadata": getattr(response, "response_metadata", {})
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Chat with the model using message history.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}
            system_prompt: Optional system message

        Returns:
            Generated response string
        """
        lc_messages = []

        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))

        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            else:
                from langchain_core.messages import AIMessage
                lc_messages.append(AIMessage(content=msg["content"]))

        response = self._llm.invoke(lc_messages)
        return response.content.strip()

    def swap_model(self, model_name: str) -> None:
        """Hot-swap to a different model.

        Args:
            model_name: New Ollama model name
        """
        self.model_name = model_name
        self._llm = ChatOllama(
            model=model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=self.max_tokens,
            reasoning=False
        )

    def set_prompt_template(self, template: str) -> None:
        """Update the default prompt template.

        Args:
            template: New template with {context} and {question} placeholders
        """
        self._prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    @property
    def llm(self) -> BaseChatModel:
        """Get the underlying LangChain LLM."""
        return self._llm


def create_generator(
    model_name: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    prompt_template: Optional[str] = None
) -> Generator:
    """Factory function to create a generator.

    Args:
        model_name: Ollama model name
        base_url: Ollama server URL
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        prompt_template: Custom prompt template

    Returns:
        Configured Generator instance
    """
    return Generator(
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_template=prompt_template
    )


# Prompt templates for common use cases
PROMPTS = {
    "qa": """Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:""",

    "qa_short": """Answer the question using only the context below.

IMPORTANT:
- Give ONLY the exact answer, nothing else
- Use 1-5 words maximum
- No explanations, no full sentences

Context:
{context}

Question: {question}

Answer:""",

    "qa_detailed": """You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Provide a detailed answer with explanations:""",

    "summarize": """Summarize the following context to answer the question.

Context:
{context}

Question: {question}

Summary:""",
}


def get_generator(
    model_name: str = "llama3.1:8b",
    prompt_preset: str = "qa",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
) -> Generator:
    """Get a generator with a preset prompt template.

    Args:
        model_name: Ollama model name
        prompt_preset: Prompt preset (qa, qa_short, qa_detailed, summarize)
        base_url: Ollama server URL
        temperature: Generation temperature

    Returns:
        Configured Generator instance
    """
    if prompt_preset not in PROMPTS:
        raise ValueError(f"Unknown preset: {prompt_preset}. Available: {list(PROMPTS.keys())}")

    return create_generator(
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
        prompt_template=PROMPTS[prompt_preset]
    )
