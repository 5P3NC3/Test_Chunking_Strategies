# src/ragas_adapters.py (Corrected Code)

from __future__ import annotations
import asyncio
from typing import Any, List

from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.run_config import RunConfig

from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from langchain_core.outputs import Generation, LLMResult

def _to_string(prompt: Any) -> str:
    """Converts a prompt object to a string if it has the method."""
    if hasattr(prompt, 'to_string'):
        return prompt.to_string()
    return str(prompt)

class RagasLlamaIndexLLM(BaseRagasLLM):
    """
    Adapter class to wrap a LlamaIndex LLM for use with Ragas.
    """
    def __init__(self, llm: LLM):
        self.llm = llm
        self.run_config = RunConfig()

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

    async def agenerate_text(self, prompt: Any, **kwargs) -> str:
        """Asynchronously generates text from a single prompt."""
        prompt_str = _to_string(prompt)
        result = await self.llm.acomplete(prompt_str)
        return result.text

    # --- FIX: Re-implementing the required abstract method ---
    def generate_text(self, prompt: Any, **kwargs) -> str:
        """Synchronously generates text from a single prompt."""
        prompt_str = _to_string(prompt)
        result = self.llm.complete(prompt_str)
        return result.text

    async def generate(self, prompt: Any, n: int = 1, **kwargs) -> LLMResult:
        """Asynchronously generates text and wraps it in the LLMResult format."""
        tasks = [self.agenerate_text(prompt, **kwargs) for _ in range(n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        generations = []
        for res in results:
            if isinstance(res, Exception):
                generations.append([Generation(text=f"Error: {str(res)}")])
            else:
                generations.append([Generation(text=str(res))])
                
        return LLMResult(generations=generations)

class RagasLlamaIndexEmbeddings(BaseRagasEmbeddings):
    """
    Adapter class to wrap a LlamaIndex Embedding model for use with Ragas.
    """
    def __init__(self, embeddings: BaseEmbedding):
        self.embeddings = embeddings
        self.run_config = RunConfig()

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        if not text or not text.strip():
            text = "empty"
        return self.embeddings.get_text_embedding(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        clean_texts = [t if t and t.strip() else "empty" for t in texts]
        return self.embeddings.get_text_embedding_batch(clean_texts, show_progress=False)