# src/venice_ragas_evaluator.py

import os
import json
import time
import asyncio
import requests
from typing import Optional, List, Any

from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.run_config import RunConfig
from langchain_core.outputs import Generation, LLMResult
from llama_index.core.embeddings import BaseEmbedding
from loguru import logger

def extract_json_from_text(text: str) -> str:
    """
    A more robust helper function to find and extract a JSON object from a string,
    including those wrapped in markdown or with missing list brackets.
    """
    text = text.strip()
    
    # Handle markdown code blocks
    if text.startswith("```") and text.endswith("```"):
        text = text[text.find('{'):text.rfind('}')+1] if '{' in text else text[text.find('['):text.rfind(']')+1]

    # First, try to parse the text as-is
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        # If it fails, check if it's a series of objects needing brackets
        if text.startswith("{") and text.endswith("}"):
            try:
                # Wrap the text in brackets to form a JSON array
                repaired_json = f"[{text}]"
                json.loads(repaired_json)
                logger.warning("Repaired a malformed JSON list by adding brackets.")
                return repaired_json
            except json.JSONDecodeError:
                # If even that fails, the JSON is truly malformed
                pass
    
    logger.warning(f"Could not parse valid JSON from response: {text}")
    return "{}"


class VeniceRagasLLM(BaseRagasLLM):
    """
    A robust, custom LLM wrapper for Venice AI that handles asynchronous
    requests correctly for RAGAS evaluation.
    """
    def __init__(self, model_name: str = "llama-3.3-70b", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv('VENICE_API_KEY')
        self.api_base = "https://api.venice.ai/api/v1/chat/completions"
        if not self.api_key:
            raise ValueError("Venice AI API key not found.")
        self.run_config = RunConfig()

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

    def _generate_text_sync(self, prompt: Any) -> str:
        """The core synchronous method that makes the API call."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that only responds in valid JSON format."},
            {"role": "user", "content": str(prompt)}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        
        retries = getattr(self.run_config, 'max_retries', 3)
        timeout = getattr(self.run_config, 'timeout', 120)
        wait_time = getattr(self.run_config, 'max_wait', 5)

        for attempt in range(retries):
            try:
                response = requests.post(
                    self.api_base,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content']
                return extract_json_from_text(content)
            except requests.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
                time.sleep(wait_time)
        
        raise RuntimeError("All retries failed for Venice AI API call.")

    async def agenerate_text(self, prompt: Any, **kwargs) -> str:
        """Asynchronously runs the synchronous API call in a separate thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._generate_text_sync, prompt
        )
    
    def generate_text(self, prompt: Any, **kwargs) -> str:
        """Required synchronous method."""
        return self._generate_text_sync(prompt)

    async def generate(self, prompt: Any, n: int = 1, **kwargs) -> LLMResult:
        """The main async method called by Ragas."""
        tasks = [self.agenerate_text(prompt, **kwargs) for _ in range(n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        generations = [[Generation(text=str(r))] for r in results]
        return LLMResult(generations=generations)


class RagasVeniceEmbeddings(BaseRagasEmbeddings):
    """A wrapper for using LlamaIndex embeddings with Ragas."""
    def __init__(self, embeddings: BaseEmbedding):
        self.embeddings = embeddings
        self.run_config = RunConfig()

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.get_text_embedding(text or " ")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.get_text_embedding_batch(
            [t or " " for t in texts], show_progress=False
        )


def create_venice_evaluator(
    model_name: str = "llama-3.3-70b",
    api_key: Optional[str] = None,
    local_embeddings: Optional[BaseEmbedding] = None,
):
    """Creates a Ragas-compatible evaluator using the custom Venice AI LLM."""
    llm = VeniceRagasLLM(model_name=model_name, api_key=api_key)
    embeddings = RagasVeniceEmbeddings(embeddings=local_embeddings)
    return llm, embeddings


def test_venice_evaluator(api_key: Optional[str] = None) -> bool:
    """Tests the Venice AI connection."""
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        logger.info("Testing Venice AI connection...")
        
        dummy_embeds = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        test_llm, _ = create_venice_evaluator(
            api_key=api_key, local_embeddings=dummy_embeds
        )
        response_str = test_llm.generate_text("Is the sky blue? Answer in valid JSON format.")
        response_json = json.loads(response_str)

        logger.info(f"✅ Venice AI test successful. Response: {response_json}")
        return True
    except Exception as e:
        logger.error(f"Venice AI test failed: {e}", exc_info=True)
        return False