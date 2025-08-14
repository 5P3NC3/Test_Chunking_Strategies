import asyncio
from typing import List, Any, Dict, Optional, Union

import torch
import structlog
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult, Generation

logger = structlog.get_logger()

class UltimateRAGASLLM(BaseLLM):
    """
    A RAGAS-compatible LLM wrapper that inherits from LangChain's BaseLLM
    for full compatibility with the evaluation framework.
    """
    
    model_name: str
    device: str = "cpu"
    max_tokens: int = 256
    
    _model: Any = None
    _tokenizer: Any = None
    run_config: Dict[str, Any] = {}

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        logger.info(f"🔧 FIXED VERSION - Initializing Ultimate RAGAS LLM: {self.model_name} on {self.device}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            padding_side="left"
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        if self.device == "cpu":
            self._model.to(self.device)
        self._model.eval()
        logger.info("✅ FIXED VERSION - Ultimate RAGAS LLM initialized successfully")

    def set_run_config(self, run_config: Dict[str, Any]) -> None:
        """A simple method to satisfy the Ragas framework."""
        self.run_config = run_config

    def _extract_prompt_text(self, prompt: Any) -> str:
        """
        Extract text from prompt, handling various types.
        """
        logger.debug(f"DIAGNOSTIC: Prompt type: {type(prompt)}")
        
        if isinstance(prompt, str):
            return prompt
        elif hasattr(prompt, 'to_string'):
            result = prompt.to_string()
            logger.debug(f"DIAGNOSTIC: Used to_string(), result: {result[:100]}...")
            return result
        elif hasattr(prompt, 'text'):
            result = prompt.text
            logger.debug(f"DIAGNOSTIC: Used .text attribute, result: {result[:100]}...")
            return result
        else:
            result = str(prompt)
            logger.debug(f"DIAGNOSTIC: Used str() conversion, result: {result[:100]}...")
            return result

    def _generate_text(self, prompt: Any) -> str:
        """Internal synchronous method to generate text for a single prompt."""
        try:
            prompt_text = self._extract_prompt_text(prompt)
            logger.debug(f"DIAGNOSTIC: Processing prompt of length: {len(prompt_text)}")
            
            inputs = self._tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask', None),
                    max_new_tokens=self.max_tokens,
                    pad_token_id=self._tokenizer.pad_token_id,
                    do_sample=False,
                    temperature=1.0,
                )
            
            generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
            result = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            logger.debug(f"DIAGNOSTIC: Generated response of length: {len(result)}")
            return result

        except Exception as e:
            logger.error(f"DIAGNOSTIC: Generation error for prompt: {e}", exc_info=True)
            return "Error: Could not generate response"

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "ultimate_ragas_llm"

    def _generate(
        self, prompts: List[Any], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LLMResult:
        """
        The synchronous method for generating responses.
        """
        logger.info(f"DIAGNOSTIC: _generate called with {len(prompts)} prompts")
        logger.info(f"DIAGNOSTIC: First prompt type: {type(prompts[0]) if prompts else 'None'}")
        
        generations = []
        for i, prompt in enumerate(prompts):
            try:
                logger.debug(f"DIAGNOSTIC: Processing prompt {i+1}/{len(prompts)}, type: {type(prompt)}")
                text = self._generate_text(prompt)
                generations.append([Generation(text=text)])
                logger.debug(f"DIAGNOSTIC: Generated response {i+1}/{len(prompts)}")
            except Exception as e:
                logger.error(f"DIAGNOSTIC: Error generating response for prompt {i+1}: {e}")
                generations.append([Generation(text=f"Error: {str(e)}")])
        
        logger.info(f"DIAGNOSTIC: _generate completed, returning {len(generations)} generations")
        return LLMResult(generations=generations)

    async def _agenerate(
        self, prompts: List[Any], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LLMResult:
        """
        The asynchronous method for generating responses.
        """
        logger.info(f"DIAGNOSTIC: _agenerate called with {len(prompts)} prompts")
        logger.info(f"DIAGNOSTIC: First prompt type: {type(prompts[0]) if prompts else 'None'}")
        
        loop = asyncio.get_running_loop()
        
        tasks = [
            loop.run_in_executor(None, self._generate_text, prompt) 
            for prompt in prompts
        ]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            generations = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"DIAGNOSTIC: Async generation error for prompt {i+1}: {result}")
                    generations.append([Generation(text=f"Error: {str(result)}")])
                else:
                    generations.append([Generation(text=result)])
                    
            logger.info(f"DIAGNOSTIC: _agenerate completed, returning {len(generations)} generations")
            return LLMResult(generations=generations)
            
        except Exception as e:
            logger.error(f"DIAGNOSTIC: Async generation batch error: {e}")
            error_generations = [
                [Generation(text=f"Batch error: {str(e)}")] 
                for _ in prompts
            ]
            return LLMResult(generations=error_generations)


class RAGASEmbeddingWrapper:
    """
    Wrapper for embedding models to work with RAGAS evaluation.
    """
    def __init__(self, base_embedding):
        self.base_embedding = base_embedding
        self.run_config = {}
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        clean_texts = [
            str(t).strip()[:1000] if t and str(t).strip() else "empty" 
            for t in texts
        ]
        return self.base_embedding.get_text_embedding_batch(
            clean_texts, 
            show_progress=False
        )
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        clean_text = str(text).strip()[:500] if text and text.strip() else "empty"
        return self.base_embedding.get_text_embedding(clean_text)
    
    def set_run_config(self, run_config: Dict[str, Any]) -> None:
        """Set run configuration."""
        self.run_config = run_config