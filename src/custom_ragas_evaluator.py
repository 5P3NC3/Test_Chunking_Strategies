# src/custom_ragas_evaluator.py
"""
Custom evaluation module for RAGAs using LlamaIndex components.
This version uses adapter classes for compatibility with Ragas.
"""
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from llama_index.core.llms import LLM as LlamaIndexLLMInterface
from llama_index.core.embeddings import BaseEmbedding as LlamaIndexEmbeddingInterface
import structlog

# Import the new adapter classes
from .ragas_adapters import RagasLlamaIndexLLM, RagasLlamaIndexEmbeddings

logger = structlog.get_logger()

def custom_evaluate(dataset, llm: LlamaIndexLLMInterface, embed_model: LlamaIndexEmbeddingInterface):
    """
    Custom evaluation function that wraps LlamaIndex components in Ragas-compatible
    adapters before passing them to the evaluation library.
    """
    logger.info("Adapting LlamaIndex components for Ragas evaluation...")
    
    # Wrap the LlamaIndex components with our Ragas adapters
    ragas_llm = RagasLlamaIndexLLM(llm=llm)
    ragas_embeddings = RagasLlamaIndexEmbeddings(embeddings=embed_model)

    metrics_to_evaluate = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    logger.info("Starting Ragas evaluation with adapted components...")
    try:
        # Pass the wrapped objects to the evaluate function
        result = evaluate(
            dataset,
            metrics=metrics_to_evaluate,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False,  # Log errors from ragas instead of crashing
        )
        
        logger.info("✅ Ragas evaluation completed.")
        return result
    except Exception as e:
        logger.error(f"Ragas evaluation failed catastrophically: {e}", exc_info=True)
        return None