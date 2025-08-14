import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from ragas import evaluate
# --- FIX: Import RunConfig ---
from ragas.run_config import RunConfig
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from datasets import Dataset
from loguru import logger

from src.stats import ChunkingStats
from src.venice_ragas_evaluator import create_venice_evaluator
from src.ragas_adapters import RagasLlamaIndexLLM, RagasLlamaIndexEmbeddings
from src.custom_pptx_reader import CustomPptxReader



load_dotenv()

class VLLMChunkingPipeline:
    def __init__(self, config, embed_model=None, evaluator_llm=None, evaluator_embeddings=None):
        self.config = config
        self.llm = self._get_llm()
        self.embed_model = embed_model
        self.evaluator_llm = evaluator_llm
        self.evaluator_embeddings = evaluator_embeddings
        
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    def _get_llm(self):
        provider = os.getenv("LLM_PROVIDER", self.config.get('llm_provider', 'ollama')).lower()
        logger.info(f"Initializing LOCAL LLM for RAG generation: {provider}")
        if provider == "ollama":
            llm = Ollama(model=self.config.get('llm_model'), request_timeout=300.0)
        elif provider == "venice":
            api_key = os.getenv('VENICE_API_KEY')
            if not api_key: raise ValueError("Venice AI API key not found")
            llm = OpenAI(
                model=self.config.get('llm_model'),
                api_base="https://api.venice.ai/api/v1",
                api_key=api_key,
                temperature=0.7,
                max_tokens=1024,
                timeout=120.0,
            )
        else:
            raise NotImplementedError(f"LLM provider '{provider}' is not yet supported.")
        logger.info(f"✅ LOCAL LLM '{self.config.get('llm_model')}' initialized for RAG generation.")
        return llm

    def load_documents(self, data_path: str) -> list:
        """Load documents from the specified path."""
        logger.info(f"Loading documents from {data_path}...")
        
        # Check document type from path
        doc_type = data_path.split('/')[-1]  # Gets 'pdf', 'docx', 'pptx', 'txt', etc.
        
        # For PPTX files, use a workaround until PyTorch is upgraded
        if doc_type == 'pptx':
            try:
                # Try the unstructured reader first if available
                from llama_index.readers.unstructured import UnstructuredReader
                reader = SimpleDirectoryReader(
                    input_dir=data_path,
                    file_extractor={".pptx": UnstructuredReader()},
                    recursive=True
                )
                logger.info("Using UnstructuredReader for PPTX files")
            except ImportError:
                # Fallback: Skip image extraction, just get text
                logger.warning("UnstructuredReader not available. Using basic text extraction for PPTX.")
                # This will extract text but not process images
                from llama_index.core.readers.file.base import DEFAULT_FILE_READER_CLS
                # Remove the problematic PptxReader
                custom_readers = DEFAULT_FILE_READER_CLS.copy()
                if '.pptx' in custom_readers:
                    del custom_readers['.pptx']
                
                reader = SimpleDirectoryReader(
                    input_dir=data_path,
                    recursive=True,
                    file_extractor=custom_readers
                )
        else:
            # For all other file types (PDF, DOCX, TXT, etc.), use default readers
            reader = SimpleDirectoryReader(input_dir=data_path, recursive=True)
        
        documents = reader.load_data()
        
        # Log what was loaded
        if documents:
            file_types = {}
            for doc in documents:
                ext = Path(doc.metadata.get('file_name', '')).suffix
                file_types[ext] = file_types.get(ext, 0) + 1
            logger.info(f"Loaded {len(documents)} documents: {file_types}")
        else:
            logger.warning("No documents loaded!")
        
        return documents

    def evaluate_with_ragas(self, dataset: Dataset) -> pd.DataFrame:
        """
        Evaluate with RAGAS and always return a pandas DataFrame for stability.
        """
        logger.info("Starting RAGAS evaluation...")
        
        # Determine which LLM and embeddings to use for evaluation
        if self.evaluator_llm and self.config.get('use_venice_evaluation', False):
            logger.info("Using Venice AI for RAGAS evaluation - high quality results expected")
            ragas_llm = self.evaluator_llm
            ragas_embeddings = self.evaluator_embeddings
            run_config = RunConfig(max_workers=1, max_wait=30, max_retries=3, timeout=120)
        else:
            logger.warning("Using local LLM for RAGAS evaluation - results may be less accurate")
            ragas_llm = RagasLlamaIndexLLM(llm=self.llm)
            ragas_embeddings = RagasLlamaIndexEmbeddings(embeddings=self.embed_model) if self.embed_model is not None else None
            run_config = RunConfig(max_workers=8, max_wait=60, max_retries=3, timeout=120)

        logger.info(f"Evaluating {len(dataset)} samples with max_workers={run_config.max_workers}")

        # --- FIX: Convert result to DataFrame regardless of success or partial failure ---
        try:
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                raise_exceptions=False,
                run_config=run_config,
            )
            logger.info("✅ RAGAS evaluation complete.")
            # The result object can be a Dataset or a Result object.
            # to_pandas() is the most reliable way to handle both cases.
            df = result.to_pandas()
            if isinstance(df, pd.DataFrame):
                return df
            else:
                # If it's an iterator of DataFrames, concatenate
                return pd.concat(list(df), ignore_index=True)
        except Exception as e:
            logger.error(f"A critical error occurred during RAGAS evaluation: {e}", exc_info=True)
            return pd.DataFrame() # Return an empty DataFrame on critical failure