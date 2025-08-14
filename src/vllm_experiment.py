# src/vllm_experiment.py

import time
import pandas as pd
from pathlib import Path
import qdrant_client
from qdrant_client.http import models
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter,
    SemanticSplitterNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from ragas import evaluate
from datasets import Dataset
import torch
import gc
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any

from src.pipeline import VLLMChunkingPipeline
from src.stats import ChunkingStats
from src.venice_ragas_evaluator import create_venice_evaluator, test_venice_evaluator
from loguru import logger
import structlog

# =====================================================================================
# MERGED CODE FROM analysis.py
# =====================================================================================
@dataclass
class DocumentStats:
    file_name: str
    file_size: int
    chunk_count: int
    processing_time: float
    avg_chunk_size: float

class StatsAnalyzer:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_chunking_performance(self, documents: List, nodes: List, processing_time: float, metadata: List, strategy_name: str, config: Dict[str, Any]) -> ChunkingStats:
        chunk_sizes = [len(node.get_content()) for node in nodes]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        chunks_per_second = len(nodes) / processing_time if processing_time > 0 else 0
        
        return ChunkingStats(
            total_documents=len(documents),
            total_chunks=len(nodes),
            processing_time=processing_time,
            avg_chunk_size=avg_chunk_size,
            min_chunk_size=min(chunk_sizes) if chunk_sizes else 0,
            max_chunk_size=max(chunk_sizes) if chunk_sizes else 0,
            chunks_per_second=chunks_per_second,
            strategy_name=strategy_name,
            embedding_model=config.get('embedding_model', 'Unknown'),
            chunk_size_config=config.get('chunk_size', 0),
            chunk_overlap_config=config.get('chunk_overlap', 0)
        )
    
    def print_console_summary(self, chunking_stats: ChunkingStats):
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE STATISTICS SUMMARY")
        print("=" * 60)
        print(f"Strategy:               {chunking_stats.strategy_name}")
        print(f"Total Chunks:           {chunking_stats.total_chunks}")
        print(f"Processing Time:        {chunking_stats.processing_time:.2f} seconds")
        print(f"Chunks per Second:      {chunking_stats.chunks_per_second:.1f}")
        print(f"Average Chunk Size:     {chunking_stats.avg_chunk_size:.0f} characters")
        print("=" * 60)

    def save_stats_results(self, chunking_stats: ChunkingStats, document_stats: List[DocumentStats], config: Dict[str, Any]):
        base_name = f"stats_{chunking_stats.strategy_name}_{int(time.time())}"
        json_data = {
            'config': config,
            'chunking_stats': chunking_stats.__dict__,
            'document_stats': [doc.__dict__ for doc in document_stats]
        }
        json_file = self.output_dir / f"{base_name}.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"📊 Statistics saved to: {json_file}")

class ResultAnalyzer:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_results = []
    
    def add_evaluation(self, ragas_df: pd.DataFrame):
        if not ragas_df.empty:
            self.all_results.append(ragas_df)
    
    def export_to_spreadsheet(self):
        if not self.all_results:
            return
        combined_df = pd.concat(self.all_results, ignore_index=True)
        excel_file = self.output_dir / f"ragas_results_{int(time.time())}.xlsx"
        combined_df.to_excel(excel_file, sheet_name="All Results", index=False)
        logger.info(f"📊 Exported results to: {excel_file}")

    def generate_markdown_report(self):
        if not self.all_results:
            return
        combined_df = pd.concat(self.all_results, ignore_index=True)
        report_file = self.output_dir / f"ragas_report_{int(time.time())}.md"
        with open(report_file, 'w') as f:
            f.write("# RAGAS Evaluation Report\n\n")
            f.write(combined_df.to_markdown(index=False))
        logger.info(f"📄 Generated report: {report_file}")
# =====================================================================================

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print("🧹 Cleared GPU memory")

class VLLMChunkingExperiment:
    def __init__(self, config: dict):
        self.config = config
        self.llm = self._get_llm()
        self.embed_model = self._get_embed_model()
        self.qdrant_client = self._get_qdrant_client()
        
        self.evaluator_llm = None
        self.evaluator_embeddings = None
        if self.config.get('use_venice_evaluation', False):
            self._initialize_venice_evaluator() 
        
        self.pipeline = VLLMChunkingPipeline(
            self.config,
            embed_model=self.embed_model,
            evaluator_llm=self.evaluator_llm,
            evaluator_embeddings=self.evaluator_embeddings
        )

    def _get_llm(self):
        provider = self.config.get('llm_provider', 'ollama').lower()
        model = self.config.get('llm_model', 'llama3.1:8b')
        logger.info(f"Initializing LOCAL LLM for RAG generation: {provider}")
        if provider == "ollama":
            logger.info(f"Initializing Ollama with model: {model}")
            return Ollama(model=model, request_timeout=120.0)
        return None

    def _initialize_venice_evaluator(self):
        try:
            logger.info("Initializing Venice AI evaluator...")
            api_key = os.getenv('VENICE_API_KEY')
            if not api_key:
                raise ValueError("VENICE_API_KEY not found in environment.")

            if not test_venice_evaluator(api_key=api_key):
                raise ConnectionError("Venice AI evaluator test failed.")
            
            venice_model_name = self.config.get('venice_model', 'llama-3.3-70b')
            self.evaluator_llm, self.evaluator_embeddings = create_venice_evaluator(
                model_name=venice_model_name,
                api_key=api_key,
                local_embeddings=self.embed_model
            )
            logger.info(f"✅ Venice AI evaluator initialized with model: {venice_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Venice AI evaluator: {e}")
            self.config['use_venice_evaluation'] = False
            self.evaluator_llm = None
            self.evaluator_embeddings = None
    
    def _get_embed_model(self):
        embedding_model_name = self.config.get('embedding_model', 'BAAI/bge-m3')
        logger.info(f"Initializing HuggingFace embedding model: {embedding_model_name}")
        clear_gpu_memory()

        device = "cpu"
        logger.info("⚠️ Forcing CPU for embeddings to save VRAM for the LLM.")

        os.makedirs("./cache", exist_ok=True)
        return HuggingFaceEmbedding(
            model_name=embedding_model_name,
            cache_folder="./cache",
            device=device
        )

    def _get_qdrant_client(self):
        host = self.config.get('vector_db_host', 'localhost')
        port = self.config.get('vector_db_port', 6333)
        return qdrant_client.QdrantClient(host=host, port=port)

    def _get_node_parser(self, strategy_name: str):
        chunk_size = self.config.get('chunk_size', 512)
        chunk_overlap = self.config.get('chunk_overlap', 50)
        logger.info(f"Using chunk_size={chunk_size}, overlap={chunk_overlap} for {strategy_name}")
        
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        if strategy_name == "sentence_splitter":
            return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif strategy_name == "token_text_splitter":
            return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif strategy_name == "semantic_splitter":
            return SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embed_model)
        elif strategy_name == "hierarchical":
            return HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
        elif strategy_name == "recursive_character":
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from llama_index.core.node_parser import LangchainNodeParser
            return LangchainNodeParser(RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    def run_full_experiment(self, data_path: str, output_dir: str, strategies_to_test: list, questions: list, ground_truths: list):
        self.print_evaluation_setup()
        documents = self.pipeline.load_documents(data_path)
        
        stats_analyzer = StatsAnalyzer(output_dir)
        result_analyzer = ResultAnalyzer(output_dir)

        for strategy in strategies_to_test:
            logger.info(f"--- Running test for strategy: {strategy} ---")
            
            node_parser = self._get_node_parser(strategy)
            start_time = time.time()
            nodes = node_parser.get_nodes_from_documents(documents)
            if strategy == "hierarchical":
                nodes = get_leaf_nodes(nodes)
            processing_time = time.time() - start_time
            
            chunking_stats = stats_analyzer.analyze_chunking_performance(
                documents=documents, nodes=nodes, processing_time=processing_time,
                metadata=[], strategy_name=strategy, config=self.config
            )
            stats_analyzer.print_console_summary(chunking_stats)
            stats_analyzer.save_stats_results(chunking_stats, [], self.config)

            if self.config.get('mode') == 'stats_only':
                logger.info(f"Stats only mode. Skipping RAG evaluation for {strategy}.")
                continue
            
            collection_name = f"rag_exp_{strategy}_{int(time.time())}"
            vector_store = QdrantVectorStore(client=self.qdrant_client, collection_name=collection_name)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=self.embed_model, show_progress=True)

            # --- MODIFIED SECTION ---
            top_k = self.config.get('similarity_top_k', 5) # Get the value from config
            logger.info(f"Creating query engine with similarity_top_k={top_k}")
            query_engine = index.as_query_engine(
                llm=self.llm, 
                similarity_top_k=top_k
            )
            # --- END MODIFIED SECTION ---
            
            responses, contexts = [], []
            logger.info(f"Generating answers for {len(questions)} questions...")
            for q in questions:
                response = query_engine.query(q)
                responses.append(str(response))
                contexts.append([c.node.get_content() for c in response.source_nodes])
            
            eval_data = {"question": questions, "answer": responses, "contexts": contexts, "ground_truth": ground_truths}
            dataset = Dataset.from_dict(eval_data)
            
            ragas_df = self.pipeline.evaluate_with_ragas(dataset)
            
            if not ragas_df.empty:
                ragas_df['strategy'] = strategy # Add strategy for better analysis
                result_analyzer.add_evaluation(ragas_df)
                logger.info(f"RAGAS results for '{strategy}' processed successfully.")
                for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                    if col in ragas_df.columns:
                        mean_score = ragas_df[col].dropna().mean()
                        logger.info(f"  Average {col}: {mean_score:.4f}")
            else:
                logger.error(f"RAGAS evaluation for '{strategy}' produced no results.")

            # Cleanup
            self.qdrant_client.delete_collection(collection_name=collection_name)
            logger.info(f"Cleaned up Qdrant collection: {collection_name}")
            clear_gpu_memory()

        if self.config.get('mode') != 'stats_only':
            result_analyzer.export_to_spreadsheet()
            result_analyzer.generate_markdown_report()

    def print_evaluation_setup(self):
        print("\n" + "="*60)
        print("🔧 EVALUATION SETUP")
        print("="*60)
        print(f"RAG Generation LLM:   {self.config.get('llm_provider')} ({self.config.get('llm_model')})")
        print(f"Embedding Model:      {self.config.get('embedding_model')}")
        print(f"Similarity Top K:     {self.config.get('similarity_top_k', 'Default')}")
        if self.config.get('use_venice_evaluation'):
            print(f"RAGAS Evaluation LLM: Venice AI ({self.config.get('venice_model')})")
        else:
              print(f"RAGAS Evaluation LLM: Same as RAG Generation LLM")
        print("="*60)