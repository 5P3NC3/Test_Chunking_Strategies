"""
Results Analysis Module for RAG Experiments
Handles both statistics analysis and RAGAS result compilation
"""

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

@dataclass
class ChunkingStats:
    """Statistics for chunking performance"""
    total_documents: int
    total_chunks: int
    processing_time: float
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    chunks_per_second: float
    strategy_name: str
    embedding_model: str
    chunk_size_config: int
    chunk_overlap_config: int

@dataclass
class DocumentStats:
    """Statistics for individual documents"""
    file_name: str
    file_size: int
    chunk_count: int
    processing_time: float
    avg_chunk_size: float

class StatsAnalyzer:
    """Analyze and report performance statistics"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_chunking_performance(self, 
                                   documents: List,
                                   nodes: List,
                                   processing_time: float,
                                   metadata: List,
                                   strategy_name: str,
                                   config: Dict[str, Any]) -> ChunkingStats:
        """Analyze chunking performance and generate statistics"""
        
        # Calculate chunk size statistics
        chunk_sizes = [len(node.get_content()) for node in nodes]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        # Calculate performance metrics
        chunks_per_second = len(nodes) / processing_time if processing_time > 0 else 0
        
        stats = ChunkingStats(
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
        
        return stats
    
    def analyze_document_stats(self, documents: List, nodes: List, metadata: List) -> List[DocumentStats]:
        """Analyze statistics for individual documents"""
        doc_stats = []
        
        # Group chunks by source document
        doc_chunks = {}
        for node in nodes:
            source_file = node.metadata.get('file_name', 'unknown')
            if source_file not in doc_chunks:
                doc_chunks[source_file] = []
            doc_chunks[source_file].append(node)
        
        # Calculate stats for each document
        for doc in documents:
            file_name = doc.metadata.get('file_name', 'unknown')
            file_size = doc.metadata.get('file_size', 0)
            
            # Find chunks for this document
            chunks = doc_chunks.get(file_name, [])
            chunk_count = len(chunks)
            
            # Calculate average chunk size for this document
            if chunks:
                chunk_sizes = [len(chunk.get_content()) for chunk in chunks]
                avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
            else:
                avg_chunk_size = 0
            
            # Find processing time from metadata
            processing_time = 0
            for meta in metadata:
                if hasattr(meta, 'source_file') and meta.source_file == file_name:
                    processing_time = meta.processing_time
                    break
            
            doc_stats.append(DocumentStats(
                file_name=file_name,
                file_size=file_size,
                chunk_count=chunk_count,
                processing_time=processing_time,
                avg_chunk_size=avg_chunk_size
            ))
        
        return doc_stats
    
    def generate_stats_report(self, 
                            chunking_stats: ChunkingStats, 
                            document_stats: List[DocumentStats],
                            config: Dict[str, Any]) -> str:
        """Generate a comprehensive statistics report"""
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# RAG Chunking Performance Report

**Generated:** {timestamp}  
**Strategy:** {chunking_stats.strategy_name}  
**Embedding Model:** {chunking_stats.embedding_model}

## Configuration
- **Chunk Size:** {chunking_stats.chunk_size_config}
- **Chunk Overlap:** {chunking_stats.chunk_overlap_config}
- **Document Type:** {config.get('document_type', 'Unknown')}
- **Data Path:** {config.get('data_path', 'Unknown')}

## Overall Performance

| Metric | Value |
|--------|-------|
| Total Documents | {chunking_stats.total_documents} |
| Total Chunks | {chunking_stats.total_chunks} |
| Processing Time | {chunking_stats.processing_time:.2f} seconds |
| Chunks per Second | {chunking_stats.chunks_per_second:.1f} |
| Average Chunk Size | {chunking_stats.avg_chunk_size:.0f} characters |
| Min Chunk Size | {chunking_stats.min_chunk_size} characters |
| Max Chunk Size | {chunking_stats.max_chunk_size} characters |

## Document-Level Statistics

| Document | Size (bytes) | Chunks | Avg Chunk Size | Processing Time |
|----------|--------------|--------|----------------|-----------------|
"""
        
        for doc_stat in document_stats:
            report += f"| {doc_stat.file_name} | {doc_stat.file_size:,} | {doc_stat.chunk_count} | {doc_stat.avg_chunk_size:.0f} | {doc_stat.processing_time:.3f}s |\n"
        
        report += f"""
## Performance Analysis

### Efficiency Metrics
- **Documents per Second:** {chunking_stats.total_documents / chunking_stats.processing_time:.2f}
- **Average Chunks per Document:** {chunking_stats.total_chunks / chunking_stats.total_documents:.1f}
- **Character Processing Rate:** {sum(doc.file_size for doc in document_stats) / chunking_stats.processing_time:.0f} chars/sec

---
*Report generated by RAG Experiment Runner*
"""
        
        return report
    
    def save_stats_results(self, 
                          chunking_stats: ChunkingStats,
                          document_stats: List[DocumentStats],
                          config: Dict[str, Any]) -> Dict[str, str]:
        """Save statistics results to multiple formats"""
        
        timestamp = int(time.time())
        base_name = f"stats_{chunking_stats.strategy_name}_{timestamp}"
        
        # Save JSON results
        json_data = {
            'timestamp': timestamp,
            'config': config,
            'chunking_stats': {
                'total_documents': chunking_stats.total_documents,
                'total_chunks': chunking_stats.total_chunks,
                'processing_time': chunking_stats.processing_time,
                'avg_chunk_size': chunking_stats.avg_chunk_size,
                'min_chunk_size': chunking_stats.min_chunk_size,
                'max_chunk_size': chunking_stats.max_chunk_size,
                'chunks_per_second': chunking_stats.chunks_per_second,
                'strategy_name': chunking_stats.strategy_name,
                'embedding_model': chunking_stats.embedding_model
            },
            'document_stats': [
                {
                    'file_name': doc.file_name,
                    'file_size': doc.file_size,
                    'chunk_count': doc.chunk_count,
                    'processing_time': doc.processing_time,
                    'avg_chunk_size': doc.avg_chunk_size
                }
                for doc in document_stats
            ]
        }
        
        json_file = self.output_dir / f"{base_name}.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save CSV for easy analysis
        df_data = []
        for doc in document_stats:
            df_data.append({
                'document': doc.file_name,
                'file_size': doc.file_size,
                'chunk_count': doc.chunk_count,
                'processing_time': doc.processing_time,
                'avg_chunk_size': doc.avg_chunk_size,
                'strategy': chunking_stats.strategy_name,
                'embedding_model': chunking_stats.embedding_model
            })
        
        df = pd.DataFrame(df_data)
        csv_file = self.output_dir / f"{base_name}.csv"
        df.to_csv(csv_file, index=False)
        
        # Save markdown report
        report = self.generate_stats_report(chunking_stats, document_stats, config)
        md_file = self.output_dir / f"{base_name}_report.md"
        with open(md_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📊 Statistics saved to:")
        logger.info(f"   JSON: {json_file}")
        logger.info(f"   CSV:  {csv_file}")
        logger.info(f"   Report: {md_file}")
        
        return {
            'json': str(json_file),
            'csv': str(csv_file),
            'report': str(md_file)
        }
    
    def print_console_summary(self, chunking_stats: ChunkingStats):
        """Print a summary to console"""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE STATISTICS SUMMARY")
        print("=" * 60)
        print(f"Strategy:            {chunking_stats.strategy_name}")
        print(f"Total Documents:     {chunking_stats.total_documents}")
        print(f"Total Chunks:        {chunking_stats.total_chunks}")
        print(f"Processing Time:     {chunking_stats.processing_time:.2f} seconds")
        print(f"Chunks per Second:   {chunking_stats.chunks_per_second:.1f}")
        print(f"Average Chunk Size:  {chunking_stats.avg_chunk_size:.0f} characters")
        print(f"Min Chunk Size:      {chunking_stats.min_chunk_size}")
        print(f"Max Chunk Size:      {chunking_stats.max_chunk_size}")
        print(f"Embedding Model:     {chunking_stats.embedding_model}")
        print("=" * 60)


class ResultAnalyzer:
    """RAGAS results analyzer"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_results = []
    
    def add_evaluation(self, ragas_df: pd.DataFrame):
        """Add evaluation results"""
        if not ragas_df.empty:
            self.all_results.append(ragas_df)
    
    def export_to_spreadsheet(self):
        """Export all results to Excel"""
        if not self.all_results:
            logger.warning("No results to export")
            return
        
        timestamp = int(time.time())
        excel_file = self.output_dir / f"ragas_results_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_file) as writer:
            # Combine all results
            combined_df = pd.concat(self.all_results, ignore_index=True)
            combined_df.to_excel(writer, sheet_name="All Results", index=False)
            
            # Summary statistics
            if 'faithfulness' in combined_df.columns:
                summary_data = {
                    'Metric': ['Faithfulness', 'Answer Relevancy', 'Context Recall', 'Context Precision'],
                    'Average': [
                        combined_df['faithfulness'].mean() if 'faithfulness' in combined_df else 0,
                        combined_df['answer_relevancy'].mean() if 'answer_relevancy' in combined_df else 0,
                        combined_df['context_recall'].mean() if 'context_recall' in combined_df else 0,
                        combined_df['context_precision'].mean() if 'context_precision' in combined_df else 0
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        logger.info(f"📊 Exported results to: {excel_file}")
    
    def generate_markdown_report(self):
        """Generate markdown report of results"""
        if not self.all_results:
            logger.warning("No results to report")
            return
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        report_file = self.output_dir / f"ragas_report_{int(time.time())}.md"
        
        report = f"""# RAGAS Evaluation Report

Generated: {timestamp}

## Summary

"""
        # Add content based on results
        combined_df = pd.concat(self.all_results, ignore_index=True)
        
        if 'faithfulness' in combined_df.columns:
            report += f"""
### Average Scores
- **Faithfulness:** {combined_df['faithfulness'].mean():.4f}
- **Answer Relevancy:** {combined_df['answer_relevancy'].mean():.4f}
- **Context Recall:** {combined_df.get('context_recall', pd.Series()).mean():.4f}
- **Context Precision:** {combined_df.get('context_precision', pd.Series()).mean():.4f}

### Sample Questions and Answers

"""
            # Add sample Q&A
            for idx, row in combined_df.head(5).iterrows():
                report += f"**Q{idx+1}:** {row.get('question', 'N/A')}\n\n"
                report += f"**A{idx+1}:** {row.get('answer', 'N/A')}\n\n"
                report += f"**Ground Truth:** {row.get('ground_truth', 'N/A')}\n\n"
                report += "---\n\n"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Generated report: {report_file}")


def save_evaluation_results(result, output_dir: str):
    """Save evaluation results - maintains compatibility"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    
    # Save basic result info
    result_data = {
        'strategy_name': result.strategy_name,
        'total_chunks': result.total_chunks,
        'avg_query_time': result.avg_query_time,
        'timestamp': timestamp
    }
    
    if hasattr(result, 'ragas_scores') and result.ragas_scores:
        result_data['ragas_scores'] = result.ragas_scores
    
    # Save to JSON
    json_file = output_path / f"eval_result_{result.strategy_name}_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    # Save DataFrame if available
    if hasattr(result, 'ragas_scores_df') and result.ragas_scores_df is not None:
        csv_file = output_path / f"ragas_scores_{result.strategy_name}_{timestamp}.csv"
        result.ragas_scores_df.to_csv(csv_file, index=False)
        logger.info(f"💾 Saved RAGAS scores to {csv_file}")
    
    logger.info(f"💾 Saved evaluation results to {json_file}")