# run.py (Final Version with similarity_top_k)

import os
import sys
import argparse
import time
import yaml
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv

# --- Setup Paths and Environment Variables FIRST ---
os.environ["VLLM_USE_V1"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

# --- Now Import Project and Library Modules ---
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from src.vllm_experiment import VLLMChunkingExperiment

class RAGExperimentRunner:
    """
    A class to run RAG chunking experiments, handling user input,
    configuration, and orchestration of the experiment pipeline.
    """

    def __init__(self):
        """Initializes the runner with a default configuration."""
        self.config = self._load_default_config()
        self.questions = []
        self.ground_truths = []
        self.console = Console()

    def _load_default_config(self):
        """Loads the default configuration."""
        import torch
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb < 4.0:
                print(f"  Limited GPU memory detected ({gpu_memory_gb:.1f}GB). Using small embedding model.")
                embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            else:
                embedding_model = "BAAI/bge-m3"
        else:
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Fallback defaults
        return {
            'llm_provider': 'ollama',
            'llm_model': 'llama3.1:8b',
            'mode': 'quick',
            'document_type': 'pdf',
            'strategies': ['sentence_splitter'],
            'chunk_size': 512,
            'chunk_overlap': 50,
            'similarity_top_k': 5, # <-- ADDED
            'embedding_model': embedding_model,
            'use_vllm': False,
            'use_venice_evaluation': False,
            'venice_model': 'llama-3.3-70b'
        }

    def _get_venice_models(self):
        """Get list of available Venice AI models."""
        return {
            "llama-3.2-3b": "Llama 3.2 3B - Fast, function calling support",
            "qwen3-4b": "Venice Small (Qwen3 4B) - Fast, reasoning, function calling",
            "mistral-31-24b": "Venice Medium (Mistral 31 24B) - Vision, 131k context",
            "llama-3.3-70b": "Llama 3.3 70B - Default model, function calling",
            "qwen3-235b": "Venice Large (Qwen3 235B) - Most intelligent, 131k context",
            "llama-3.1-405b": "Llama 3.1 405B - Most intelligent, 65k context",
        }

    def _prompt_for_selection(self, title: str, default: str, choices: list) -> str:
        self.console.print(Panel(f" {title}", expand=False, border_style="blue"))
        return Prompt.ask(f"Select {title.lower()}", choices=choices, default=default)

    def _prompt_for_input(self, title: str, default) -> str:
        return Prompt.ask(f"{title}", default=str(default))

    def _get_venice_evaluation_choice(self):
        if not os.getenv('VENICE_API_KEY'):
            self.console.print(Panel(
                "[bold yellow] Venice AI Not Available[/bold yellow]\n\n"
                "To enable high-quality RAGAS evaluation, set your Venice AI API key in a `.env` file.\n"
                "`VENICE_API_KEY='your-key-here'`\n\nUsing local LLM for evaluation instead.",
                expand=False, border_style="yellow"
            ))
            return False
        
        self.console.print(Panel(
            "[bold green] Venice AI Available[/bold green]\n\n"
            "Use a powerful Venice AI model (llama-3.3-70b) for more accurate RAGAS evaluation.",
            expand=False, border_style="green"
        ))
        
        return Prompt.ask(
            "Use Venice AI for RAGAS evaluation?",
            choices=["y", "n"], default="y"
        ).lower() == "y"

    def _get_embedding_models(self):
        """Get dictionary of available embedding models with descriptions."""
        return {
            "sentence-transformers/all-MiniLM-L6-v2": "Small, fast model (384 dims) - Good for limited resources",
            "sentence-transformers/all-mpnet-base-v2": "Medium model (768 dims) - Better quality than MiniLM",
            "BAAI/bge-small-en-v1.5": "Small BGE model (384 dims) - Efficient and accurate",
            "BAAI/bge-base-en-v1.5": "Base BGE model (768 dims) - Balanced performance",
            "BAAI/bge-large-en-v1.5": "Large BGE model (1024 dims) - Best quality BGE",
            "BAAI/bge-m3": "Multilingual BGE model (1024 dims) - Supports 100+ languages",
            "thenlper/gte-small": "Small GTE model (384 dims) - Good performance",
            "thenlper/gte-base": "Base GTE model (768 dims) - Better than small",
            "thenlper/gte-large": "Large GTE model (1024 dims) - Best GTE quality",
            "intfloat/e5-small-v2": "Small E5 model (384 dims) - Efficient",
            "intfloat/e5-base-v2": "Base E5 model (768 dims) - Good balance",
            "intfloat/e5-large-v2": "Large E5 model (1024 dims) - Highest quality E5",
        }

    def _select_embedding_model(self):
        """Prompts user to select an embedding model."""
        self.console.print(Panel(" Embedding Model Selection", expand=False, border_style="blue"))
        current_model = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        import torch
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.console.print(f"[yellow]GPU Memory: {gpu_memory_gb:.1f}GB[/yellow]")
            if gpu_memory_gb < 4.0:
                self.console.print("[yellow]  Limited GPU memory - smaller models recommended[/yellow]")
        self.console.print(f"\nCurrent embedding model: [cyan]{current_model}[/cyan]")
        change_model = Prompt.ask(
            "Change embedding model?",
            choices=["y", "n"],
            default="n"
        ).lower() == "y"
        if not change_model:
            return
        embedding_models = self._get_embedding_models()
        self.console.print("\n[bold]Available Embedding Models:[/bold]")
        self.console.print("\n[green]Small Models (384 dims) - Fast, low memory:[/green]")
        for model, desc in embedding_models.items():
            if "384 dims" in desc:
                self.console.print(f"  • {model}")
                self.console.print(f"    [dim]{desc}[/dim]")
        self.console.print("\n[yellow]Medium Models (768 dims) - Balanced:[/yellow]")
        for model, desc in embedding_models.items():
            if "768 dims" in desc:
                self.console.print(f"  • {model}")
                self.console.print(f"    [dim]{desc}[/dim]")
        self.console.print("\n[red]Large Models (1024 dims) - Best quality, high memory:[/red]")
        for model, desc in embedding_models.items():
            if "1024 dims" in desc:
                self.console.print(f"  • {model}")
                self.console.print(f"    [dim]{desc}[/dim]")
        self.console.print("\n[bold]Quick Selection:[/bold]")
        quick_choices = {
            "1": "sentence-transformers/all-MiniLM-L6-v2",
            "2": "BAAI/bge-small-en-v1.5", 
            "3": "BAAI/bge-base-en-v1.5",
            "4": "BAAI/bge-m3",
            "5": "intfloat/e5-large-v2",
            "6": "custom"
        }
        for key, model in quick_choices.items():
            if model != "custom":
                self.console.print(f"{key}. {model.split('/')[-1]}")
            else:
                self.console.print(f"{key}. Enter custom model name")
        choice = Prompt.ask("Select embedding model", choices=list(quick_choices.keys()), default="1")
        if quick_choices[choice] == "custom":
            self.config['embedding_model'] = Prompt.ask("Enter custom embedding model name")
        else:
            self.config['embedding_model'] = quick_choices[choice]
        self.console.print(f"[green]✓ Selected embedding model: {self.config['embedding_model']}[/green]")

    def _get_user_input(self):
        """Gets user input for the experiment configuration."""
        self.config['llm_provider'] = self._prompt_for_selection("LLM Provider", self.config.get('llm_provider') or "ollama", ["vllm", "ollama", "venice"])
        if self.config['llm_provider'] == 'ollama':
            try:
                result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
                lines = result.stdout.strip().split('\n')
                model_names = [line.split()[0] for line in lines[1:]] if len(lines) > 1 else []
                if not model_names:
                    self.console.print("[bold red]No models found via 'ollama list'. Please ensure models are installed.[/bold red]")
                    exit()
                self.config['llm_model'] = self._prompt_for_selection("Ollama model name", model_names[0], model_names)
            except Exception:
                self.console.print("[bold red]Could not execute 'ollama list'. Using default.[/bold red]")
                self.config['llm_model'] = self._prompt_for_input("LLM Model", self.config.get('llm_model'))
        elif self.config['llm_provider'] == 'venice':
            self.config['llm_model'] = self._prompt_for_input("LLM Model", 'llama-3.2-3b')
        else: # vllm
            self.config['llm_model'] = self._prompt_for_input("LLM Model", 'microsoft/phi-2')
        self._select_embedding_model()
        self.config['use_venice_evaluation'] = self._get_venice_evaluation_choice()
        if self.config['use_venice_evaluation']:
            self.config['venice_model'] = 'llama-3.3-70b'
        self.config['mode'] = self._prompt_for_selection("Evaluation Mode", self.config.get('mode') or "quick", ["quick", "full", "stats_only"])
        self.config['document_type'] = self._prompt_for_selection("Document Type", self.config.get('document_type') or "pdf", ["pdf", "docx", "pptx","txt","json"])
        self.console.print(Panel(" Chunking Strategies", expand=False, border_style="blue"))
        strategy_choices = {"1": "sentence_splitter", "2": "token_text_splitter", "3": "semantic_splitter", "4": "recursive_character", "5": "hierarchical"}
        for key, value in strategy_choices.items(): self.console.print(f"{key}. {value}")
        strategy_input = Prompt.ask("Select strategies (comma-separated, or 'all')", default="1").lower()
        if strategy_input == 'all':
            self.config['strategies'] = list(strategy_choices.values())
        else:
            selected_keys = [key.strip() for key in strategy_input.split(',')]
            self.config['strategies'] = [strategy_choices[key] for key in selected_keys if key in strategy_choices]
        
        self.console.print(Panel(" Chunking & Retrieval Parameters", expand=False, border_style="blue"))
        self.config['chunk_size'] = int(self._prompt_for_input("Chunk size", self.config.get('chunk_size')))
        self.config['chunk_overlap'] = int(self._prompt_for_input("Chunk overlap", self.config.get('chunk_overlap')))
        self.config['similarity_top_k'] = int(self._prompt_for_input("Similarity Top K (chunks to retrieve)", self.config.get('similarity_top_k')))

    def run_experiment(self, config_file=None):
        """Main method to run the experiment."""
        self.console.print(Panel(
            "RAG CHUNKING EXPERIMENT RUNNER ", expand=False, border_style="green", title="Experiment Runner"
        ))

        if config_file:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self._get_user_input()

        experiment = VLLMChunkingExperiment(self.config)
        
        output_dir = Path(f"results/exp_{int(time.time())}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.config.get('Questions'):
            try:
                with open(f'{self.config.get("document_type")}_ground_truth.json', 'r') as f:
                    self.config['Questions'] = [{'question': q, 'answer': a} for q, a in json.load(f).items()]
            except FileNotFoundError:
                   self.console.print("[bold yellow] Ground truth file not found. Using default test question.[/bold yellow]")
                   self.config['Questions'] = [{'question': "What is it considered a sin to kill in Harper Lee's novel?", 'answer': 'A Mockingbird.'}]

        questions_data = self.config.get('Questions', [])
        self.questions = [q['question'] for q in questions_data]
        self.ground_truths = [q['answer'] for q in questions_data]

        if self.config.get('mode') == 'quick':
            test_size = min(3, len(self.questions))
            self.questions = self.questions[:test_size]
            self.ground_truths = self.ground_truths[:test_size]

        experiment.run_full_experiment(
            data_path=f"data/raw/{self.config.get('document_type')}",
            output_dir=str(output_dir),
            strategies_to_test=self.config.get('strategies', ['sentence_splitter']),
            questions=self.questions,
            ground_truths=self.ground_truths,
        )
        self.console.print(f"\n[bold green] Experiment finished! Results saved in {output_dir}[/bold green]")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run RAG Experiment with specified config.")
    parser.add_argument('--config', type=str, help='Path to the YAML config file.')
    args = parser.parse_args()

    runner = RAGExperimentRunner()
    runner.run_experiment(config_file=args.config)

if __name__ == "__main__":
    main()