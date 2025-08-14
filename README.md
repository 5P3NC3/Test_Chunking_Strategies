# Test_Chunking_Strategies
This code was a small test to see how different chunking strategies perform locally using qdrant, llamaindex, and RAGAS. This is the first iteration of this test and so not all parts are working. If you would like to run the code here are the commands to run from the home directory:

python -m venv .venv

source .venv/bin/activate

uv pip install -r requirements.txt

docker compose up -d

python run.py

Currently ollama works perfectly for the LLM and embedding model and Qdrant works correctly. The api also works for any OpenAI-compatible endpoint. This test used Venice, but this api can be openai or other options as long as you use a model that can properly grade RAGAS and format json scores. All results were collected with those options selected. The other features in the code are not supported in this early iteration, like vllm, ppt or local grading, they are just included so the code is the exact same as it was when the results were created. 

# Results
If you are just interested in the results or findings from the blog post, just take a look in the directory helpfully named "results". All findngs are included, the final resuls were just averages collected from the raw stats in this directory. The exact configurations used on each test are included in the json files. The raw data and ground truths are also included. 

In the results you may notice that the testing sizes are not that large or that not every score was perfectly measured as you would want them to be. This is because each test takes over a couple of hours on local hardware and the API keys can run up a bill with these tests. The results are only meant to show how different the scores can be using different chunking strategies, which it successfully does. We will be creating more thorough test on larger data sets in the future, likely with different methodologies.

# Code Flow

Example: Ollama + Venice AI + Full Test

User runs python run.py and selects: Ollama (llama3.1:8b), Venice AI evaluation, Full mode, PDF documents, All strategies
System loads PDFs from data/raw/pdf/ and test questions from pdf_ground_truth.json
For each strategy (sentence, token, semantic, recursive, hierarchical):

Chunks documents → Stores in Qdrant → Builds vector index → Answers questions → RAGAS evaluation with large LLM


Generates comparative reports showing which chunking strategy performed best
Outputs Excel spreadsheets, JSON results, and markdown reports to results/ folder

# File Functions

run.py - Main entry point, handles user input and orchestrates entire experiment
src/pipeline.py - Core RAG pipeline logic, handles document loading and RAGAS evaluation

Evaluation & Analysis:

src/venice_ragas_evaluator.py - High-quality evaluation using llama3.3 70b
src/ragas_adapters.py - Compatibility layer between LlamaIndex and RAGAS frameworks
src/stats.py - Performance statistics, analysis, and report generation

Configuration & Setup:

configs/*.yaml - Pre-configured experiment settings for different scenarios
select_model.py - Interactive model selection helper with GPU memory recommendations
docker-compose.yml - Qdrant vector database setup
requirements.txt - Python dependencies

Data & Ground Truth:

*_ground_truth.json - Test questions and expected answers for each document type
data/raw/{pdf,docx,pptx,txt}/ - Input documents for processing
results/ - Generated reports, statistics, and evaluation scores
