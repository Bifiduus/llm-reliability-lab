## Installation

LLM Reliability Lab can be set up automatically on Linux or Ubuntu/WSL.

### Prerequisites

- Python 3.10+
- Git
- curl
- Python venv support (`python3-venv`)
- Sufficient RAM and disk space for the local models
- Internet connection

### Automatic setup

Clone the repository:

```bash
git clone <repository-url>
cd llm-reliability-lab
```

Run the installation script:

```bash
bash scripts/setup.sh
```

The setup script automatically:

1. Creates a Python virtual environment.
2. Installs the Python dependencies.
3. Downloads the scientific document corpus from arXiv.
4. Generates the document chunks.
5. Downloads the embedding and reranking models.
6. Installs Ollama if necessary and downloads Qwen2.5 3B.

### Running the project

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the unit tests:

```bash
pytest -v
```

Evaluate the retrieval methods:

```bash
python src/evaluate_retrieval.py
```

Retrieve the most relevant chunks for each benchmark question using the different retrieval methods:

```bash
python src/retrieve_rag_contexts.py
```
The retrieved contexts are saved to results/retrieved_contexts.json.

Ollama must be running to evaluate the RAG pipelines. If necessary, start it with:

```bash
ollama serve
```
If the address is already in use, Ollama is likely already running.

Generate answers using the previously retrieved contexts:

```bash
python src/generate_rag_answers.py
```
The generated answers are saved incrementally to results/rag_results.json, allowing generation to resume after an interruption.

### Reproducibility

The benchmark questions are version-controlled, while the PDF corpus, generated chunks and experiment outputs are excluded from Git.

The setup script reconstructs the document corpus and downloads the required models.

Results can vary depending on model versions, hardware and generation settings.