# Climate+Tech FactChecker

Climate+Tech FactChecker is designed to serve as a comprehensive toolkit for both experimentation and production environments, focusing on the verification of claims. It offers a robust suite of tools and methodologies to assist researchers, developers, and practitioners in the field of claim verification, enabling them to efficiently test hypotheses, validate data, and deploy reliable fact-checking solutions.

## Installation

     python3 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt

## Project Structure

The project follows a modular structure:

- `factchecker/`: Main package directory
  - `core/`: Core functionality including LLM and embedding models
  - `experiments/`: Contains experiment scripts for different fact-checking approaches
  - `strategies/`: Core fact-checking strategy implementations
  - `utils/`: Utility functions and helper modules
  - `tools/`: Utility scripts for tasks like downloading sources
- `tests/`: Test suite following the same structure as the main package
- `storage/`: Data storage for indices and other persistent data
- `data/`: (gitignored) Directory for storing downloaded source documents

## Configuration

### Setting Up Environment Variables

Before running the Climate+Tech FactChecker, you need to configure your environment variables. This is done using the `.env.example` file provided in the repository.

1. **Copy the `.env.example` file to a new file named `.env`:**

   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file to include your specific configuration:**

   - Replace `your_openai_api_key_here` with your actual OpenAI API key.
   - Update `OPENAI_ORGANIZATION` with your OpenAI organization ID if applicable.
   - Adjust other variables like `LLM_TYPE`, `OLLAMA_API_BASE_URL`, and `OLLAMA_MODEL` as needed for your setup.
   - Configure embedding model settings (see Embedding Models section below)

### Configuring the LLM

The project uses LlamaIndex's LLM interface through the `factchecker/core/llm.py` module:

1. **OpenAI (Default)**
   - Set `LLM_TYPE=openai` in `.env`
   - Required settings:
     - `OPENAI_API_KEY`: Your OpenAI API key
   - Optional settings:
     - `OPENAI_API_BASE`: Custom API endpoint (default: OpenAI's API)
     - `OPENAI_API_MODEL`: Model to use (default: "gpt-3.5-turbo-1106")
     - `OPENAI_ORGANIZATION`: Your organization ID
     - `TEMPERATURE`: Model temperature (default: 0.1) - shared with Ollama
   - Uses `llama_index.llms.openai.OpenAI` under the hood

2. **Ollama**
   - Set `LLM_TYPE=ollama` in `.env`
   - Required settings:
     - `OLLAMA_MODEL`: Model to use (e.g., "llama2", "mistral")
   - Optional settings:
     - `OLLAMA_API_BASE_URL`: Custom API endpoint (default: "http://localhost:11434")
     - `OLLAMA_REQUEST_TIMEOUT`: Request timeout in seconds (default: 120.0)
     - `TEMPERATURE`: Model temperature (default: 0.1) - shared with OpenAI
   - Uses `llama_index.llms.ollama.Ollama` under the hood

Example usage:
```python
from factchecker.core.llm import load_llm

# Using default OpenAI settings
llm = load_llm()

# Using OpenAI with custom settings
llm = load_llm(
    llm_type="openai",
    model="gpt-3.5-turbo-1106",
    temperature=0.1,
    api_key="your-key",
    organization="your-org",
    context_window=4096  # Optional: control context window size
)

# Using Ollama with custom settings
llm = load_llm(
    llm_type="ollama",
    model="mistral",
    temperature=0.1,
    request_timeout=120.0,
    context_window=4096  # Optional: control context window size
)
```

Note: The LLM interface is compatible with LlamaIndex's query engine, retriever, and other components. You can use any LlamaIndex-supported LLM by modifying the loader implementation.

### Embedding Models

The project uses LlamaIndex's embedding interface through the `factchecker/core/embeddings.py` module:

1. **OpenAI Embeddings (Default)**
   - Set `EMBEDDING_TYPE=openai` in `.env` (or omit for default)
   - Required settings:
     - `OPENAI_API_KEY`: Your OpenAI API key
   - Optional settings:
     - `OPENAI_EMBEDDING_MODEL`: Model to use (default: "text-embedding-ada-002")
     - `OPENAI_API_BASE`: Custom API endpoint
   - Features:
     - High-quality embeddings
     - Consistent dimensionality (1536)
     - Production-ready reliability
   - Uses `llama_index.embeddings.openai.OpenAIEmbedding`

2. **HuggingFace Embeddings**
   - Set `EMBEDDING_TYPE=huggingface` in `.env`
   - Optional settings:
     - `HUGGINGFACE_EMBEDDING_MODEL`: Model to use (default: "BAAI/bge-small-en-v1.5")
   - Additional kwargs support:
     - `device`: CPU/GPU selection ("cpu", "cuda", etc.)
     - `normalize_embeddings`: Whether to normalize vectors
     - Any other kwargs supported by the model
   - Features:
     - Local execution capability
     - Wide range of available models
     - Customizable model loading
   - Uses `llama_index.embeddings.huggingface.HuggingFaceEmbedding`

3. **Ollama Embeddings**
   - Set `EMBEDDING_TYPE=ollama` in `.env`
   - Required settings:
     - `OLLAMA_MODEL`: Model to use (default: "nomic-embed-text")
   - Optional settings:
     - `OLLAMA_API_BASE_URL`: Custom API endpoint (default: "http://localhost:11434")
   - Additional kwargs support:
     - `request_timeout`: Specific request timeout
   - Features:
     - Local execution
     - Integration with Ollama's model ecosystem
     - No API key required
   - Uses `llama_index.embeddings.ollama.OllamaEmbedding`

Example usage:
```python
from factchecker.core.embeddings import load_embedding_model

# Default OpenAI embeddings
embeddings = load_embedding_model()

# OpenAI with custom settings
embeddings = load_embedding_model(
    embedding_type="openai",
    model_name="text-embedding-ada-002",
    api_key="your-key",
    api_base="custom-endpoint"
)

# HuggingFace with custom settings
embeddings = load_embedding_model(
    embedding_type="huggingface",
    model_name="BAAI/bge-small-en-v1.5",
    device="cuda",
    normalize_embeddings=True
)

# Ollama with custom settings
embeddings = load_embedding_model(
    embedding_type="ollama",
    model_name="nomic-embed-text",
    base_url="http://custom-server:11434",
    request_timeout=60
)
```

Note: The embedding interface is compatible with LlamaIndex's vector stores, retrievers, and other components. The embeddings are used for semantic search and similarity comparisons in the fact-checking process.

## Utility Modules

The `factchecker/utils/` directory contains shared functionality used across the project:

- `verdict_mapping.py`: Standardizes the mapping of different verdict labels
  - Converts between various rating systems (e.g., Climate Feedback ratings)
  - Provides consistent verdict categories across the project
  - Includes comprehensive test coverage in `tests/utils/test_verdict_mapping.py`

Additional utilities handle common operations like data processing, API interactions, and shared helper functions. All utility modules follow TDD principles with corresponding test files in the `tests/utils/` directory.

## Running Experiments

The project includes several experiment scripts to evaluate different fact-checking approaches:

1. Climate Feedback Advocate-Mediator Experiment:
   ```bash
   python -m factchecker.experiments.advocate_mediator_climatefeedback
   ```
   This experiment implements a fact-checking approach using the advocate-mediator pattern with Climate Feedback data. The process:
   - Uses Climate Feedback's expert-reviewed claims as ground truth
   - Implements an advocate-mediator pattern where:
     - Advocates argue for/against the claim's validity
     - A mediator evaluates the arguments and provides a final verdict
   - Verdicts are standardized using the utility mapping system
   - Results are compared against expert ratings for evaluation

   Building Your Own Advocate-Mediator Experiment:
   ```python
   # Core components needed:
   from factchecker.strategies.advocate_mediator import AdvocateMediatorStrategy
   from factchecker.utils.verdict_mapping import map_verdict

   # 1. Define your strategy with custom prompts
   strategy = AdvocateMediatorStrategy(
       advocate_prompt="Your custom advocate prompt...",
       mediator_prompt="Your custom mediator prompt..."
   )

   # 2. Process your claims
   claim = "Your claim text..."
   context = "Supporting context/evidence..."
   result = strategy.evaluate_claim(claim, context)

   # 3. Map the verdict to standardized format
   standardized_verdict = map_verdict(result.verdict)
   ```

   The experiment structure consists of:
   - A strategy class in `strategies/` implementing the core logic
   - An experiment script in `experiments/` handling data loading and evaluation
   - Utility functions in `utils/` for standardization and common operations

   LLM Setup Options:
   1. Using Ollama:
      ```python
      from factchecker.llm.ollama import OllamaLLM
      
      # Initialize Ollama with your chosen model
      llm = OllamaLLM(model_name="llama2")  # or any other supported model
      strategy = AdvocateMediatorStrategy(
          advocate_prompt="...",
          mediator_prompt="...",
          llm=llm
      )
      ```
      - Requires Ollama to be installed and running locally
      - No API key needed
      - Supports various open-source models
      - Full control over model deployment and infrastructure
      - Can run completely offline

   2. Using OpenAI:
      - Requires setting up OPENAI_API_KEY environment variable
      - Uses GPT models
      - Pay-per-use pricing model
      - No local infrastructure needed

2. Evidence Evaluation:
   ```bash
   python -m factchecker.experiments.evidence_evaluation_1
   ```

When running experiments:
- Set up your preferred LLM backend (Ollama recommended for development)
- Check the experiment's source code for any specific configuration options
- Results will typically be saved in the experiment's output directory

## Testing

The project follows Test-Driven Development (TDD) principles:

1. Running Tests:
   ```bash
   pytest
   ```
   This will run all tests in the `tests/` directory.

2. Writing Tests:
   - Create test files in the `tests/` directory mirroring the main package structure
   - Name test files with `test_` prefix (e.g., `test_advocate_mediator.py`)
   - Each test function should start with `test_`
   - Include both positive and negative test cases
   - Mock external API calls (e.g., OpenAI) to ensure tests run without actual API usage

3. Test Coverage:
   ```bash
   pytest --cov=factchecker
   ```
   This command will show test coverage statistics.

## Contribution

You accept the CONTRIBUTOR_LICENSE_AGREEMENT by contributing.

## Dealing with sources

### Annotate sources

Request access to https://docs.google.com/spreadsheets/d/1R0-q5diheG3zXDBq8V2aoUGOQyRI6HuUisTf-4wTsWY/edit#gid=0

### Download sources

The project includes a source downloader tool to fetch PDFs and other source documents. The tool supports:

1. **Basic Usage**
   ```bash
   python -m factchecker.tools.sources_downloader
   ```
   This uses default settings:
   - Source file: `sources/sources.csv`
   - Output folder: `data/`
   - URL column: "external_link"

2. **Custom Configuration**
   ```bash
   python -m factchecker.tools.sources_downloader \
     --sourcefile custom_sources.csv \
     --output_folder custom_folder \
     --url_column url_field \
     --rows 1 2 3  # Optional: Download specific rows only
   ```

3. **Features**
   - Automatically creates output directory if it doesn't exist
   - Handles failed downloads gracefully with error messages
   - Supports CSV files with custom column names
   - Optional row selection for partial downloads
   - Progress tracking for batch downloads

Note: The `/data` folder is specified in the `.gitignore` file, so the downloaded PDFs will not be tracked by Git.
