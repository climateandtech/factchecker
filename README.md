# Climate+Tech FactChecker

Climate+Tech FactChecker is designed to serve as a comprehensive toolkit for both experimentation and production environments, focusing on the verification of claims. It offers a robust suite of tools and methodologies to assist researchers, developers, and practitioners in the field of claim verification, enabling them to efficiently test hypotheses, validate data, and deploy reliable fact-checking solutions.

## Installation

     python3 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt

## Project Structure

The project follows a modular structure:

- `factchecker/`: Main package directory
  - `experiments/`: Contains experiment scripts for different fact-checking approaches
  - `strategies/`: Core fact-checking strategy implementations
  - `utils/`: Utility functions and helper modules
- `tests/`: Test suite following the same structure as the main package
- `storage/`: Data storage for indices and other persistent data
- `data/`: (gitignored) Directory for storing downloaded source documents

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

## Contact

Connect with us through various channels:

- **LinkedIn**: [Climate&Tech](https://www.linkedin.com/company/climateandtech/)
- **Email**: contact@climateandtech.com
- **Discord**: [Join our community](https://discord.gg/TQC6qTfV)
- **Slack**: [Join our workspace](https://climatetechai-uos8147.slack.com/)
- **Web**: [climateandtech.com](https://climateandtech.com)

## License

This project is dual-licensed:

- For researchers, academic institutions, universities, and fact-checking organizations: This project is available under the GNU Affero General Public License (AGPL), with the requirement that any results and improvements are shared back with the community.

- For other organizations: A license can be requested and can be granted free if the project's use is not purely commercial, under similar sharing conditions. Please contact the maintainers for licensing details.

See the LICENSE file for the complete terms.

## Dealing with sources

### Annotate sources

Request access to https://docs.google.com/spreadsheets/d/1R0-q5diheG3zXDBq8V2aoUGOQyRI6HuUisTf-4wTsWY/edit#gid=0

### Download sources

To use the `sources_downloader.py` script to download PDFs listed in the `sources.csv` file into the `/data` folder (which is gitignored), follow these steps:

1. Ensure that you have the `requests` library installed in your Python environment. If not, you can install it using pip:
   ```bash
   pip install requests
   ```

2. Navigate to the directory containing the `sources_downloader.py` script.

3. Run the script:
   ```bash
   python sources_downloader.py
   ```
   By default, this uses `sources/sources.csv` as the source file and `data` as the output folder.

   For custom paths:
   ```bash
   python sources_downloader.py --sourcefile your_custom_sources.csv --output_folder your_custom_folder
   ```

Note: The `/data` folder is specified in the `.gitignore` file, so the downloaded PDFs will not be tracked by Git.
