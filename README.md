# Climate+Tech FactChecker

Climate+Tech FactChecker is designed to serve as a comprehensive toolkit for both experimentation and production environments, focusing on the verification of claims. It offers a robust suite of tools and methodologies to assist researchers, developers, and practitioners in the field of claim verification, enabling them to efficiently test hypotheses, validate data, and deploy reliable fact-checking solutions.

## Installation

     python3 -m venv venv
     source venv/bin/activate

     pip install -r requirements.txt

     # Install a package

     pip install some-package

     # Update requirements.txt

     pip freeze > requirements.txt

## Configuration

### Setting Up Environment Variables

Before running the Climate+Tech FactChecker, you need to configure your environment variables. This is done using the `.env.example` file provided in the repository.

1. **Copy the `.env.example` file to a new file named `.env`:**

   ```
   cp .env.example .env
   ```

2. **Edit the `.env` file to include your specific configuration:**

   - Replace `your_openai_api_key_here` with your actual OpenAI API key.
   - Update `OPENAI_ORGANIZATION` with your OpenAI organization ID if applicable.
   - Adjust other variables like `LLM_TYPE`, `OLLAMA_API_BASE_URL`, and `OLLAMA_MODEL` as needed for your setup.

### Configuring the LLM

The `llm.py` file in the `factchecker/core` directory is responsible for loading the language model (LLM) based on your configuration.

- **LLM Type:** The `LLM_TYPE` environment variable determines which LLM to use. It can be set to either `ollama` or `openai`.
- **Model and API Settings:** Depending on the `LLM_TYPE`, ensure that the corresponding model and API settings are correctly configured in your `.env` file. For example, if using OpenAI, ensure `OPENAI_API_KEY` and `OPENAI_API_BASE` are set.

## Contribution

You accept the CONTRIBUTOR_LICENSE_AGREEMENT by contributing.

## Dealing with sources

### Annotate sources

Request access to https://docs.google.com/spreadsheets/d/1R0-q5diheG3zXDBq8V2aoUGOQyRI6HuUisTf-4wTsWY/edit#gid=0

### Download sources

To use the `sources_downloader.py` script to download PDFs listed in the `sources.csv` file into the `/data` folder (which is gitignored), follow these steps:

1. Ensure that you have the `requests` library installed in your Python environment. If not, you can install it using pip:

   ```
   pip install requests
   ```

2. Navigate to the directory containing the `sources_downloader.py` script.

3. Run the script with the following command:

   ```
   python sources_downloader.py
   ```

   By default, this command will use `sources/sources.csv` as the source file and `data` as the output folder. These parameters are optional and can be customized as needed.

   If you wish to specify a different CSV file or output folder, you can use the `--sourcefile` and `--output_folder` flags, respectively:

   ```
   python sources_downloader.py --sourcefile your_custom_sources.csv --output_folder your_custom_folder
   ```

4. The script will download each PDF listed in the CSV file and save it to the specified output folder with the title provided in the CSV or a default name if the title is not provided.

Note: The `/data` folder is specified in the `.gitignore` file, so the downloaded PDFs will not be tracked by Git.

## Run Example

| python3 -m factchecker.experiments.evidence_evaluation_1

## Testing

To write tests, follow the convention of creating basic test cases for each function in the corresponding `tests/` directory file. Ensure each test function name starts with `test_`.

To run the tests, execute `pytest` in the project's root directory. This will discover and run all test files named `test_*.py`.

For more information on writing and running tests, refer to the `pytest` documentation.
