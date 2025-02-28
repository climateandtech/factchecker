# Climate+Tech FactChecker

Climate+Tech FactChecker is designed to serve as a comprehensive toolkit for both experimentation and production environments, focusing on the verification of claims. It offers a robust suite of tools and methodologies to assist researchers, developers, and practitioners in the field of claim verification, enabling them to efficiently test hypotheses, validate data, and deploy reliable fact-checking solutions.


## License

This project is dual-licensed:

- For researchers, academic institutions, universities, and fact-checking organizations: This project is available under the GNU Affero General Public License (AGPL), with the requirement that any results and improvements are shared back with the community.

- For other organizations: A license can be requested and can be granted free if the project's use is not purely commercial, under similar sharing conditions. Please contact the maintainers for licensing details.

See the LICENSE file for the complete terms.


## Installation

     python3 -m venv venv
     source venv/bin/activate

     pip install -r requirements.txt

     # Install a package

     pip install some-package

     # Update requirements.txt

     pip freeze > requirements.txt


## Contribution

You accept the CONTRIBUTOR_LICENSE_AGREEMENT by contributing. 



## Dealing with sources

### Source Configuration & UI Integration

The FactChecker uses an advocate-mediator system where each source gets its own advocate. Sources are handled through several key functions:

1. **Source Discovery** (`get_source_choices()`):
   - Automatically scans the `/data` directory for available sources
   - Returns file names (without extensions) as source options
   - These appear in the UI as checkboxes under "Select Advocates"
   - Falls back to "ipcc_ar6_wg1" if no sources are found

2. **Source Configuration** (`get_strategy()`):
   - Creates an indexer for each selected source
   - Configurable parameters include:
     - Chunk size (50-1000, default 150)
     - Chunk overlap (0-100, default 20)
     - Top K results (1-20, default 8)
     - Minimum similarity score (0.1-1.0, default 0.75)

3. **Source Management**:
   - Sources can be added through the UI using the download functionality
   - Each source gets its own dedicated index for efficient retrieval
   - Sources are stored in the `/data` directory (gitignored)

### UI Components

The source configuration is available in the "Settings & Sources" tab of the web interface, where you can:
- Select which sources to use as advocates
- Configure retrieval and processing parameters
- Apply settings for immediate use

### Annotate sources

Request access to https://docs.google.com/spreadsheets/d/1R0-q5diheG3zXDBq8V2aoUGOQyRI6HuUisTf-4wTsWY/edit#gid=0





# Download sources


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
