import argparse
import csv
import logging
import os
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

class SourcesDownloader:
    """
    Script for downloading source documents used in fact-checking claims.

    This class provides functionality to download source documents (PDFs) that are referenced
    in fact-checking claims. These sources are listed in a CSV file along with their metadata
    and are essential for verifying factual accuracy. The class supports both batch downloads
    of all sources and selective downloads of specific entries.

    The downloaded documents serve as the knowledge base for the fact-checking system,
    enabling verification of claims against original sources.
    """

    def __init__(self, output_folder: str = "data/sources") -> None:
        """
        Initialize the SourcesDownloader with the main output folder.

        Args:
            output_folder (str): The main folder where source documents will be stored.
                                      If the folder does not exist, it will be created.

        """
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid before attempting to download from it.
        
        Args:
            url (str): URL to validate
            
        Returns:
            bool: True if the URL is valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def download_pdf(self, url: str, output_folder: str, output_filename: str) -> bool:
        """
        Download a source document (PDF) from a given URL and save it to the specified folder.
        
        These documents are used as primary sources for fact-checking claims. Each document
        is saved with a specific title that corresponds to its entry in the claims database.
        
        Args:
            url (str): The URL of the source document to download.
            output_folder (str): The folder path where the source document should be saved.
            output_filename (str): The filename to use for the saved document.
            
        Returns:
            bool: True if download was successful or file already exists, False otherwise
        """
        # Validate URL before attempting download
        if not self._is_valid_url(url):
            logger.error(f"Invalid URL format: {url}")
            return False
            
        # Ensure the filename ends with '.pdf'
        if not output_filename.lower().endswith('.pdf'):
            output_filename += '.pdf'
            
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        output_path = os.path.join(output_folder, output_filename)
        
        # Check if file already exists
        if os.path.exists(output_path):
            logger.info(f"File already exists, skipping: {output_path}")
            return True
            
        try:
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save the file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
            logger.info(f"Downloaded {output_filename} to {output_folder}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            # Clean up partially downloaded file if it exists
            if os.path.exists(output_path):
                os.remove(output_path)
            return False

    def download_pdfs_from_csv(
            self, 
            sourcefile: str = "sources/sources.csv", 
            row_indices: list[int] | None = None, 
            url_column: str = "url",
            output_filename_column: str = "output_filename",
            output_subfolder_column: str = "output_subfolder",
        ) -> list[str]:
        """
        Download source documents from URLs specified in a claims database CSV file.
        
        This function processes a CSV file containing fact-checking claims and their associated
        source documents. It downloads the source documents that support or are referenced by the claims,
        maintaining the connection between claims and their supporting evidence.
        
        Args:
            sourcefile (str): Path to the CSV file containing claims and their source URLs.
            row_indices (list[int], optional): List of specific claim indices to download sources for.
                                        If None, downloads sources for all claims.
            url_column (str): Name of the column containing source document URLs.
            output_filename_column (str): Name of the column specifying the filename for the downloaded file
            output_subfolder_column (str): Name of the column specifying the subfolder where to download each file

        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
            KeyError: If the specified url_column does not exist in the CSV file.
        
        Returns:
            list[str]: A list of file paths for all available documents (both downloaded and existing).

        """
        downloaded_files = []
        
        # Check if file exists before attempting to open it
        if not os.path.isfile(sourcefile):
            logger.error(f"Source file not found: {sourcefile}")
            raise FileNotFoundError(f"Source file not found: {sourcefile}")
            
        try:
            with open(sourcefile, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Verify required columns exist
                if url_column not in reader.fieldnames:
                    raise KeyError(f"URL column '{url_column}' not found in CSV")
                if output_filename_column not in reader.fieldnames:
                    raise KeyError(f"Output filename column '{output_filename_column}' not found in CSV")
                if output_subfolder_column not in reader.fieldnames:
                    raise KeyError(f"Output subfolder column '{output_subfolder_column}' not found in CSV")
                
                # Process each row
                for idx, row in enumerate(reader):
                    if row_indices is not None and idx not in row_indices:
                        continue
                        
                    url = row[url_column].strip()
                    output_filename = row[output_filename_column].strip()
                    output_subfolder = row[output_subfolder_column].strip()
                    
                    if not url or not output_filename:
                        logger.warning(f"Skipping row {idx}: Missing URL or filename")
                        continue
                        
                    # Create full output path
                    output_folder = os.path.join(self.output_folder, output_subfolder)
                    output_path = os.path.join(output_folder, output_filename)
                    
                    # Check if file already exists
                    if os.path.exists(output_path):
                        logger.info(f"File already exists: {output_path}")
                        downloaded_files.append(output_path)
                        continue
                    
                    # Download the file
                    if self.download_pdf(url, output_folder, output_filename):
                        downloaded_files.append(output_path)
                    
        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            raise
            
        return downloaded_files

    @staticmethod
    def run_cli() -> None:
        """
        Main function to handle command-line arguments and initiate source document downloads.
        
        This script is used to build the knowledge base for fact-checking by downloading
        all referenced source documents. These documents are then used by the fact-checking
        system to verify claims against original sources.
        
        Command-line Arguments:
            --sourcefile: Path to the CSV file containing claims and source links (default: 'sources/sources.csv')
            --row_indices: Specific claim indices to download sources for (optional, 0-indexed)
            --url_column: Name of the column containing source URLs (default: 'external_link')
            --output_filename_column: Name of the column specifying filename for downloaded file (default: 'output_filename')
            --output_subfolder_column: Name of the column specifying subfolder for each file (default: 'output_subfolder')
            --output_folder: Main output folder for the downloaded source documents (default: 'data')
        
        Returns:
            None
            
        """
        parser = argparse.ArgumentParser(
            description="Download source documents for fact-checking from a CSV file."
        )
        parser.add_argument(
            '--sourcefile', type=str, default='sources/sources.csv',
            help='Path to the CSV file containing the source document links.'
        )
        parser.add_argument(
            '--row_indices', type=int, nargs='*',
            help='Specify which claims to download sources for (0-indexed).'
        )
        parser.add_argument(
            '--url_column', type=str, default='url',
            help='Specify the column containing source URLs.'
        )
        parser.add_argument(
            '--output_filename_column', type=str, default='output_filename',
            help='Specify the column containing output filenames.'
        )
        parser.add_argument(
            '--output_subfolder_column', type=str, default='output_subfolder',
            help='Specify the column containing output subfolders.'
        )
        parser.add_argument(
            '--output_folder', type=str, default='data',
            help='Main output folder for downloaded documents.'
        )
        
        args = parser.parse_args()
        
        downloader = SourcesDownloader(output_folder=args.output_folder)
        downloader.download_pdfs_from_csv(
            sourcefile=args.sourcefile,
            row_indices=args.row_indices,
            url_column=args.url_column,
            output_filename_column=args.output_filename_column,
            output_subfolder_column=args.output_subfolder_column
        )

if __name__ == "__main__":
    SourcesDownloader.run_cli()