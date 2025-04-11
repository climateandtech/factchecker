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
            bool: True if download was successful, False otherwise
        """
        # Validate URL before attempting download
        if not self._is_valid_url(url):
            logger.error(f"Invalid URL format: {url}")
            return False
            
        # Ensure the filename ends with '.pdf'
        if not output_filename.lower().endswith('.pdf'):
            output_filename += '.pdf'
        
        try:
            # Add timeout to prevent hanging on slow servers
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            pdf_path = os.path.join(output_folder, output_filename)
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded {output_filename} to {output_folder}")
            return True
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out for {url}")
            return False
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error occurred for {url}: {e}")
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error occurred for {url}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error occurred during request for {url}: {e}")
            return False
        except IOError as e:
            logger.error(f"IO error occurred while saving {output_filename}: {e}")
            return False

    def download_pdfs_from_csv(
            self, 
            sourcefile: str, 
            row_indices: list[int] | None, 
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
            list[str]: A list of file paths for the downloaded documents.

        """
        downloaded_files = []
        
        # Check if file exists before attempting to open it
        if not os.path.isfile(sourcefile):
            logger.error(f"Source file not found: {sourcefile}")
            raise FileNotFoundError(f"Source file not found: {sourcefile}")
        
        with open(sourcefile, 'r') as csvfile:
            reader = csv.DictReader(csvfile, skipinitialspace=True)
            
            # Validate that required columns exist
            if reader.fieldnames and url_column not in reader.fieldnames:
                logger.error(f"Column {url_column} does not exist in the CSV file.")
                raise KeyError(f"Column {url_column} does not exist in the CSV file.")
            
            for i, row in enumerate(reader):
                if row_indices and i not in row_indices:
                    continue
                
                url = row.get(url_column, "").strip()
                if not url:
                    logger.warning(f"Empty URL in row {i}, skipping")
                    continue
                    
                output_filename = row.get(output_filename_column, f"document_{i}.pdf")
                subfolder = row.get(output_subfolder_column, "").strip()
                output_folder = os.path.join(self.output_folder, subfolder) if subfolder else self.output_folder
                
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                
                success = self.download_pdf(url, output_folder, output_filename)
                if success:
                    downloaded_files.append(os.path.join(output_folder, output_filename))
                
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
            '--url_column', type=str, default='external_link',
            help='Specify the column containing source URLs.'
        )
        parser.add_argument(
            '--output_filename_column', type=str, default='output_filename',
            help='Specify the column containing filenames for downloaded files.'
        )
        parser.add_argument(
            '--output_subfolder_column', type=str, default='output_subfolder',
            help='Specify the column containing subfolders for downloaded files.'
        )
        parser.add_argument(
            '--output_folder', type=str, default='data',
            help='Main output folder for the downloaded source documents.'
        )
        args = parser.parse_args()

        downloader = SourcesDownloader(args.output_folder)
        try:
            downloader.download_pdfs_from_csv(
                args.sourcefile, 
                args.row_indices, 
                args.url_column,
                args.output_filename_column,
                args.output_subfolder_column
            )
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Error: {e}")
            exit(1)

if __name__ == "__main__":
    SourcesDownloader.run_cli()