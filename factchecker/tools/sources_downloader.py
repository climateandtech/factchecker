import argparse
import csv
import logging
import os

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

    def __init__(self, main_output_folder: str = "data/sources") -> None:
        """
        Initialize the SourcesDownloader with the main output folder.

        Args:
            main_output_folder (str): The main folder where source documents will be stored.
                                      If the folder does not exist, it will be created.

        """
        self.main_output_folder = main_output_folder
        if not os.path.exists(self.main_output_folder):
            os.makedirs(self.main_output_folder)

    def download_pdf(self, url: str, output_folder: str, output_filename: str) -> None:
        """
        Download a source document (PDF) from a given URL and save it to the specified folder.
        
        These documents are used as primary sources for fact-checking claims. Each document
        is saved with a specific title that corresponds to its entry in the claims database.
        
        Args:
            url (str): The URL of the source document to download.
            output_folder (str): The folder path where the source document should be saved.
            output_filename (str): The filename to use for the saved document.
            
        Raises:
            KeyError: If the specified url_column does not exist in the CSV file.
        
        Returns:
            None
        """
        # Ensure the filename ends with '.pdf'
        if not output_filename.lower().endswith('.pdf'):
            output_filename += '.pdf'
        
        response = requests.get(url)
        if response.status_code == 200:
            pdf_path = os.path.join(output_folder, output_filename)
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Downloaded {output_filename} to {output_folder}")
        else:
            logging.error(f"Failed to download {url} to {output_folder}/{output_filename}")

    def download_pdfs_from_csv(
            self, 
            sourcefile: str, 
            rows: list[int] | None, 
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
            rows (list[int], optional): List of specific claim indices to download sources for.
                                        If None, downloads sources for all claims.
            url_column (str): Name of the column containing source document URLs.
            output_filename_column (str): Name of the column specifying the filename for the downloaded file
            output_subfolder_column (str): Name of the column specifying the subfolder where to download each file

        Raises:
            KeyError: If the specified url_column does not exist in the CSV file.
        
        Returns:
            list[str]: A list of file paths for the downloaded documents.

        """
        downloaded_files = []
        with open(sourcefile, 'r') as csvfile:
            reader = csv.DictReader(csvfile, skipinitialspace=True)
            for i, row in enumerate(reader):
                if rows and i not in rows:
                    continue
                try:
                    url = row[url_column]
                    output_filename = row.get(output_filename_column, f"document_{i}.pdf")
                    subfolder = row.get(output_subfolder_column, "").strip()
                    output_folder = os.path.join(self.main_output_folder, subfolder) if subfolder else self.main_output_folder
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    self.download_pdf(url, output_folder, output_filename)
                    downloaded_files.append(os.path.join(output_folder, output_filename))
                except KeyError:
                    logging.error(f"Column {url_column} does not exist in the CSV file.")
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
            --rows: Specific claim indices to download sources for (optional, 0-indexed)
            --url_column: Name of the column containing source URLs (default: 'external_link')
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
            '--rows', type=int, nargs='*',
            help='Specify which claims to download sources for (0-indexed).'
        )
        parser.add_argument(
            '--url_column', type=str, default='external_link',
            help='Specify the column containing source URLs.'
        )
        parser.add_argument(
            '--output_folder', type=str, default='data',
            help='Main output folder for the downloaded source documents.'
        )
        args = parser.parse_args()

        downloader = SourcesDownloader(args.output_folder)
        downloader.download_pdfs_from_csv(args.sourcefile, args.rows, args.url_column)

if __name__ == "__main__":
    SourcesDownloader.run_cli()