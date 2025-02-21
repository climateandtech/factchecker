"""
Script for downloading source documents used in fact-checking claims.

This module provides functionality to download source documents (PDFs) that are referenced
in fact-checking claims. These sources are listed in a CSV file along with their claims
and are essential for verifying factual accuracy. The module supports both batch downloads
of all sources and selective downloads of specific entries.

The downloaded documents serve as the knowledge base for the fact-checking system,
enabling verification of claims against original sources.
"""

import argparse
import csv
import os
import requests

def download_pdf(url, output_folder, pdf_title):
    """
    Download a source document (PDF) from a given URL and save it to the specified folder.
    
    These documents are used as primary sources for fact-checking claims. Each document
    is saved with a specific title that corresponds to its entry in the claims database.
    
    Args:
        url (str): The URL of the source document to download.
        output_folder (str): The folder path where the source document should be saved.
        pdf_title (str): The filename to use for the saved document, preserving the link
                        to its corresponding claim in the database.
        
    Returns:
        None
    
    Prints:
        Success or failure message for the download operation.
    """
    response = requests.get(url)
    if response.status_code == 200:
        pdf_path = os.path.join(output_folder, pdf_title)
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {pdf_title}")
    else:
        print(f"Failed to download {pdf_title}")

def download_from_csv(sourcefile, rows, url_column, output_folder):
    """
    Download source documents from URLs specified in a claims database CSV file.
    
    This function processes a CSV file containing fact-checking claims and their
    associated source documents. It downloads the source documents that support
    or are referenced by the claims, maintaining the connection between claims
    and their supporting evidence.
    
    Args:
        sourcefile (str): Path to the CSV file containing claims and their source URLs.
        rows (list[int], optional): List of specific claim indices to download sources for.
                                  If None, downloads sources for all claims.
        url_column (str): Name of the column containing source document URLs.
        output_folder (str): Folder path where source documents should be saved.
        
    Returns:
        None
        
    Raises:
        KeyError: If the specified url_column does not exist in the CSV file.
    """
    with open(sourcefile, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if rows and i not in rows:
                continue
            try:
                url = row[url_column]
                # Use provided PDF title or generate a default one
                pdf_title = row['pdf_title'] if row['pdf_title'] else f"document_{i}.pdf"
                download_pdf(url, output_folder, pdf_title)
            except KeyError:
                print(f"Column {url_column} does not exist in the CSV file.")

def main():
    """
    Main function to handle command-line arguments and initiate source document downloads.
    
    This script is used to build the knowledge base for fact-checking by downloading
    all referenced source documents. These documents are then used by the fact-checking
    system to verify claims against original sources.
    
    Command-line Arguments:
        --sourcefile: Path to the CSV file containing claims and source links (default: 'sources/sources.csv')
        --rows: Specific claims to download sources for (optional, 0-indexed)
        --url_column: Name of the column containing source URLs (default: 'external_link')
        --output_folder: Output folder for downloaded source documents (default: 'data')
    """
    parser = argparse.ArgumentParser(description="Download source documents for fact-checking from a CSV file.")
    parser.add_argument('--sourcefile', type=str, default='sources/sources.csv', help='Path to the CSV file containing the source document links.')
    parser.add_argument('--rows', type=int, nargs='*', help='Specify which claims to download sources for (0-indexed).')
    parser.add_argument('--url_column', type=str, default='external_link', help='Specify the column containing source URLs.')
    parser.add_argument('--output_folder', type=str, default='data', help='Output folder for the downloaded source documents.')
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    download_from_csv(args.sourcefile, args.rows, args.url_column, args.output_folder)

if __name__ == "__main__":
    main()
