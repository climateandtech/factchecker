import os
from unittest.mock import Mock, mock_open, patch

import pytest

from factchecker.tools.sources_downloader import SourcesDownloader


# Test for download_pdf function of Sources Downloader
def test_download_pdf_success():
    downloader = SourcesDownloader("output_folder")
    # Mock the requests.get call to return a response with status_code 200
    with patch('factchecker.tools.sources_downloader.requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b'PDF content'

        # Mock the open function to simulate file writing
        with patch('builtins.open', mock_open()) as mock_file:
            downloader.download_pdf('http://example.com/pdf', 'output_folder', 'test.pdf')

            # Check if the file was opened in write-binary mode
            mock_file.assert_called_with(os.path.join('output_folder', 'test.pdf'), 'wb')

            # Check if the content was written to the file
            mock_file().write.assert_called_once_with(b'PDF content')


def test_download_pdf_failure(capfd):
    downloader = SourcesDownloader("output_folder")
    with patch('factchecker.tools.sources_downloader.requests.get') as mock_get:
        mock_get.return_value.status_code = 404
        downloader.download_pdf('http://example.com/pdf', 'output_folder', 'test.pdf')
        out, _ = capfd.readouterr()
        assert "Failed to download test.pdf" in out

def test_output_folder_creation():
    testargs = ["prog", "--output_folder", "test_data"]
    with patch('sys.argv', testargs), \
         patch('os.path.exists', return_value=False), \
         patch('os.makedirs') as mock_makedirs, \
         patch('builtins.open', mock_open()) as mock_file:
        # Call the CLI entry point
        SourcesDownloader.run_cli()
        mock_makedirs.assert_called_once_with('test_data')

def test_output_folder_exists():
    """Test that existing output folders are handled correctly"""
    mock_args = Mock(
        sourcefile='test.csv',
        output_folder='test_data',
        rows=None,
        url_column='external_link'
    )
    
    # Patch argparse to return our mock arguments.
    with patch('gettext.translation'), \
         patch('argparse.ArgumentParser.parse_args', return_value=mock_args), \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs') as mock_makedirs, \
         patch('factchecker.tools.sources_downloader.SourcesDownloader.download_pdfs_from_csv') as mock_download:
            
        SourcesDownloader.run_cli()
        mock_makedirs.assert_not_called()
        # Since the output folder is passed to the constructor, download_pdfs_from_csv is called with sourcefile, rows, and url_column.
        mock_download.assert_called_once_with('test.csv', None, 'external_link')


# Test the CLI argument parsing
def test_cli_arguments():
    testargs = ["prog", "--sourcefile", "test.csv", "--rows", "1", "2", "--url_column", "test_url", "--output_folder", "test_data"]
    with patch('sys.argv', testargs):
        with patch('factchecker.tools.sources_downloader.SourcesDownloader.download_pdfs_from_csv') as mock_download:
            SourcesDownloader.run_cli()
            # Assert that the parsed arguments are passed correctly.
            mock_download.assert_called_once_with('test.csv', [1, 2], 'test_url')
