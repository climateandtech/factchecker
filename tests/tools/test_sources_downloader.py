import os
import pytest
from unittest.mock import mock_open, patch
from factchecker.tools.sources_downloader import download_pdf
from unittest.mock import patch, mock_open, Mock
from factchecker.tools.sources_downloader import main

# Test for download_pdf function
def test_download_pdf_success():
    # Mock the requests.get call to return a response with status_code 200
    with patch('factchecker.tools.sources_downloader.requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b'PDF content'

        # Mock the open function to simulate file writing
        with patch('builtins.open', mock_open()) as mock_file:
            download_pdf('http://example.com/pdf', 'output_folder', 'test.pdf')

            # Check if the file was opened in write-binary mode
            mock_file.assert_called_with(os.path.join('output_folder', 'test.pdf'), 'wb')

            # Check if the content was written to the file
            mock_file().write.assert_called_once_with(b'PDF content')

            # Check if the file was opened in write-binary mode
            mock_file.assert_called_with(os.path.join('output_folder', 'test.pdf'), 'wb')

def test_download_pdf_failure(capfd):
    with patch('factchecker.tools.sources_downloader.requests.get') as mock_get:
        mock_get.return_value.status_code = 404
        download_pdf('http://example.com/pdf', 'output_folder', 'test.pdf')
        out, _ = capfd.readouterr()
        assert "Failed to download test.pdf" in out

def test_output_folder_creation():
    testargs = ["prog", "--output_folder", "test_data"]
    with patch('sys.argv', testargs), \
         patch('os.path.exists', return_value=False), \
         patch('os.makedirs') as mock_makedirs, \
         patch('builtins.open', mock_open()) as mock_file:
        main()
        mock_makedirs.assert_called_once_with('test_data')
        mock_file.assert_called()  # Add more specific assertions if necessary

def test_output_folder_exists():
    """Test that existing output folders are handled correctly"""
    mock_args = Mock(
        sourcefile='test.csv',
        output_folder='test_data',
        rows=None,
        url_column='external_link'
    )
    
    # Add patch for gettext to prevent translation file reading
    with patch('gettext.translation'), \
         patch('argparse.ArgumentParser.parse_args', return_value=mock_args), \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs') as mock_makedirs, \
         patch('factchecker.tools.sources_downloader.download_from_csv') as mock_download:
            
        main()
        mock_makedirs.assert_not_called()
        mock_download.assert_called_once_with(
            'test.csv', None, 'external_link', 'test_data'
        )

def test_output_folder_exists():
    """
    Test behavior when output folder already exists.
    
    Verifies that:
    - No attempt is made to create an existing folder
    - The file operations proceed as expected
    """
    mock_csv_content = "url_column,pdf_title\nhttp://example.com,test.pdf"
    mock_args = type('Args', (), {
        'sourcefile': 'sources/sources.csv',
        'rows': None,
        'url_column': 'external_link',
        'output_folder': 'test_data'
    })()
    
    mock_parser = Mock()
    mock_parser.parse_args.return_value = mock_args
    
    with patch('argparse.ArgumentParser', return_value=mock_parser), \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs') as mock_makedirs, \
         patch('builtins.open', mock_open(read_data=mock_csv_content)) as mock_file:
        main()
        mock_makedirs.assert_not_called()
        mock_file.assert_called()

# Test the CLI argument parsing
def test_cli_arguments():
    testargs = ["prog", "--sourcefile", "test.csv", "--rows", "1", "2", "--url_column", "test_url", "--output_folder", "test_data"]
    with patch('sys.argv', testargs):
        with patch('factchecker.tools.sources_downloader.download_from_csv') as mock_download:
            main()
            mock_download.assert_called_once_with('test.csv', [1, 2], 'test_url', 'test_data')
