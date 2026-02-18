import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import types
from unittest.mock import Mock, mock_open, patch

import requests

# Import SourcesDownloader without loading factchecker package (avoids openai/llama_index/numpy)
# Prefer loading by path so tests run in minimal envs and avoid venv numpy segfaults on some systems.
_test_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_test_dir))
_module_path = os.path.join(_project_root, "factchecker", "tools", "sources_downloader.py")
if os.path.isfile(_module_path):
    for _pkg in ("factchecker", "factchecker.tools"):
        if _pkg not in sys.modules:
            sys.modules[_pkg] = types.ModuleType(_pkg)
    _spec = importlib.util.spec_from_file_location(
        "factchecker.tools.sources_downloader", _module_path
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["factchecker.tools.sources_downloader"] = _mod
    _spec.loader.exec_module(_mod)
    sys.modules["factchecker"].tools = sys.modules["factchecker.tools"]
    sys.modules["factchecker.tools"].sources_downloader = _mod
    SourcesDownloader = _mod.SourcesDownloader
else:
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


def test_download_pdf_failure(caplog):
    downloader = SourcesDownloader("output_folder")
    with patch('factchecker.tools.sources_downloader.requests.get') as mock_get:
        # Configure the mock to simulate a 404 response
        mock_get.return_value.status_code = 404
        mock_get.return_value.content = b''  # Ensure a bytes object is provided
        # Simulate raise_for_status() raising an HTTPError
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")
    
        with caplog.at_level(logging.ERROR, logger="factchecker.tools.sources_downloader"):
            downloader.download_pdf('http://example.com/pdf', 'output_folder', 'test.pdf')
            # Expect the error log message to contain "HTTP error occurred"
            assert "HTTP error occurred" in caplog.text

def test_output_folder_creation():
    testargs = ["prog", "--output_folder", "test_data"]
    with patch('sys.argv', testargs), \
         patch('os.path.exists', return_value=False), \
         patch('os.makedirs') as mock_makedirs, \
         patch('builtins.open', mock_open()) as mock_file:
        # Call the CLI entry point
        SourcesDownloader.run_cli()
        mock_makedirs.assert_called_once_with('test_data')

def test_download_pdfs_from_csv_with_project_format():
    """With project CSV format (external_link, pdf_title), defaults work without overrides."""
    csv_content = "external_link,title,pdf_title\nhttps://example.com/doc.pdf,My Doc,mydoc.pdf\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_path = f.name
    try:
        out_dir = tempfile.mkdtemp()
        try:
            downloader = SourcesDownloader(out_dir)
            with patch('factchecker.tools.sources_downloader.requests.get') as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.content = b'pdf'
                mock_get.return_value.raise_for_status = Mock()
                downloader.download_pdfs_from_csv(csv_path)
                mock_get.assert_called_once_with('https://example.com/doc.pdf', timeout=30)
                assert os.path.isfile(os.path.join(out_dir, 'mydoc.pdf'))
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
    finally:
        os.unlink(csv_path)


def test_empty_pdf_title_derives_from_url():
    """When pdf_title is empty, filename is derived from URL path."""
    csv_content = "external_link,title,pdf_title\nhttps://example.com/reports/IPCC_AR6_SYR_LongerReport.pdf,IPCC,\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_path = f.name
    try:
        out_dir = tempfile.mkdtemp()
        try:
            downloader = SourcesDownloader(out_dir)
            with patch('factchecker.tools.sources_downloader.requests.get') as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.content = b'pdf'
                mock_get.return_value.raise_for_status = Mock()
                downloader.download_pdfs_from_csv(csv_path)
                expected_path = os.path.join(out_dir, 'IPCC_AR6_SYR_LongerReport.pdf')
                assert os.path.isfile(expected_path)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
    finally:
        os.unlink(csv_path)


def test_cli_defaults_match_project_csv():
    """CLI with no column flags passes external_link and pdf_title as defaults."""
    with patch('factchecker.tools.sources_downloader.SourcesDownloader.download_pdfs_from_csv') as mock_download:
        with patch('sys.argv', ['prog']):
            SourcesDownloader.run_cli()
        mock_download.assert_called_once()
        call_kw = mock_download.call_args
        assert call_kw[0][2] == 'external_link'
        assert call_kw[0][3] == 'pdf_title'
        assert call_kw[0][4] == 'output_subfolder'


def test_output_folder_exists():
    """Test that existing output folders are handled correctly"""
    mock_args = Mock(
        sourcefile='test.csv',
        output_folder='test_data',
        row_indices=None,
        url_column='external_link',
        output_filename_column='pdf_title',
        output_subfolder_column='output_subfolder'
    )
    
    # Patch argparse to return our mock arguments.
    with patch('gettext.translation'), \
         patch('argparse.ArgumentParser.parse_args', return_value=mock_args), \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs') as mock_makedirs, \
         patch('factchecker.tools.sources_downloader.SourcesDownloader.download_pdfs_from_csv') as mock_download:
            
        SourcesDownloader.run_cli()
        mock_makedirs.assert_not_called()
        mock_download.assert_called_once_with(
            'test.csv', None, 'external_link', 'pdf_title', 'output_subfolder'
        )


# Test the CLI argument parsing
def test_cli_arguments():
    testargs = ["prog", "--sourcefile", "test.csv", "--row_indices", "1", "2", "--url_column", "test_url", "--output_folder", "test_data"]
    with patch('sys.argv', testargs):
        with patch('factchecker.tools.sources_downloader.SourcesDownloader.download_pdfs_from_csv') as mock_download:
            SourcesDownloader.run_cli()
            # The row_indices parameter should now be parsed as [1, 2]; output_filename_column keeps default pdf_title
            mock_download.assert_called_once_with('test.csv', [1, 2], 'test_url', 'pdf_title', 'output_subfolder')
