import logging
import os
from unittest.mock import Mock, mock_open, patch

import requests

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

def test_output_folder_exists():
    """Test that existing output folders are handled correctly"""
    mock_args = Mock(
        sourcefile='test.csv',
        output_folder='test_data',
        row_indices=None,
        url_column='external_link',
        output_filename_column='pdf_title',
        output_subfolder_column='output_subfolder',
        limit=None,
        no_skip_existing=False,
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
            'test.csv', None, 'external_link', 'pdf_title', 'output_subfolder',
            skip_existing=True, max_sources=None
        )


# Test the CLI argument parsing
def test_cli_arguments():
    testargs = ["prog", "--sourcefile", "test.csv", "--row_indices", "1", "2", "--url_column", "test_url", "--output_folder", "test_data"]
    with patch('sys.argv', testargs):
        with patch('factchecker.tools.sources_downloader.SourcesDownloader.download_pdfs_from_csv') as mock_download:
            SourcesDownloader.run_cli()
            # The row_indices parameter should now be parsed as [1, 2]; defaults include pdf_title, skip_existing, max_sources
            mock_download.assert_called_once_with(
                'test.csv', [1, 2], 'test_url', 'pdf_title', 'output_subfolder',
                skip_existing=True, max_sources=None
            )


# --- Skip existing files (TDD) ---
def test_skip_existing_file(tmp_path):
    """When file already exists and skip_existing=True, download_pdf is not called."""
    csv_path = tmp_path / "sources.csv"
    csv_path.write_text("url,output_filename\nhttp://example.com/a.pdf,existing.pdf")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    existing_pdf = out_dir / "existing.pdf"
    existing_pdf.write_bytes(b"existing content")

    downloader = SourcesDownloader(str(out_dir))
    with patch.object(downloader, 'download_pdf') as mock_download:
        result = downloader.download_pdfs_from_csv(
            str(csv_path),
            url_column="url",
            output_filename_column="output_filename",
            skip_existing=True,
        )
    mock_download.assert_not_called()
    assert result == [str(existing_pdf)]


def test_download_when_file_missing(tmp_path):
    """When file does not exist, download_pdf is called."""
    csv_path = tmp_path / "sources.csv"
    csv_path.write_text("url,output_filename\nhttp://example.com/a.pdf,missing.pdf")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    downloader = SourcesDownloader(str(out_dir))
    with patch.object(downloader, 'download_pdf', return_value=True) as mock_download:
        downloader.download_pdfs_from_csv(
            str(csv_path),
            url_column="url",
            output_filename_column="output_filename",
            skip_existing=True,
        )
    mock_download.assert_called_once()
    call_args = mock_download.call_args[0]
    assert call_args[2] == "missing.pdf"


def test_skip_existing_false(tmp_path):
    """When skip_existing=False, download is attempted even if file exists."""
    csv_path = tmp_path / "sources.csv"
    csv_path.write_text("url,output_filename\nhttp://example.com/a.pdf,existing.pdf")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "existing.pdf").write_bytes(b"old")

    downloader = SourcesDownloader(str(out_dir))
    with patch.object(downloader, 'download_pdf', return_value=True) as mock_download:
        downloader.download_pdfs_from_csv(
            str(csv_path),
            url_column="url",
            output_filename_column="output_filename",
            skip_existing=False,
        )
    mock_download.assert_called_once()


# --- max_sources (TDD) ---
def test_max_sources_limits_downloads(tmp_path):
    """With max_sources=2, at most 2 downloads are attempted."""
    csv_path = tmp_path / "sources.csv"
    csv_path.write_text(
        "url,output_filename\n"
        "http://a.com/1.pdf,one.pdf\n"
        "http://a.com/2.pdf,two.pdf\n"
        "http://a.com/3.pdf,three.pdf\n"
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    downloader = SourcesDownloader(str(out_dir))
    with patch.object(downloader, 'download_pdf', return_value=True) as mock_download:
        result = downloader.download_pdfs_from_csv(
            str(csv_path),
            url_column="url",
            output_filename_column="output_filename",
            max_sources=2,
        )
    assert mock_download.call_count == 2
    assert len(result) == 2


def test_max_sources_none(tmp_path):
    """With max_sources=None, all rows are processed."""
    csv_path = tmp_path / "sources.csv"
    csv_path.write_text(
        "url,output_filename\n"
        "http://a.com/1.pdf,one.pdf\n"
        "http://a.com/2.pdf,two.pdf\n"
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    downloader = SourcesDownloader(str(out_dir))
    with patch.object(downloader, 'download_pdf', return_value=True) as mock_download:
        result = downloader.download_pdfs_from_csv(
            str(csv_path),
            url_column="url",
            output_filename_column="output_filename",
            max_sources=None,
        )
    assert mock_download.call_count == 2
    assert len(result) == 2
