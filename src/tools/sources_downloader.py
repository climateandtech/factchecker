import argparse
import csv
import os
import requests

def download_pdf(url, output_folder, pdf_title):
    response = requests.get(url)
    if response.status_code == 200:
        pdf_path = os.path.join(output_folder, pdf_title)
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {pdf_title}")
    else:
        print(f"Failed to download {pdf_title}")

def download_from_csv(sourcefile, rows, url_column, output_folder):
    with open(sourcefile, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if rows and i not in rows:
                continue
            try:
                url = row[url_column]
                pdf_title = row['pdf_title'] if row['pdf_title'] else f"document_{i}.pdf"
                download_pdf(url, output_folder, pdf_title)
            except KeyError:
                print(f"Column {url_column} does not exist in the CSV file.")

def main():
    parser = argparse.ArgumentParser(description="Download PDFs from a CSV file.")
    parser.add_argument('--sourcefile', type=str, default='sources/sources.csv', help='Path to the CSV file containing the PDF links.')
    parser.add_argument('--rows', type=int, nargs='*', help='Specify the rows to download (0-indexed).')
    parser.add_argument('--url_column', type=str, default='external_link', help='Specify the column with the URL.')
    parser.add_argument('--output_folder', type=str, default='data', help='Output folder for the downloaded PDFs.')
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    download_from_csv(args.sourcefile, args.rows, args.url_column, args.output_folder)

if __name__ == "__main__":
    main()
