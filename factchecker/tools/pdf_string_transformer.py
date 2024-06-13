
import fitz  # PyMuPDF

def transform_pdf_to_txt(pdf_path):
    """
    Transform a PDF file into text.
    
    Args:
    - pdf_path (str): Path to the PDF file.
    
    Returns:
    - str: Extracted text from the PDF.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text