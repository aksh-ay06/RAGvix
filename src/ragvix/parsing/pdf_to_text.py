"""PDF to text extraction using PyMuPDF."""

from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from ragvix.utils.logging import get_logger

logger = get_logger(__name__)


def extract_text(pdf_path: Path) -> str:
    """Extract text from PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text content
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF processing fails
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Extracting text from: {pdf_path.name}")
    
    try:
        doc = fitz.open(pdf_path)
        text_content = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content += page.get_text()
            text_content += "\n\n"  # Add page separator
        
        doc.close()
        
        logger.info(f"Extracted {len(text_content)} characters from {len(doc)} pages")
        return text_content.strip()
        
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
        raise


def extract_text_with_metadata(pdf_path: Path) -> dict:
    """Extract text and metadata from PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with text content and metadata
    """
    text = extract_text(pdf_path)
    
    # Basic metadata stub - can be expanded
    metadata = {
        "filename": pdf_path.name,
        "file_size": pdf_path.stat().st_size,
        "text_length": len(text),
        "extraction_method": "pymupdf",
    }
    
    return {
        "text": text,
        "metadata": metadata,
    }