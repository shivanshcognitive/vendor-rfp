"""
pdf_tool.py
Document Tool: extracts clean text from an uploaded supplier RFP PDF.
"""

import pdfplumber
import io


def extract_text_from_pdf(file_like) -> str:
    """
    Accepts a file path (str) or a file-like object (e.g. Streamlit's
    UploadedFile, or a BytesIO buffer) and returns concatenated, cleaned
    plain text from every page.
    """
    text_parts = []

    if isinstance(file_like, (bytes, bytearray)):
        file_like = io.BytesIO(file_like)
    elif hasattr(file_like, "read") and not hasattr(file_like, "seek"):
        file_like = io.BytesIO(file_like.read())

    if hasattr(file_like, "seek"):
        file_like.seek(0)

    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)

    full_text = "\n".join(text_parts)
    # Light cleanup: collapse excessive whitespace but keep line structure
    lines = [ln.strip() for ln in full_text.splitlines()]
    lines = [ln for ln in lines if ln]  # drop empty lines
    return "\n".join(lines)


def extract_text_from_path(path: str) -> str:
    with open(path, "rb") as f:
        return extract_text_from_pdf(f)
