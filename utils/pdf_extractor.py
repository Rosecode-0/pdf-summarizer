from functools import lru_cache
import io
import pymupdf as fitz
import pytesseract
from PIL import Image


@lru_cache(maxsize=1)
def check_tesseract() -> bool:
    """Checks once whether Tesseract OCR is available on the system."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_page_with_fitz(page: fitz.Page, dpi: int = 150) -> str:
    """Renders a single PDF page in grayscale via PyMuPDF and runs Tesseract OCR."""
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    image = Image.frombytes("L", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(image)


def extract_text_from_pdf(
    pdf_bytes: bytes,
    max_pages: int | None = None,
    ocr_dpi: int = 150,
) -> tuple[str, int, bool, str | None]:
    """
    Extracts text from a PDF file using PyMuPDF.
    Only falls back to per-page grayscale OCR for pages without extractable text.
    Returns: (full_text, total_pages, ocr_used, error_message)
    """
    if not pdf_bytes:
        return "", 0, False, "Uploaded PDF file is empty."

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
            total_pages = pdf_document.page_count
            if total_pages == 0:
                return "", 0, False, "PDF contains no pages."

            pages_to_read = (
                min(total_pages, max_pages) if max_pages and max_pages > 0 else total_pages
            )

            page_texts: list[str] = []
            empty_page_indices: list[int] = []

            for page_num in range(pages_to_read):
                text = pdf_document[page_num].get_text()
                page_texts.append(text)
                if len(text.strip()) < 20:
                    empty_page_indices.append(page_num)

            combined_text = "\n".join(page_texts)
            ocr_used = False

            # Run OCR only if almost no digital text was extracted across the processed pages
            if len(combined_text.strip()) < 50 and empty_page_indices:
                if check_tesseract():
                    try:
                        for page_num in empty_page_indices:
                            ocr_text = _ocr_page_with_fitz(pdf_document[page_num], dpi=ocr_dpi)
                            page_texts[page_num] = ocr_text
                        combined_text = "\n".join(page_texts)
                        ocr_used = True
                    except Exception as ocr_error:
                        # Fallback to pdf2image if PyMuPDF pixmap OCR fails
                        try:
                            from pdf2image import convert_from_bytes

                            images = convert_from_bytes(
                                pdf_bytes,
                                dpi=ocr_dpi,
                                grayscale=True,
                                first_page=1,
                                last_page=pages_to_read,
                            )
                            ocr_pages = [pytesseract.image_to_string(img) for img in images]
                            combined_text = "\n".join(ocr_pages)
                            ocr_used = True
                        except Exception:
                            return "", total_pages, False, f"OCR processing failed: {ocr_error}"
                else:
                    return (
                        "",
                        total_pages,
                        False,
                        "No extractable text found and Tesseract OCR is not available.",
                    )

            return combined_text, total_pages, ocr_used, None
    except Exception as e:
        return "", 0, False, f"Error processing PDF: {str(e)}"
