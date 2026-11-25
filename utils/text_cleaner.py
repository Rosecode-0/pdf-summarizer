import re

_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u00a0": " ",
}


def clean_text(text: str) -> str:
    """
    Cleans extracted PDF text by normalizing ligatures, whitespace, hyphenated
    line breaks, and standalone page numbers while preserving paragraph structure.
    """
    if not text:
        return ""

    for lig, replacement in _LIGATURES.items():
        text = text.replace(lig, replacement)

    # Remove control chars except newline and tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize CRLF / CR to LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Join hyphenated words split across line breaks (e.g., "algo-\nrithm" -> "algorithm")
    text = re.sub(r"(?<=[A-Za-z])-\n\s*(?=[A-Za-z])", "", text)

    # Remove standalone page numbers on their own line
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

    # Collapse excessive newlines and horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()
