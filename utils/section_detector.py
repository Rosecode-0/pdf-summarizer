import re

_SECTION_SPECS = [
    ("Abstract", [r"abstract"]),
    ("Introduction", [r"introduction"]),
    ("Related Work", [r"related work", r"literature review", r"background"]),
    (
        "Methodology",
        [
            r"methodology",
            r"methods",
            r"materials and methods",
            r"proposed method",
            r"approach",
        ],
    ),
    ("Experiments", [r"experiments", r"experimental setup", r"evaluation"]),
    ("Results", [r"results and discussion", r"results", r"findings"]),
    ("Discussion", [r"discussion"]),
    ("Conclusion", [r"conclusions?", r"conclusion and future work"]),
    ("Acknowledgments", [r"acknowledg(?:e)?ments?"]),
    ("References", [r"references", r"bibliography"]),
]

_PREFIX = r"(?:(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+)?"


def _compile_section_patterns() -> dict[str, re.Pattern]:
    patterns = {}
    for name, titles in _SECTION_SPECS:
        title_group = "|".join(titles)
        if name == "Abstract":
            # Matches standalone "Abstract" or IEEE inline "Abstract—..." / "Abstract: ..."
            regex = rf"^{_PREFIX}(?:{title_group})(?:\s*[:—\-]\s*|\s*$)"
        else:
            # Must be a standalone heading line (optional trailing colon/spaces)
            regex = rf"^{_PREFIX}(?:{title_group})\s*:?\s*$"
        patterns[name] = re.compile(regex, flags=re.IGNORECASE | re.MULTILINE)
    return patterns


_COMPILED_PATTERNS = _compile_section_patterns()


def detect_sections(text: str) -> dict[str, str]:
    """
    Heuristically detect sections in a scientific paper based on standalone headings.
    Returns a dictionary mapping section names to their corresponding text content.
    """
    if not text:
        return {}

    found_headings = []
    for section_name, pattern in _COMPILED_PATTERNS.items():
        for match in pattern.finditer(text):
            found_headings.append(
                {
                    "name": section_name,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    if not found_headings:
        return {}

    found_headings.sort(key=lambda x: x["start"])

    # Deduplicate overlapping matches at the same start offset
    deduped = []
    for h in found_headings:
        if deduped and h["start"] < deduped[-1]["end"]:
            continue
        deduped.append(h)

    sections: dict[str, str] = {}
    for i, heading in enumerate(deduped):
        start_idx = heading["end"]
        end_idx = deduped[i + 1]["start"] if i + 1 < len(deduped) else len(text)
        section_text = text[start_idx:end_idx].strip()

        if len(section_text) > 50:
            name = heading["name"]
            if name in sections:
                sections[name] += "\n\n" + section_text
            else:
                sections[name] = section_text

    return sections


def remove_references_section(text: str) -> str:
    """
    Removes trailing References / Bibliography (and trailing Acknowledgments)
    to avoid wasting compute on citation lists during summarization.
    """
    if not text:
        return ""

    ref_pattern = re.compile(
        rf"^{_PREFIX}(?:references|bibliography)\s*:?\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    matches = list(ref_pattern.finditer(text))
    if not matches:
        return text

    # Use the last match if it occurs after the first 25% of the document
    for match in reversed(matches):
        if match.start() >= len(text) * 0.25:
            return text[: match.start()].strip()

    return text
