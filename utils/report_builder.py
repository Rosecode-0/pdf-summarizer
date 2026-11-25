from datetime import datetime
import re


def count_sentences(text: str) -> int:
    """Counts sentences accurately without overcounting trailing periods or decimals."""
    if not text or not text.strip():
        return 0
    matches = re.findall(r"[^.!?]+(?:[.!?]+|$)", text.strip())
    return len([m for m in matches if m.strip()])


def build_analysis_report(
    paper_name: str,
    model_used: str,
    total_pages: int,
    word_count: int,
    detected_sections: list[str],
    textrank_summary: str,
    textrank_time: float,
    abstractive_summary: str,
    abstractive_time: float,
    section_summaries: dict[str, str] | None = None,
    rouge_tr: dict | None = None,
    rouge_pg: dict | None = None,
) -> str:
    """Builds the standardized plain-text analysis report for download."""
    rouge_tr = rouge_tr or {}
    rouge_pg = rouge_pg or {}

    def _fmt(val) -> str:
        return f"{val:.4f}" if isinstance(val, (int, float)) else "N/A"

    sections_str = ", ".join(detected_sections) if detected_sections else "None"
    report = f"""Scientific Research Paper Summarizer - Analysis Report
===================================================

Paper Name: {paper_name}
Model Used: {model_used}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Document Statistics:
- Pages: {total_pages}
- Words: {word_count}
- Detected Sections: {sections_str}

---------------------------------------------------
TEXTRANK BASELINE SUMMARY
---------------------------------------------------
{textrank_summary or "N/A"}

Processing Time: {textrank_time:.2f}s

---------------------------------------------------
ABSTRACTIVE SUMMARY ({model_used})
---------------------------------------------------
{abstractive_summary or "N/A"}

Processing Time: {abstractive_time:.2f}s

---------------------------------------------------
SECTION-WISE SUMMARIES
---------------------------------------------------
"""
    if section_summaries:
        for sec, sum_text in section_summaries.items():
            report += f"\n[{sec}]\n{sum_text}\n"
    else:
        report += "Not enabled or no sections detected.\n"

    report += f"""
---------------------------------------------------
EVALUATION SCORES
---------------------------------------------------
TextRank: ROUGE-1: {_fmt(rouge_tr.get("rouge1"))}, ROUGE-2: {_fmt(rouge_tr.get("rouge2"))}, ROUGE-L: {_fmt(rouge_tr.get("rougeL"))}
Abstractive: ROUGE-1: {_fmt(rouge_pg.get("rouge1"))}, ROUGE-2: {_fmt(rouge_pg.get("rouge2"))}, ROUGE-L: {_fmt(rouge_pg.get("rougeL"))}
"""
    return report
