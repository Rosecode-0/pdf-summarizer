import base64
import hashlib
import html
from pathlib import Path
import time

import pandas as pd
import streamlit as st

from utils.evaluation import calculate_rouge
from utils.pdf_extractor import extract_text_from_pdf
from utils.pegasus import generate_pegasus_summary, load_pegasus_model
from utils.report_builder import build_analysis_report, count_sentences
from utils.section_detector import detect_sections, remove_references_section
from utils.text_cleaner import clean_text
from utils.textrank import generate_textrank_summary

# Configure page settings
st.set_page_config(
    page_title="Scientific Research Paper Summarizer",
    page_icon="📄",
    layout="wide",
)

MODEL_OPTIONS = {
    "TextRank Only (Instant, 0 MB download)": "textrank",
    "Falconsai/text_summarization (Fast AI, ~240 MB)": "Falconsai/text_summarization",
    "google/pegasus-arxiv (Scientific, ~2.3 GB)": "google/pegasus-arxiv",
    "google/pegasus-pubmed (Biomedical, ~2.3 GB)": "google/pegasus-pubmed",
}

SPEED_PROFILES = {
    "Fast (Recommended: Greedy, max 8 chunks)": {"num_beams": 1, "max_chunks": 8},
    "Balanced (2 beams, max 12 chunks)": {"num_beams": 2, "max_chunks": 12},
    "Full Quality (4 beams, all chunks)": {"num_beams": 4, "max_chunks": None},
}


def _is_model_cached_locally(model_id: str) -> bool:
    if model_id == "textrank":
        return True
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
    return cache_dir.exists()


@st.cache_resource
def get_cached_model(model_id: str):
    return load_pegasus_model(model_id)


@st.cache_data(show_spinner=False)
def process_pdf_cached(file_hash: str, pdf_bytes: bytes, max_pages: int | None):
    raw_text, total_pages, ocr_used, error_msg = extract_text_from_pdf(
        pdf_bytes, max_pages=max_pages
    )
    if error_msg or not raw_text.strip():
        return raw_text, "", {}, total_pages, ocr_used, error_msg
    cleaned = clean_text(raw_text)
    sections = detect_sections(cleaned)
    return raw_text, cleaned, sections, total_pages, ocr_used, None


def _reset_summary_state():
    for key in [
        "summary_generated",
        "textrank_summary",
        "textrank_time",
        "pegasus_summary",
        "pegasus_time",
        "section_summaries",
        "device",
        "last_model_label",
        "rouge_scores_tr",
        "rouge_scores_pg",
    ]:
        st.session_state.pop(key, None)


def _direct_download_link(content: str, filename: str, label: str, mime: str = "text/plain") -> str:
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    safe_filename = html.escape(filename, quote=True)
    safe_label = html.escape(label)
    return (
        f'<a href="data:{mime};charset=utf-8;base64,{b64}" '
        f'download="{safe_filename}" '
        f'style="display:inline-block;padding:0.45rem 0.9rem;border-radius:0.4rem;'
        f'background:#0f62fe;color:#ffffff;text-decoration:none;font-weight:600;">'
        f"⬇️ {safe_label}</a>"
    )


# Default to a cached neural model if present locally, otherwise Fast AI (~240 MB)
default_index = 2 if _is_model_cached_locally("google/pegasus-arxiv") else 1

# Sidebar
st.sidebar.title("📄 Configuration")
st.sidebar.markdown("Scientific Research Paper Summarizer")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🤖 Model Selection")
model_label = st.sidebar.selectbox(
    "Choose Summarization Model:",
    list(MODEL_OPTIONS.keys()),
    index=default_index,
    help="Select TextRank for instant 0-MB summarization, Falconsai for fast low-data AI summarization, or PEGASUS for heavy scientific summarization.",
)
model_choice = MODEL_OPTIONS[model_label]

if model_choice == "textrank":
    st.sidebar.caption("⚡ Instant extractive mode (0 MB download).")
elif _is_model_cached_locally(model_choice):
    st.sidebar.caption("✅ Model cached locally (0 MB network download).")
else:
    st.sidebar.caption("⬇️ First run downloads weights once and caches them locally.")

st.sidebar.markdown("### ⚡ Performance & Data")
speed_label = st.sidebar.selectbox(
    "Speed / Compute Profile:",
    list(SPEED_PROFILES.keys()),
    index=0,
    help="Fast mode uses greedy decoding and limits chunks so long papers finish quickly.",
)
speed_cfg = SPEED_PROFILES[speed_label]
exclude_references = st.sidebar.checkbox(
    "Exclude References / Bibliography",
    value=True,
    help="Skips trailing References/Bibliography sections, saving ~25–40% runtime.",
)
max_pages_limit = st.sidebar.number_input(
    "Max pages to extract (0 = all):",
    min_value=0,
    max_value=500,
    value=0,
    step=5,
    help="Limit pages extracted for very large PDFs/books.",
)

st.sidebar.markdown("### ⚙️ TextRank Baseline")
num_sentences = st.sidebar.slider("Number of sentences:", min_value=1, max_value=30, value=10)

st.sidebar.markdown("### ⚙️ AI Summary Length")
pegasus_max_length = st.sidebar.slider("Max length per chunk:", min_value=80, max_value=500, value=220)
pegasus_min_length = st.sidebar.slider("Min length per chunk:", min_value=20, max_value=200, value=40)

st.sidebar.markdown("### 📋 Features")
enable_section_wise = st.sidebar.checkbox(
    "Enable Section-wise Summary (slower)",
    value=False,
    help="Runs an extra AI summarization pass per detected section.",
)
enable_evaluation = st.sidebar.checkbox("Enable Reference-based Evaluation", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("Run Dataset Evaluation Mode"):
    st.sidebar.info(
        "Dataset evaluation mode is planned for future phases. Currently focusing on single PDF analysis."
    )

# Main title
st.title("📄 Scientific Research Paper Summarizer")
st.subheader("Fast Extractive and Abstractive Summarization (TextRank, Fast T5, PEGASUS)")

# File uploader
st.markdown("### STEP 1 — Upload Research Paper")
uploaded_file = st.file_uploader("Choose a PDF file (up to 50 MB)", type="pdf")

if uploaded_file is None:
    st.session_state.pop("active_file_hash", None)
    _reset_summary_state()
else:
    pdf_bytes = uploaded_file.getvalue()
    file_size_mb = len(pdf_bytes) / (1024 * 1024)
    effective_max_pages = int(max_pages_limit) if max_pages_limit > 0 else None
    file_hash = hashlib.sha256(pdf_bytes).hexdigest() + f":{effective_max_pages}"

    if st.session_state.get("active_file_hash") != file_hash:
        st.session_state["active_file_hash"] = file_hash
        _reset_summary_state()

    with st.spinner("Extracting & caching PDF text..."):
        raw_text, cleaned_text, sections, total_pages, ocr_used, error_msg = process_pdf_cached(
            file_hash, pdf_bytes, effective_max_pages
        )

    if error_msg:
        st.error(error_msg)
    elif not cleaned_text.strip():
        st.error("No text could be extracted from the uploaded PDF.")
    else:
        st.success(f"PDF ready ({file_size_mb:.2f} MB) — cached for instant interactions!")
        if ocr_used:
            st.info("OCR was used for scanned pages.")

        summarization_text = (
            remove_references_section(cleaned_text) if exclude_references else cleaned_text
        )

        # STEP 2
        st.markdown("### STEP 2 — Document Information")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("File name", uploaded_file.name)
        col2.metric("Total Pages", total_pages)
        col3.metric("Word Count", len(cleaned_text.split()))
        col4.metric("Summarizable Words", len(summarization_text.split()))

        st.markdown("**Detected Sections:**")
        if sections:
            st.write(", ".join(sections.keys()))
        else:
            st.write("No standard sections detected. Treating as full text.")

        # STEP 3
        with st.expander("STEP 3 — Extracted Content Preview"):
            preview = cleaned_text[:2000] + ("..." if len(cleaned_text) > 2000 else "")
            st.text_area("Cleaned Text (First 2000 chars)", preview, height=200)

        # Generate Summaries
        if st.button("Generate Summaries", type="primary"):
            _reset_summary_state()
            st.session_state["last_model_label"] = model_choice

            # TextRank
            try:
                with st.spinner("Generating TextRank extractive summary..."):
                    start_time = time.time()
                    textrank_summary = generate_textrank_summary(
                        summarization_text, num_sentences=num_sentences
                    )
                    st.session_state["textrank_summary"] = textrank_summary
                    st.session_state["textrank_time"] = time.time() - start_time
            except Exception as e:
                st.error(f"TextRank error: {e}")
                st.session_state["textrank_summary"] = "TextRank summarization failed."
                st.session_state["textrank_time"] = 0.0

            # Abstractive Model (or instant skip when TextRank-only selected)
            if model_choice == "textrank":
                st.session_state["device"] = "CPU"
                st.session_state["pegasus_summary"] = (
                    "Skipped (TextRank-Only fast mode selected). Choose an AI model in the sidebar for abstractive summarization."
                )
                st.session_state["pegasus_time"] = 0.0
                st.session_state["section_summaries"] = {}
            else:
                model = None
                tokenizer = None
                try:
                    with st.spinner(f"Loading model ({model_choice})..."):
                        model, tokenizer = get_cached_model(model_choice)
                        device = "CUDA" if next(model.parameters()).device.type == "cuda" else "CPU"
                        st.session_state["device"] = device

                    with st.spinner("Generating abstractive summary..."):
                        pbar = st.progress(0, text="Processing chunks...")
                        start_time = time.time()
                        pegasus_summary = generate_pegasus_summary(
                            summarization_text,
                            model,
                            tokenizer,
                            max_length=pegasus_max_length,
                            min_length=pegasus_min_length,
                            num_beams=speed_cfg["num_beams"],
                            max_chunks=speed_cfg["max_chunks"],
                            progress_callback=lambda pct: pbar.progress(pct, text=f"Processing chunks... {pct}%"),
                        )
                        pbar.empty()
                        st.session_state["pegasus_summary"] = pegasus_summary
                        st.session_state["pegasus_time"] = time.time() - start_time
                except Exception as e:
                    st.error(f"Abstractive summarization error: {e}")
                    st.session_state["pegasus_summary"] = "Abstractive summarization failed."
                    st.session_state["pegasus_time"] = 0.0

                # Section-wise
                st.session_state["section_summaries"] = {}
                if (
                    enable_section_wise
                    and sections
                    and model is not None
                    and st.session_state.get("pegasus_summary") != "Abstractive summarization failed."
                ):
                    try:
                        with st.spinner("Generating section-wise summaries..."):
                            for sec_name, sec_text in sections.items():
                                if sec_name in ["Abstract", "Introduction", "Methodology", "Results", "Conclusion"]:
                                    sec_sum = generate_pegasus_summary(
                                        sec_text,
                                        model,
                                        tokenizer,
                                        max_length=150,
                                        min_length=30,
                                        num_beams=speed_cfg["num_beams"],
                                        max_chunks=3,
                                    )
                                    st.session_state["section_summaries"][sec_name] = sec_sum
                    except Exception as e:
                        st.warning(f"Section-wise summarization failed: {e}")

            st.session_state["summary_generated"] = True

        if st.session_state.get("summary_generated", False):
            tr_summary = st.session_state.get("textrank_summary", "")
            pg_summary = st.session_state.get("pegasus_summary", "")
            tr_time = st.session_state.get("textrank_time", 0.0)
            pg_time = st.session_state.get("pegasus_time", 0.0)
            used_model = st.session_state.get("last_model_label", model_choice)

            # STEP 4
            st.markdown("### STEP 4 — TextRank Baseline")
            if tr_summary:
                st.info(tr_summary)

            # STEP 5
            st.markdown("### STEP 5 — Abstractive Summary")
            st.caption(f"Model: {used_model} | Device: {st.session_state.get('device', 'CPU')}")
            if pg_summary:
                st.success(pg_summary)

            # STEP 6
            if enable_section_wise and st.session_state.get("section_summaries"):
                st.markdown("### STEP 6 — Section-wise Summary")
                for sec_name, sec_sum in st.session_state["section_summaries"].items():
                    st.markdown(f"#### {sec_name} Summary")
                    st.write(sec_sum)

            # STEP 7
            rouge_scores_tr = st.session_state.get("rouge_scores_tr") or {
                "rouge1": "N/A",
                "rouge2": "N/A",
                "rougeL": "N/A",
            }
            rouge_scores_pg = st.session_state.get("rouge_scores_pg") or {
                "rouge1": "N/A",
                "rouge2": "N/A",
                "rougeL": "N/A",
            }

            if enable_evaluation:
                st.markdown("### STEP 7 — Evaluation")
                st.markdown("**Reference-based evaluation**")
                reference_summary = st.text_area(
                    "Paste Reference Summary (e.g., ground-truth Abstract) here to calculate ROUGE:"
                )
                if st.button("Calculate ROUGE"):
                    if reference_summary.strip():
                        rouge_scores_tr = calculate_rouge([tr_summary], [reference_summary])
                        st.session_state["rouge_scores_tr"] = rouge_scores_tr

                        if used_model != "textrank" and pg_summary != "Abstractive summarization failed.":
                            rouge_scores_pg = calculate_rouge([pg_summary], [reference_summary])
                        else:
                            rouge_scores_pg = {"rouge1": "N/A", "rouge2": "N/A", "rougeL": "N/A"}
                        st.session_state["rouge_scores_pg"] = rouge_scores_pg
                        st.success("ROUGE scores calculated locally (0 network calls)!")
                    else:
                        st.warning("Please provide a reference summary.")

            # STEP 8
            st.markdown("### STEP 8 — Model Comparison")

            tr_len = len(tr_summary.split())
            pg_len = len(pg_summary.split())
            tr_sents = count_sentences(tr_summary)
            pg_sents = count_sentences(pg_summary)

            def _fmt_rouge(val) -> str:
                return f"{val:.4f}" if isinstance(val, (int, float)) else "N/A"

            tr_rouge1 = _fmt_rouge(rouge_scores_tr.get("rouge1"))
            tr_rouge2 = _fmt_rouge(rouge_scores_tr.get("rouge2"))
            tr_rougeL = _fmt_rouge(rouge_scores_tr.get("rougeL"))
            pg_rouge1 = _fmt_rouge(rouge_scores_pg.get("rouge1"))
            pg_rouge2 = _fmt_rouge(rouge_scores_pg.get("rouge2"))
            pg_rougeL = _fmt_rouge(rouge_scores_pg.get("rougeL"))

            comparison_data = {
                "Model": ["TextRank", f"Abstractive ({used_model})"],
                "Summary Length (words)": [tr_len, pg_len],
                "Number of sentences": [tr_sents, pg_sents],
                "ROUGE-1": [tr_rouge1, pg_rouge1],
                "ROUGE-2": [tr_rouge2, pg_rouge2],
                "ROUGE-L": [tr_rougeL, pg_rougeL],
                "Processing Time (s)": [f"{tr_time:.2f}", f"{pg_time:.2f}"],
            }
            st.table(pd.DataFrame(comparison_data))

            # Build full report and provide direct download links + buttons
            report_content = build_analysis_report(
                paper_name=uploaded_file.name,
                model_used=used_model,
                total_pages=total_pages,
                word_count=len(cleaned_text.split()),
                detected_sections=list(sections.keys()),
                textrank_summary=tr_summary,
                textrank_time=tr_time,
                abstractive_summary=pg_summary,
                abstractive_time=pg_time,
                section_summaries=st.session_state.get("section_summaries") if enable_section_wise else None,
                rouge_tr=rouge_scores_tr,
                rouge_pg=rouge_scores_pg,
            )

            base_stem = Path(uploaded_file.name).stem
            report_filename = f"{base_stem}_summary_report.txt"

            st.markdown("### Direct Download Link & Report Downloads")
            st.markdown(
                _direct_download_link(report_content, report_filename, f"Direct Download: {report_filename}"),
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "Download TextRank Summary",
                    tr_summary,
                    file_name=f"{base_stem}_textrank_summary.txt",
                )
            with col2:
                st.download_button(
                    "Download Abstractive Summary",
                    pg_summary,
                    file_name=f"{base_stem}_abstractive_summary.txt",
                )
            with col3:
                st.download_button(
                    "Download Full Analysis Report",
                    report_content,
                    file_name=report_filename,
                )
