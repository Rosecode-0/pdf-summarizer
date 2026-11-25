from collections import OrderedDict
import hashlib
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response
import uvicorn

from utils.pdf_extractor import extract_text_from_pdf
from utils.pegasus import generate_pegasus_summary, load_pegasus_model
from utils.report_builder import build_analysis_report
from utils.section_detector import detect_sections, remove_references_section
from utils.text_cleaner import clean_text
from utils.textrank import generate_textrank_summary

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_PDF_CACHE = 20
MAX_REPORTS = 100

app = FastAPI(title="Scientific Research Paper Summarizer API")

_PDF_CACHE: OrderedDict[str, tuple[str, dict[str, str], int, bool]] = OrderedDict()
_REPORT_STORE: OrderedDict[str, tuple[str, str]] = OrderedDict()


def _cache_put(store: OrderedDict, key: str, value, limit: int):
    if key in store:
        store.move_to_end(key)
    store[key] = value
    while len(store) > limit:
        store.popitem(last=False)


async def _run_summarization(
    file: UploadFile,
    model_name: str,
    max_length: int,
    min_length: int,
    use_textrank: bool,
    textrank_sentences: int,
    exclude_references: bool,
    num_beams: int,
    max_chunks: int | None,
    max_pages: int | None,
):
    pdf_bytes = await file.read()
    if not pdf_bytes:
        return None, JSONResponse({"error": "Uploaded PDF file is empty."}, status_code=400)
    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        return None, JSONResponse(
            {"error": "PDF exceeds 50 MB upload limit."}, status_code=413
        )

    effective_max_pages = max_pages if max_pages and max_pages > 0 else None
    cache_key = f"{hashlib.sha256(pdf_bytes).hexdigest()}:{effective_max_pages}"

    if cache_key in _PDF_CACHE:
        _PDF_CACHE.move_to_end(cache_key)
        cleaned_text, sections, total_pages, ocr_used = _PDF_CACHE[cache_key]
    else:
        raw_text, total_pages, ocr_used, error_msg = extract_text_from_pdf(
            pdf_bytes, max_pages=effective_max_pages
        )
        if error_msg:
            return None, JSONResponse({"error": error_msg}, status_code=400)
        if not raw_text.strip():
            return None, JSONResponse(
                {"error": "No extractable text found in PDF."}, status_code=400
            )
        cleaned_text = clean_text(raw_text)
        sections = detect_sections(cleaned_text)
        _cache_put(_PDF_CACHE, cache_key, (cleaned_text, sections, total_pages, ocr_used), MAX_PDF_CACHE)

    summarization_text = (
        remove_references_section(cleaned_text) if exclude_references else cleaned_text
    )

    tr_summary = ""
    tr_time = 0.0
    if use_textrank or model_name == "textrank":
        t0 = time.time()
        tr_summary = generate_textrank_summary(summarization_text, num_sentences=textrank_sentences)
        tr_time = time.time() - t0

    pg_summary = ""
    pg_time = 0.0
    if model_name != "textrank":
        try:
            t0 = time.time()
            summarizer, tokenizer = load_pegasus_model(model_name)
            pg_summary = generate_pegasus_summary(
                summarization_text,
                summarizer,
                tokenizer,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                max_chunks=max_chunks if max_chunks and max_chunks > 0 else None,
            )
            pg_time = time.time() - t0
        except Exception as e:
            return None, JSONResponse(
                {"error": f"Model loading or summarization failed: {str(e)}"},
                status_code=500,
            )

    paper_name = file.filename or "research_paper.pdf"
    report_content = build_analysis_report(
        paper_name=paper_name,
        model_used=model_name,
        total_pages=total_pages,
        word_count=len(cleaned_text.split()),
        detected_sections=list(sections.keys()),
        textrank_summary=tr_summary or "Not requested.",
        textrank_time=tr_time,
        abstractive_summary=pg_summary or "Skipped (TextRank-only mode).",
        abstractive_time=pg_time,
    )

    report_id = uuid.uuid4().hex[:12]
    report_filename = f"{Path(paper_name).stem}_summary_report.txt"
    _cache_put(_REPORT_STORE, report_id, (report_filename, report_content), MAX_REPORTS)

    payload = {
        "report_id": report_id,
        "filename": report_filename,
        "textrank_summary": tr_summary if (use_textrank or model_name == "textrank") else None,
        "pegasus_summary": pg_summary if model_name != "textrank" else None,
        "stats": {
            "total_pages": total_pages,
            "word_count": len(cleaned_text.split()),
            "summarizable_words": len(summarization_text.split()),
            "ocr_used": ocr_used,
            "detected_sections": list(sections.keys()),
        },
        "timings": {
            "textrank_seconds": round(tr_time, 3),
            "abstractive_seconds": round(pg_time, 3),
        },
        "report_content": report_content,
    }
    return payload, None


@app.get("/")
def root():
    return {
        "service": "Scientific Research Paper Summarizer API",
        "endpoints": {
            "summarize": "POST /summarize",
            "summarize_and_download": "POST /summarize/download",
            "download_report": "GET /download/{report_id}",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize")
async def summarize(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form("Falconsai/text_summarization"),
    max_length: int = Form(220),
    min_length: int = Form(40),
    use_textrank: bool = Form(True),
    textrank_sentences: int = Form(10),
    exclude_references: bool = Form(True),
    num_beams: int = Form(1),
    max_chunks: int | None = Form(8),
    max_pages: int | None = Form(None),
):
    payload, err = await _run_summarization(
        file=file,
        model_name=model_name,
        max_length=max_length,
        min_length=min_length,
        use_textrank=use_textrank,
        textrank_sentences=textrank_sentences,
        exclude_references=exclude_references,
        num_beams=num_beams,
        max_chunks=max_chunks,
        max_pages=max_pages,
    )
    if err is not None:
        return err

    download_url = str(request.url_for("download_report", report_id=payload["report_id"]))
    result = {
        "report_id": payload["report_id"],
        "download_url": download_url,
        "stats": payload["stats"],
        "timings": payload["timings"],
    }
    if payload["textrank_summary"] is not None:
        result["textrank_summary"] = payload["textrank_summary"]
    if payload["pegasus_summary"] is not None:
        result["pegasus_summary"] = payload["pegasus_summary"]
    return result


@app.post("/summarize/download")
async def summarize_and_download(
    file: UploadFile = File(...),
    model_name: str = Form("Falconsai/text_summarization"),
    max_length: int = Form(220),
    min_length: int = Form(40),
    use_textrank: bool = Form(True),
    textrank_sentences: int = Form(10),
    exclude_references: bool = Form(True),
    num_beams: int = Form(1),
    max_chunks: int | None = Form(8),
    max_pages: int | None = Form(None),
):
    payload, err = await _run_summarization(
        file=file,
        model_name=model_name,
        max_length=max_length,
        min_length=min_length,
        use_textrank=use_textrank,
        textrank_sentences=textrank_sentences,
        exclude_references=exclude_references,
        num_beams=num_beams,
        max_chunks=max_chunks,
        max_pages=max_pages,
    )
    if err is not None:
        return err

    return Response(
        content=payload["report_content"],
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{payload["filename"]}"'
        },
    )


@app.get("/download/{report_id}", name="download_report")
def download_report(report_id: str):
    item = _REPORT_STORE.get(report_id)
    if item is None:
        return JSONResponse({"error": "Report not found or expired."}, status_code=404)
    filename, content = item
    return Response(
        content=content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
