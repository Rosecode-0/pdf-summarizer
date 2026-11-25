import py_compile
import sys
import traceback

try:
    import api
    from utils.pdf_extractor import extract_text_from_pdf
    from utils.text_cleaner import clean_text
    from utils.section_detector import detect_sections, remove_references_section
    from utils.text_chunker import chunk_text_for_model
    from utils.textrank import generate_textrank_summary
    from utils.pegasus import load_pegasus_model, generate_pegasus_summary
    from utils.evaluation import calculate_rouge
    from utils.report_builder import build_analysis_report, count_sentences

    py_compile.compile("app.py", doraise=True)
    print("All imports and syntax checks successful!")
except Exception:
    print("Error during import:")
    traceback.print_exc()
    sys.exit(1)
