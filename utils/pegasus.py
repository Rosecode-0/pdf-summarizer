from functools import lru_cache
import importlib.util
from typing import Callable

from utils.text_chunker import chunk_text_for_model


@lru_cache(maxsize=2)
def load_pegasus_model(model_name: str = "google/pegasus-arxiv"):
    """
    Loads a Seq2Seq summarization model and tokenizer.
    Checks local Hugging Face cache first (`local_files_only=True`) to eliminate
    unnecessary network latency/data, falling back to downloading if not cached.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_kwargs = {}
    if importlib.util.find_spec("accelerate") is not None:
        load_kwargs["low_cpu_mem_usage"] = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, local_files_only=True, **load_kwargs
        ).to(device)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, **load_kwargs
        ).to(device)

    model.eval()
    return model, tokenizer


def _is_t5_model(model) -> bool:
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    return str(model_type).lower().startswith("t5")


def _select_even_chunks(chunks: list[str], max_chunks: int) -> list[str]:
    """Selects up to `max_chunks` evenly spaced chunks across the document."""
    if max_chunks <= 0 or len(chunks) <= max_chunks:
        return chunks
    if max_chunks == 1:
        return [chunks[0]]
    indices = [round(i * (len(chunks) - 1) / (max_chunks - 1)) for i in range(max_chunks)]
    # Deduplicate while preserving order
    seen = set()
    selected = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            selected.append(chunks[idx])
    return selected


def generate_pegasus_summary(
    text: str,
    model,
    tokenizer,
    max_length: int = 250,
    min_length: int = 50,
    num_beams: int = 1,
    max_chunks: int | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> str:
    """
    Generates an abstractive summary using a loaded Seq2Seq model (PEGASUS, T5, BART).
    Uses `torch.inference_mode()` and configurable `num_beams` (default 1 for speed).
    """
    if not text or not text.strip():
        return "No text provided for summarization."

    raw_max = getattr(tokenizer, "model_max_length", 1024)
    if not isinstance(raw_max, int) or raw_max <= 0 or raw_max > 4096:
        raw_max = 1024
    model_max_tokens = min(raw_max, 1024)

    # Leave space for optional T5 task prefix
    is_t5 = _is_t5_model(model)
    chunk_token_budget = model_max_tokens - (8 if is_t5 else 0)

    chunks = chunk_text_for_model(text, tokenizer, max_tokens=chunk_token_budget)
    if not chunks:
        return "Could not create chunks for summarization."

    if max_chunks and max_chunks > 0 and len(chunks) > max_chunks:
        chunks = _select_even_chunks(chunks, max_chunks)

    import torch

    chunk_summaries: list[str] = []
    total_chunks = len(chunks)

    with torch.inference_mode():
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                if progress_callback:
                    progress_callback(int((i + 1) / total_chunks * 100))
                continue

            try:
                chunk_input = (
                    f"summarize: {chunk}"
                    if is_t5 and not chunk.lstrip().lower().startswith("summarize:")
                    else chunk
                )
                inputs = tokenizer(
                    chunk_input,
                    return_tensors="pt",
                    max_length=model_max_tokens,
                    truncation=True,
                ).to(model.device)

                chunk_tokens = int(inputs["input_ids"].shape[1])
                if chunk_tokens < 12:
                    if progress_callback:
                        progress_callback(int((i + 1) / total_chunks * 100))
                    continue

                current_max_len = min(max_length, max(12, int(chunk_tokens * 0.85)))
                current_min_len = min(min_length, int(current_max_len * 0.5))
                if current_min_len >= current_max_len:
                    current_min_len = max(0, current_max_len - 1)

                summary_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=current_max_len,
                    min_length=current_min_len,
                    num_beams=max(1, int(num_beams)),
                    do_sample=False,
                    early_stopping=num_beams > 1,
                )
                summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
                if summary_text:
                    chunk_summaries.append(summary_text)

                if progress_callback:
                    progress_callback(int((i + 1) / total_chunks * 100))
            except Exception as e:
                print(f"Error summarizing chunk {i}: {e}")
                continue

    if not chunk_summaries:
        return "Failed to generate abstractive summary."

    return " ".join(chunk_summaries)
