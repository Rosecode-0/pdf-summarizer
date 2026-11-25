import re


def _split_sentences(text: str) -> list[str]:
    """
    Normalizes single line-wraps within paragraphs and splits text into sentences
    without breaking abbreviations like 'et al.' or 'Fig. 1'.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        # Join single line wraps inside a paragraph
        normalized = re.sub(r"\s*\n\s*", " ", para).strip()
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'“])", normalized)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def chunk_text_for_model(text: str, tokenizer, max_tokens: int = 1024) -> list[str]:
    """
    Splits text into chunks such that each chunk has at most `max_tokens` tokens
    according to the provided Hugging Face tokenizer.
    Uses a single batched tokenizer call for speed.
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    effective_max = max(32, max_tokens - 16)

    # Batch-tokenize all sentences at once (dramatically faster than per-sentence encode)
    tokenized_sentences = tokenizer(
        sentences,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for sentence, token_ids in zip(sentences, tokenized_sentences):
        sentence_tokens = len(token_ids)
        if sentence_tokens == 0:
            continue

        # If a single sentence exceeds effective_max, split its tokens cleanly without dropping content
        if sentence_tokens >= effective_max:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            for start in range(0, sentence_tokens, effective_max):
                sub_ids = token_ids[start : start + effective_max]
                decoded = tokenizer.decode(sub_ids, skip_special_tokens=True).strip()
                if decoded:
                    chunks.append(decoded)
            continue

        if current_tokens + sentence_tokens > effective_max:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
