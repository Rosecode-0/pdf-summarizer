import re
from collections import Counter
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

_NLTK_READY = False


def ensure_nltk_data() -> bool:
    """Checks for required NLTK data locally first to avoid redundant network calls."""
    global _NLTK_READY
    if _NLTK_READY:
        return True

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                return False

    _NLTK_READY = True
    return True


def _fallback_extractive_summary(text: str, num_sentences: int) -> str:
    """Pure-Python frequency-based extractive fallback if NLTK/sumy cannot run offline."""
    normalized = re.sub(r"\s+", " ", text).strip()
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'“])", normalized)
        if len(s.strip()) > 20
    ]
    if not sentences:
        return normalized[:1000]
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    words = re.findall(r"\b[a-zA-Z]{3,}\b", normalized.lower())
    freq = Counter(words)
    scored = []
    for idx, sent in enumerate(sentences):
        sent_words = re.findall(r"\b[a-zA-Z]{3,}\b", sent.lower())
        score = sum(freq[w] for w in sent_words) / max(1, len(sent_words))
        scored.append((score, idx, sent))

    top = sorted(scored, key=lambda x: x[0], reverse=True)[:num_sentences]
    top_in_order = sorted(top, key=lambda x: x[1])
    return " ".join(item[2] for item in top_in_order)


def generate_textrank_summary(text: str, num_sentences: int = 10) -> str:
    """
    Generates an extractive summary using the TextRank algorithm via sumy,
    with an offline pure-Python fallback if NLTK data cannot be downloaded.
    """
    if not text or not text.strip():
        return "No text provided for summarization."

    if ensure_nltk_data():
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            summary = summarizer(parser.document, num_sentences)
            result = " ".join(str(sentence) for sentence in summary).strip()
            if result:
                return result
        except Exception:
            pass

    return _fallback_extractive_summary(text, num_sentences)
