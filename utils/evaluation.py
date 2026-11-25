from rouge_score import rouge_scorer

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def calculate_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """
    Calculates ROUGE-1, ROUGE-2, and ROUGE-L F1 scores locally using `rouge-score`
    (zero network calls, instant execution).
    """
    if not predictions or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    pairs = list(zip(predictions, references))
    if not pairs:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in pairs:
        scores = _SCORER.score(ref or "", pred or "")
        for key in totals:
            totals[key] += float(scores[key].fmeasure)

    count = len(pairs)
    return {key: totals[key] / count for key in totals}
