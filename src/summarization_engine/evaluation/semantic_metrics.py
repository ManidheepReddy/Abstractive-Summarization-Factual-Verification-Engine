"""Semantic similarity via BERTScore — measures token similarity using
contextual embeddings rather than surface n-gram overlap, catching
paraphrases that ROUGE/BLEU penalize unfairly.

Wrapped behind a thin optional interface (like `ragas_faithfulness` in the
sibling financial-RAG project's eval suite): returns `None` if `bert-score`
isn't installed, so the rest of the evaluation suite still runs without it.
"""
from __future__ import annotations


def bertscore(reference: str, candidate: str, lang: str = "en") -> dict | None:
    try:
        from bert_score import score
    except ImportError:
        return None

    precision, recall, f1 = score([candidate], [reference], lang=lang, verbose=False)
    return {
        "bertscore_precision": float(precision[0]),
        "bertscore_recall": float(recall[0]),
        "bertscore_f1": float(f1[0]),
    }
