"""Lead-N extractive baseline — the standard cheap comparison point for
abstractive summarization: how much is the fine-tuned model actually buying
over "just take the first N sentences"?
"""
from __future__ import annotations

from summarization_engine.ingestion.preprocessor import split_sentences


def lead_n_summary(document: str, n: int = 3) -> str:
    sentences = split_sentences(document)
    return " ".join(sentences[:n])
