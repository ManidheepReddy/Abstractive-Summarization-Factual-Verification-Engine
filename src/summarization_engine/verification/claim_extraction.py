"""Breaks a generated summary into individual claim sentences to be verified
independently against the source document — verifying the summary as one
blob would let a single false clause hide behind several true ones.
"""
from __future__ import annotations

from summarization_engine.ingestion.preprocessor import split_sentences


def extract_claims(summary_text: str, min_words: int = 3) -> list[str]:
    """Splits a summary into sentence-level claims, dropping fragments too
    short to be independently verifiable (e.g. a stray heading or bullet marker)."""
    sentences = split_sentences(summary_text)
    return [s for s in sentences if len(s.split()) >= min_words]
