"""Ingestion & preprocessing for long-form documents (incident post-mortems,
research papers, legal briefs) before they hit the seq2seq encoder.

Two concerns kept separate on purpose:
  * `approx_token_count` / `chunk_long_document` use a cheap word-based
    heuristic so length-filtering and chunking logic can be unit-tested
    without loading a real subword tokenizer.
  * `SubwordTokenizer` wraps the actual Hugging Face tokenizer (WordPiece/BPE,
    whichever the chosen T5/BART checkpoint uses) for the real encode/decode
    path used at training and inference time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORDS_PER_TOKEN = 0.75  # rough heuristic: ~0.75 words per subword token in English


def approx_token_count(text: str) -> int:
    return max(1, int(len(text.split()) / _WORDS_PER_TOKEN))


@dataclass
class LengthFilterResult:
    keep: bool
    reason: str
    approx_tokens: int


def filter_by_length(text: str, min_tokens: int = 50, max_tokens: int = 100_000) -> LengthFilterResult:
    """Drops documents too short to summarize meaningfully or so long that
    they should be routed to chunking rather than a single encoder pass."""
    tokens = approx_token_count(text)
    if tokens < min_tokens:
        return LengthFilterResult(keep=False, reason=f"Below min_tokens ({tokens} < {min_tokens})", approx_tokens=tokens)
    if tokens > max_tokens:
        return LengthFilterResult(keep=False, reason=f"Above max_tokens ({tokens} > {max_tokens})", approx_tokens=tokens)
    return LengthFilterResult(keep=True, reason="within bounds", approx_tokens=tokens)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter — no NLTK/spaCy dependency required for
    the common case of well-punctuated technical/legal prose."""
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_long_document(text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """Splits a document that exceeds the encoder's context window into
    sentence-aligned chunks (never splitting a sentence across chunks), each
    under `max_tokens`, with an optional trailing-sentence overlap for
    context continuity between chunks.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = approx_token_count(sentence)
        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current))
            overlap = _tail_sentences_by_tokens(current, overlap_tokens)
            current = list(overlap)
            current_tokens = sum(approx_token_count(s) for s in current)
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def _tail_sentences_by_tokens(sentences: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []
    tail: list[str] = []
    tokens = 0
    for sentence in reversed(sentences):
        tokens += approx_token_count(sentence)
        tail.insert(0, sentence)
        if tokens >= overlap_tokens:
            break
    return tail


class SubwordTokenizer:
    """Thin wrapper around a Hugging Face `AutoTokenizer` for the real
    encode/truncate path used at train/inference time. Lazily loaded so
    importing this module never requires a model download.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer = None

    def _get(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                raise RuntimeError(
                    "transformers is required for real tokenization. `pip install transformers`."
                ) from e
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def encode(self, text: str, max_length: int, truncation: bool = True):
        tokenizer = self._get()
        return tokenizer(text, max_length=max_length, truncation=truncation, return_tensors="pt")

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        tokenizer = self._get()
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
