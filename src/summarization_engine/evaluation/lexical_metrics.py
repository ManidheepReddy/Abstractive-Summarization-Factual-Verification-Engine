"""Lexical overlap metrics — ROUGE-1, ROUGE-2, ROUGE-L, and BLEU — implemented
in pure Python so the evaluation suite runs without the `rouge-score` /
`sacrebleu` packages and can be unit-tested with plain strings. Swap in the
official packages for publication-grade numbers if needed; these match the
standard F-measure/precision definitions closely enough for development-time
benchmarking against the Lead-3 baseline.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from summarization_engine.models import LexicalMetrics


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _rouge_n(reference: str, candidate: str, n: int) -> float:
    """ROUGE-N F1: harmonic mean of n-gram recall and precision against the reference."""
    ref_grams = _ngrams(_tokenize(reference), n)
    cand_grams = _ngrams(_tokenize(candidate), n)
    if not ref_grams or not cand_grams:
        return 0.0

    overlap = sum((ref_grams & cand_grams).values())
    recall = overlap / sum(ref_grams.values())
    precision = overlap / sum(cand_grams.values())
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def rouge_1(reference: str, candidate: str) -> float:
    return _rouge_n(reference, candidate, 1)


def rouge_2(reference: str, candidate: str) -> float:
    return _rouge_n(reference, candidate, 2)


def _lcs_length(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L F1, based on longest common subsequence rather than n-gram
    overlap — rewards sentence-level structural similarity."""
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0

    lcs = _lcs_length(ref_tokens, cand_tokens)
    if lcs == 0:
        return 0.0
    recall = lcs / len(ref_tokens)
    precision = lcs / len(cand_tokens)
    return 2 * recall * precision / (recall + precision)


def bleu(reference: str, candidate: str, max_n: int = 4) -> float:
    """Corpus-level-style BLEU for a single reference/candidate pair: geometric
    mean of modified n-gram precisions (n=1..max_n) times a brevity penalty.
    """
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not cand_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_grams = _ngrams(ref_tokens, n)
        cand_grams = _ngrams(cand_tokens, n)
        if not cand_grams:
            precisions.append(0.0)
            continue
        clipped = sum(min(count, ref_grams.get(gram, 0)) for gram, count in cand_grams.items())
        precisions.append(clipped / sum(cand_grams.values()))

    if any(p == 0 for p in precisions):
        return 0.0

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    brevity_penalty = 1.0 if len(cand_tokens) > len(ref_tokens) else math.exp(
        1 - len(ref_tokens) / len(cand_tokens)
    )
    return brevity_penalty * geo_mean


def compute_lexical_metrics(reference: str, candidate: str) -> LexicalMetrics:
    return LexicalMetrics(
        rouge1=rouge_1(reference, candidate),
        rouge2=rouge_2(reference, candidate),
        rougeL=rouge_l(reference, candidate),
        bleu=bleu(reference, candidate),
    )
