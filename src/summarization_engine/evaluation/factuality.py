"""Factual-consistency scoring — the headline metric this project's success
criteria (near-zero hallucination) is measured against.
"""
from __future__ import annotations

from summarization_engine.models import ClaimVerification, NLILabel


def factuality_score(claims: list[ClaimVerification]) -> float:
    """Percentage of claims classified as ENTAILMENT."""
    if not claims:
        return 0.0
    entailed = sum(1 for c in claims if c.label == NLILabel.ENTAILMENT)
    return entailed / len(claims)


def factuality_breakdown(claims: list[ClaimVerification]) -> dict[str, float]:
    """Full label distribution — entailment/neutral/contradiction proportions —
    since a low factuality score alone doesn't distinguish "mostly unverifiable"
    from "actively contradicts the source", and those call for different fixes.
    """
    if not claims:
        return {label.value: 0.0 for label in NLILabel}
    total = len(claims)
    counts = {label: 0 for label in NLILabel}
    for c in claims:
        counts[c.label] += 1
    return {label.value: count / total for label, count in counts.items()}
