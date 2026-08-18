"""Automated NLI hallucination detector.

For each claim sentence extracted from the summary (the *hypothesis*), finds
the most relevant sentence(s) in the source document (the *premise*) and
classifies the pair with a DistilBERT NLI model as entailment, neutral, or
contradiction. Contradictions are the hallucinations the project is meant to
catch; neutral claims are flagged as unverifiable rather than assumed true.

Premise selection matters: NLI models have a limited context window, so the
premise can't just be "the whole source document" — `select_evidence_sentences`
picks the most lexically-relevant source sentences per claim using a
dependency-free Jaccard-overlap heuristic. A production system would swap
this for embedding similarity (e.g. reusing a sentence-transformers encoder),
but the interface here is drop-in compatible with that swap.
"""
from __future__ import annotations

from summarization_engine.ingestion.preprocessor import split_sentences
from summarization_engine.models import ClaimVerification, NLILabel
from summarization_engine.verification.claim_extraction import extract_claims

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "for", "with", "as", "by", "at", "this", "that", "it",
    "its", "from", "will", "would", "can", "could", "has", "have", "had",
}


def _content_words(text: str) -> set[str]:
    return {w.strip(".,;:!?()[]\"'").lower() for w in text.split()} - _STOPWORDS


def select_evidence_sentences(claim: str, source_sentences: list[str], top_n: int = 2) -> list[str]:
    """Ranks source sentences by word-overlap with `claim` and returns the
    top `top_n` as the NLI premise, so each claim is checked against the part
    of the source most likely to support (or contradict) it."""
    claim_words = _content_words(claim)
    if not claim_words or not source_sentences:
        return source_sentences[:top_n]

    scored = []
    for sentence in source_sentences:
        overlap = claim_words & _content_words(sentence)
        union = claim_words | _content_words(sentence)
        jaccard = len(overlap) / len(union) if union else 0.0
        scored.append((jaccard, sentence))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [sentence for _, sentence in scored[:top_n]]


class NLIVerifier:
    def __init__(self, model_name: str = "typeform/distilbert-base-uncased-mnli"):
        self.model_name = model_name
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as e:
                raise RuntimeError(
                    "transformers is required for NLI verification. `pip install transformers torch`."
                ) from e
            self._pipeline = pipeline("text-classification", model=self.model_name, top_k=None)
        return self._pipeline

    def classify(self, premise: str, hypothesis: str) -> tuple[NLILabel, float]:
        """Returns the highest-scoring NLI label for (premise, hypothesis)."""
        clf = self._get_pipeline()
        # MNLI-style models expect "<premise> </s></s> <hypothesis>"-style pairing;
        # the pipeline's text_pair argument handles the correct formatting.
        scores = clf({"text": premise, "text_pair": hypothesis})
        best = max(scores, key=lambda s: s["score"])
        label = _normalize_label(best["label"])
        return label, float(best["score"])

    def verify_claims(self, summary_text: str, source_document: str, evidence_top_n: int = 2) -> list[ClaimVerification]:
        claims = extract_claims(summary_text)
        source_sentences = split_sentences(source_document)

        results: list[ClaimVerification] = []
        for claim in claims:
            evidence_sentences = select_evidence_sentences(claim, source_sentences, top_n=evidence_top_n)
            premise = " ".join(evidence_sentences)
            label, score = self.classify(premise, claim)
            results.append(
                ClaimVerification(
                    claim_text=claim,
                    label=label,
                    score=score,
                    source_evidence=premise or None,
                )
            )
        return results


def _normalize_label(raw_label: str) -> NLILabel:
    """Different MNLI-finetuned checkpoints name labels differently
    (LABEL_0/1/2, ENTAILMENT/NEUTRAL/CONTRADICTION, contradiction/neutral/entailment...);
    normalize to our `NLILabel` enum so downstream code doesn't need to know
    which checkpoint produced the score."""
    normalized = raw_label.strip().lower()
    if "entail" in normalized or normalized == "label_2":
        return NLILabel.ENTAILMENT
    if "contra" in normalized or normalized == "label_0":
        return NLILabel.CONTRADICTION
    return NLILabel.NEUTRAL
