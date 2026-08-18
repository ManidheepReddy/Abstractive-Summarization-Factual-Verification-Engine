from summarization_engine.verification.claim_extraction import extract_claims
from summarization_engine.verification.nli_verifier import _normalize_label, select_evidence_sentences
from summarization_engine.models import NLILabel


def test_extract_claims_splits_into_sentences():
    summary = "Revenue grew 12% year over year. The company also announced a new product line."
    claims = extract_claims(summary)
    assert len(claims) == 2


def test_extract_claims_drops_short_fragments():
    summary = "Yes. Revenue grew significantly in the reported quarter."
    claims = extract_claims(summary, min_words=3)
    assert len(claims) == 1
    assert "Revenue grew" in claims[0]


def test_select_evidence_sentences_ranks_by_overlap():
    claim = "The outage was caused by a database connection leak."
    source_sentences = [
        "The team ordered lunch at noon.",
        "A connection leak in the database caused the outage.",
        "Weather was sunny that day.",
    ]
    best = select_evidence_sentences(claim, source_sentences, top_n=1)
    assert best[0] == "A connection leak in the database caused the outage."


def test_select_evidence_sentences_handles_empty_source():
    result = select_evidence_sentences("some claim", [], top_n=2)
    assert result == []


def test_normalize_label_handles_common_variants():
    assert _normalize_label("ENTAILMENT") == NLILabel.ENTAILMENT
    assert _normalize_label("contradiction") == NLILabel.CONTRADICTION
    assert _normalize_label("LABEL_1") == NLILabel.NEUTRAL
    assert _normalize_label("neutral") == NLILabel.NEUTRAL
