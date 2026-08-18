from summarization_engine.evaluation.baseline import lead_n_summary
from summarization_engine.evaluation.factuality import factuality_breakdown, factuality_score
from summarization_engine.models import ClaimVerification, NLILabel


def _claim(label: NLILabel) -> ClaimVerification:
    return ClaimVerification(claim_text="some claim", label=label, score=0.9)


def test_factuality_score_all_entailed_is_one():
    claims = [_claim(NLILabel.ENTAILMENT) for _ in range(3)]
    assert factuality_score(claims) == 1.0


def test_factuality_score_mixed_labels():
    claims = [_claim(NLILabel.ENTAILMENT), _claim(NLILabel.CONTRADICTION), _claim(NLILabel.NEUTRAL), _claim(NLILabel.ENTAILMENT)]
    assert factuality_score(claims) == 0.5


def test_factuality_score_empty_claims_is_zero():
    assert factuality_score([]) == 0.0


def test_factuality_breakdown_sums_to_one():
    claims = [_claim(NLILabel.ENTAILMENT), _claim(NLILabel.CONTRADICTION), _claim(NLILabel.NEUTRAL)]
    breakdown = factuality_breakdown(claims)
    assert abs(sum(breakdown.values()) - 1.0) < 1e-9
    assert breakdown["entailment"] == breakdown["contradiction"] == breakdown["neutral"]


def test_lead_n_summary_takes_first_n_sentences():
    document = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
    summary = lead_n_summary(document, n=2)
    assert "First sentence" in summary
    assert "Second sentence" in summary
    assert "Third sentence" not in summary


def test_lead_n_summary_handles_short_document():
    document = "Only one sentence."
    summary = lead_n_summary(document, n=3)
    assert summary == "Only one sentence."
