from summarization_engine.evaluation.lexical_metrics import bleu, rouge_1, rouge_2, rouge_l


def test_rouge_1_identical_text_is_one():
    text = "the quick brown fox jumps over the lazy dog"
    assert rouge_1(text, text) == 1.0


def test_rouge_1_no_overlap_is_zero():
    assert rouge_1("the quick brown fox", "completely different words here") == 0.0


def test_rouge_1_partial_overlap_between_bounds():
    score = rouge_1("the quick brown fox jumps", "the quick brown cat sleeps")
    assert 0.0 < score < 1.0


def test_rouge_2_stricter_than_rouge_1_on_partial_match():
    ref = "the quick brown fox jumps over the lazy dog"
    cand = "the lazy quick fox brown jumps dog over the"  # same words, shuffled
    r1 = rouge_1(ref, cand)
    r2 = rouge_2(ref, cand)
    assert r1 == 1.0  # unigrams: all words present
    assert r2 < r1     # bigrams broken by shuffling


def test_rouge_l_identical_text_is_one():
    text = "root cause was a misconfigured load balancer"
    assert rouge_l(text, text) == 1.0


def test_rouge_l_empty_candidate_is_zero():
    assert rouge_l("some reference text", "") == 0.0


def test_bleu_identical_text_is_one():
    text = "the incident was resolved within thirty minutes"
    assert bleu(text, text) == 1.0


def test_bleu_empty_candidate_is_zero():
    assert bleu("some reference text here", "") == 0.0


def test_bleu_zero_when_no_ngram_overlap():
    assert bleu("the quick brown fox", "completely different words entirely") == 0.0
