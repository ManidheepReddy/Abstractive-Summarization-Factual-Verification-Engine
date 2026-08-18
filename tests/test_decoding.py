import math

from summarization_engine.summarization.decoding import (
    apply_ngram_blocking,
    apply_repetition_penalty,
    apply_temperature,
    get_banned_ngram_tokens,
    top_k_filter,
    top_p_filter,
)

_NEG_INF = float("-inf")


def test_get_banned_ngram_tokens_blocks_repeat_continuation():
    # Sequence: [1, 2, 3, 1, 2] with no_repeat_ngram_size=3 -> "1,2" already
    # followed by 3 once, so token 3 should be banned as the next token.
    banned = get_banned_ngram_tokens([1, 2, 3, 1, 2], no_repeat_ngram_size=3)
    assert 3 in banned


def test_get_banned_ngram_tokens_empty_for_short_sequence():
    banned = get_banned_ngram_tokens([1], no_repeat_ngram_size=3)
    assert banned == set()


def test_apply_ngram_blocking_masks_banned_token_to_neg_inf():
    logits = [1.0, 2.0, 3.0, 4.0]
    generated = [0, 1, 2, 0, 1]  # "0,1" was followed by 2 before -> ban token 2 next
    masked = apply_ngram_blocking(logits, generated, no_repeat_ngram_size=3)
    assert masked[2] == _NEG_INF
    assert masked[0] == 1.0  # untouched


def test_apply_repetition_penalty_reduces_positive_logit():
    logits = [4.0, -4.0, 1.0]
    generated = [0, 1]
    out = apply_repetition_penalty(logits, generated, penalty=2.0)
    assert out[0] == 2.0     # positive logit divided by penalty
    assert out[1] == -8.0    # negative logit multiplied by penalty
    assert out[2] == 1.0     # never generated -> untouched


def test_apply_repetition_penalty_noop_at_one():
    logits = [1.0, 2.0, 3.0]
    out = apply_repetition_penalty(logits, [0, 1], penalty=1.0)
    assert out == logits


def test_top_k_filter_keeps_only_top_k():
    logits = [1.0, 5.0, 3.0, 2.0, 4.0]
    filtered = top_k_filter(logits, k=2)
    kept = [v for v in filtered if v != _NEG_INF]
    assert sorted(kept) == [4.0, 5.0]


def test_top_k_filter_noop_when_k_exceeds_length():
    logits = [1.0, 2.0]
    assert top_k_filter(logits, k=10) == logits


def test_top_p_filter_keeps_smallest_nucleus():
    # Heavily skewed distribution: token 0 should dominate, so p=0.5 should
    # keep just it (or very few tokens).
    logits = [10.0, 0.0, 0.0, 0.0]
    filtered = top_p_filter(logits, p=0.5)
    kept_indices = [i for i, v in enumerate(filtered) if v != _NEG_INF]
    assert 0 in kept_indices
    assert len(kept_indices) < len(logits)


def test_top_p_filter_noop_at_p_one():
    logits = [1.0, 2.0, 3.0]
    assert top_p_filter(logits, p=1.0) == logits


def test_apply_temperature_scales_logits():
    logits = [2.0, 4.0]
    out = apply_temperature(logits, temperature=2.0)
    assert out == [1.0, 2.0]


def test_apply_temperature_noop_at_one():
    logits = [1.0, 2.0]
    assert apply_temperature(logits, temperature=1.0) == logits
