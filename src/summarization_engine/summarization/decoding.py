"""Decoding-time mechanics implemented explicitly rather than left to library
defaults — this is the "controlled decoding" piece of the project: beam
width / length penalty are just `DecodingConfig` fields passed to
`generate()` (see `summarization/model.py`), but repetition control and
sampling truncation are implemented here at the logit level so the mechanics
are visible and testable independent of any model.

All functions operate on a plain `list[float]` of per-vocab-token logits
(index = token id) so they can be unit-tested with tiny fake vocabularies
instead of a real tokenizer/model.
"""
from __future__ import annotations

import math

_NEG_INF = float("-inf")


def get_banned_ngram_tokens(generated_ids: list[int], no_repeat_ngram_size: int) -> set[int]:
    """Returns the token ids that would complete an n-gram already seen
    earlier in `generated_ids` — the set `no_repeat_ngram_size`-gram blocking
    must forbid as the *next* token, mirroring the logic inside
    `NoRepeatNGramLogitsProcessor` in Hugging Face's generation loop.
    """
    if no_repeat_ngram_size <= 0 or len(generated_ids) < no_repeat_ngram_size - 1:
        return set()

    prefix_len = no_repeat_ngram_size - 1
    current_prefix = tuple(generated_ids[-prefix_len:]) if prefix_len else tuple()

    banned: set[int] = set()
    for i in range(len(generated_ids) - no_repeat_ngram_size + 1):
        ngram = tuple(generated_ids[i : i + no_repeat_ngram_size])
        if ngram[:prefix_len] == current_prefix:
            banned.add(ngram[-1])
    return banned


def apply_ngram_blocking(logits: list[float], generated_ids: list[int], no_repeat_ngram_size: int) -> list[float]:
    """Returns a copy of `logits` with banned n-gram continuations set to -inf."""
    banned = get_banned_ngram_tokens(generated_ids, no_repeat_ngram_size)
    if not banned:
        return list(logits)
    return [_NEG_INF if i in banned else v for i, v in enumerate(logits)]


def apply_repetition_penalty(logits: list[float], generated_ids: list[int], penalty: float) -> list[float]:
    """CTRL-style repetition penalty (Keskar et al. 2019): logits for tokens
    already generated are made less attractive — positive logits divided by
    `penalty`, negative logits multiplied by `penalty` — without a hard ban,
    unlike n-gram blocking above.
    """
    if penalty == 1.0:
        return list(logits)
    seen = set(generated_ids)
    out = list(logits)
    for token_id in seen:
        if token_id >= len(out):
            continue
        out[token_id] = out[token_id] / penalty if out[token_id] > 0 else out[token_id] * penalty
    return out


def top_k_filter(logits: list[float], k: int) -> list[float]:
    """Masks every logit outside the top-`k` to -inf. `k <= 0` disables filtering."""
    if k <= 0 or k >= len(logits):
        return list(logits)
    threshold = sorted(logits, reverse=True)[k - 1]
    return [v if v >= threshold else _NEG_INF for v in logits]


def _softmax(logits: list[float]) -> list[float]:
    finite = [v for v in logits if v != _NEG_INF]
    m = max(finite) if finite else 0.0
    exps = [math.exp(v - m) if v != _NEG_INF else 0.0 for v in logits]
    total = sum(exps)
    return [e / total for e in exps] if total > 0 else [0.0] * len(logits)


def top_p_filter(logits: list[float], p: float) -> list[float]:
    """Nucleus sampling: keeps the smallest set of highest-probability tokens
    whose cumulative probability mass is >= `p`, masking the rest to -inf.
    `p >= 1.0` disables filtering.
    """
    if p >= 1.0:
        return list(logits)
    probs = _softmax(logits)
    order = sorted(range(len(logits)), key=lambda i: probs[i], reverse=True)

    cumulative = 0.0
    keep: set[int] = set()
    for idx in order:
        if cumulative >= p and keep:
            break
        keep.add(idx)
        cumulative += probs[idx]

    return [v if i in keep else _NEG_INF for i, v in enumerate(logits)]


def apply_temperature(logits: list[float], temperature: float) -> list[float]:
    if temperature == 1.0 or temperature <= 0:
        return list(logits)
    return [v / temperature if v != _NEG_INF else v for v in logits]
