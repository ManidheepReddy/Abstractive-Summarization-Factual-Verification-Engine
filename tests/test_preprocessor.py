from summarization_engine.ingestion.preprocessor import (
    approx_token_count,
    chunk_long_document,
    filter_by_length,
    split_sentences,
)


def test_filter_by_length_rejects_too_short():
    result = filter_by_length("Too short.", min_tokens=50, max_tokens=1000)
    assert result.keep is False
    assert "min_tokens" in result.reason


def test_filter_by_length_rejects_too_long():
    long_text = "word " * 2000
    result = filter_by_length(long_text, min_tokens=1, max_tokens=100)
    assert result.keep is False
    assert "max_tokens" in result.reason


def test_filter_by_length_accepts_within_bounds():
    text = "This is a reasonably sized document. " * 10
    result = filter_by_length(text, min_tokens=5, max_tokens=10000)
    assert result.keep is True


def test_split_sentences_basic():
    text = "The system failed at 2pm. Root cause was a config error. It was fixed by 3pm."
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("The system failed")


def test_split_sentences_empty_string():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_chunk_long_document_never_splits_a_sentence():
    sentences = [f"Sentence number {i} contains some words for padding." for i in range(20)]
    text = " ".join(sentences)
    chunks = chunk_long_document(text, max_tokens=30, overlap_tokens=0)

    assert len(chunks) > 1
    # Every original sentence should appear whole in exactly one (or more, with overlap) chunk.
    for sentence in sentences:
        assert any(sentence in chunk for chunk in chunks)


def test_chunk_long_document_respects_overlap():
    sentences = [f"Sentence {i} has extra padding words here." for i in range(10)]
    text = " ".join(sentences)
    chunks = chunk_long_document(text, max_tokens=20, overlap_tokens=10)
    assert len(chunks) > 1
    # With overlap, the tail of one chunk should reappear at the start of the next.
    assert any(chunks[i].split()[0] in chunks[i - 1] for i in range(1, len(chunks)))


def test_approx_token_count_positive_for_nonempty_text():
    assert approx_token_count("hello world") > 0
