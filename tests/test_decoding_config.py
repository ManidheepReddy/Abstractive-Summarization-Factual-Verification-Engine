from summarization_engine.models import DecodingConfig, DecodingStrategy


def test_beam_search_config_produces_beam_kwargs():
    config = DecodingConfig(strategy=DecodingStrategy.BEAM_SEARCH, num_beams=5, length_penalty=1.2)
    kwargs = config.to_generate_kwargs()
    assert kwargs["do_sample"] is False
    assert kwargs["num_beams"] == 5
    assert kwargs["length_penalty"] == 1.2
    assert "temperature" not in kwargs


def test_sampling_config_produces_sampling_kwargs():
    config = DecodingConfig(strategy=DecodingStrategy.SAMPLING, temperature=0.7, top_k=40, top_p=0.9)
    kwargs = config.to_generate_kwargs()
    assert kwargs["do_sample"] is True
    assert kwargs["num_beams"] == 1
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_k"] == 40
    assert kwargs["top_p"] == 0.9
    assert "length_penalty" not in kwargs


def test_shared_repetition_controls_present_in_both_strategies():
    for strategy in (DecodingStrategy.BEAM_SEARCH, DecodingStrategy.SAMPLING):
        kwargs = DecodingConfig(strategy=strategy, no_repeat_ngram_size=4, repetition_penalty=1.3).to_generate_kwargs()
        assert kwargs["no_repeat_ngram_size"] == 4
        assert kwargs["repetition_penalty"] == 1.3
