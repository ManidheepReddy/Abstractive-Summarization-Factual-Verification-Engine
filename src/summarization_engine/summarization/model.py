"""Wraps a fine-tuned T5/BART encoder-decoder checkpoint for summary generation.

The encoder builds a bidirectional context representation of the source
document; the decoder attends to it via cross-attention (plus self-attention
over its own partial output) to generate the summary autoregressively.
Decoding behavior itself (beam search vs. sampling, repetition control) is
fully parameterized by `DecodingConfig` — see `decoding.py` for the
lower-level repetition-blocking mechanics this config drives.
"""
from __future__ import annotations

from summarization_engine.models import DecodingConfig


class Seq2SeqSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """`model_name` should point at a checkpoint already fine-tuned on the
        target domain (technical reports, biomedical papers, etc.) — see
        `trainer.py` for the fine-tuning harness that produces one."""
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as e:
                raise RuntimeError(
                    "transformers is required. `pip install transformers torch`."
                ) from e
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        return self._model, self._tokenizer

    def summarize(self, text: str, decoding_config: DecodingConfig, max_input_tokens: int = 1024) -> str:
        model, tokenizer = self._load()
        inputs = tokenizer(
            text, max_length=max_input_tokens, truncation=True, return_tensors="pt"
        )
        generate_kwargs = decoding_config.to_generate_kwargs()
        output_ids = model.generate(**inputs, **generate_kwargs)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
