"""Central, env-driven configuration for the summarization & factual
verification engine.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Models ---
    summarization_model: str = "facebook/bart-large-cnn"
    nli_model: str = "typeform/distilbert-base-uncased-mnli"

    # --- Ingestion / length filtering ---
    min_input_tokens: int = 50
    max_input_tokens: int = 100_000
    chunk_max_tokens: int = 900
    chunk_overlap_tokens: int = 50

    # --- Decoding defaults ---
    default_num_beams: int = 4
    default_max_new_tokens: int = 200
    default_no_repeat_ngram_size: int = 3

    # --- Verification ---
    nli_evidence_top_n: int = 2


settings = Settings()
