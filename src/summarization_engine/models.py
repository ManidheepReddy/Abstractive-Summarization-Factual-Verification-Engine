"""Shared data models used across ingestion, summarization, verification, and evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NLILabel(str, Enum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


class DecodingStrategy(str, Enum):
    BEAM_SEARCH = "beam_search"
    SAMPLING = "sampling"


@dataclass
class DecodingConfig:
    """Everything needed to reproduce a specific decoding run — kept as one
    object so a report can record exactly which settings produced a summary."""
    strategy: DecodingStrategy = DecodingStrategy.BEAM_SEARCH
    max_new_tokens: int = 200

    # Beam search knobs
    num_beams: int = 4
    length_penalty: float = 1.0

    # Sampling knobs
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95

    # Shared repetition control
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.0

    def to_generate_kwargs(self) -> dict:
        """Maps this config to Hugging Face `generate()` kwargs."""
        kwargs: dict = {
            "max_new_tokens": self.max_new_tokens,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.strategy == DecodingStrategy.BEAM_SEARCH:
            kwargs.update(
                do_sample=False,
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
            )
        else:
            kwargs.update(
                do_sample=True,
                num_beams=1,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
        return kwargs


@dataclass
class ClaimVerification:
    claim_text: str
    label: NLILabel
    score: float                    # NLI classifier confidence for `label`
    source_evidence: Optional[str] = None  # best-matching premise sentence, if tracked


@dataclass
class LexicalMetrics:
    rouge1: float
    rouge2: float
    rougeL: float
    bleu: float


@dataclass
class SummaryReport:
    source_document: str
    summary_text: str
    decoding_config: DecodingConfig
    claims: list[ClaimVerification] = field(default_factory=list)
    factuality_score: Optional[float] = None       # % of claims labeled ENTAILMENT
    lexical_metrics: Optional[LexicalMetrics] = None
    semantic_metrics: Optional[dict] = None         # e.g. {"bertscore_f1": 0.91}, None if unavailable
    baseline_summary: Optional[str] = None          # Lead-3 comparison point

    @property
    def contradicted_claims(self) -> list[ClaimVerification]:
        return [c for c in self.claims if c.label == NLILabel.CONTRADICTION]
