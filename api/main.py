"""FastAPI service for the Domain-Adapted Abstractive Summarization &
Hallucination Auditing Engine.

Endpoints:
  POST /summarize — generate a summary, verify every claim via NLI, and
                     return the full factuality-audited report
  GET  /health    — liveness check
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from config.settings import settings
from summarization_engine.models import DecodingConfig, DecodingStrategy
from summarization_engine.pipeline import SummarizationPipeline
from summarization_engine.summarization.model import Seq2SeqSummarizer
from summarization_engine.verification.nli_verifier import NLIVerifier

app = FastAPI(title="Abstractive Summarization & Factual Verification Engine", version="0.1.0")

_summarizer = Seq2SeqSummarizer(settings.summarization_model)
_verifier = NLIVerifier(settings.nli_model)
_pipeline = SummarizationPipeline(
    summarizer=_summarizer,
    verifier=_verifier,
    min_input_tokens=settings.min_input_tokens,
    max_input_tokens=settings.max_input_tokens,
)


class SummarizeRequest(BaseModel):
    document: str
    reference_summary: Optional[str] = None
    strategy: DecodingStrategy = DecodingStrategy.BEAM_SEARCH
    num_beams: int = 4
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    no_repeat_ngram_size: int = 3
    max_new_tokens: int = 200


class ClaimResponse(BaseModel):
    claim_text: str
    label: str
    score: float
    source_evidence: Optional[str]


class SummarizeResponse(BaseModel):
    summary: str
    baseline_summary: Optional[str]
    factuality_score: Optional[float]
    claims: list[ClaimResponse]
    lexical_metrics: Optional[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    decoding_config = DecodingConfig(
        strategy=req.strategy,
        num_beams=req.num_beams,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        no_repeat_ngram_size=req.no_repeat_ngram_size,
        max_new_tokens=req.max_new_tokens,
    )

    report = _pipeline.summarize_and_verify(
        source_document=req.document,
        decoding_config=decoding_config,
        reference_summary=req.reference_summary,
    )

    return SummarizeResponse(
        summary=report.summary_text,
        baseline_summary=report.baseline_summary,
        factuality_score=report.factuality_score,
        claims=[
            ClaimResponse(
                claim_text=c.claim_text,
                label=c.label.value,
                score=c.score,
                source_evidence=c.source_evidence,
            )
            for c in report.claims
        ],
        lexical_metrics=(
            {
                "rouge1": report.lexical_metrics.rouge1,
                "rouge2": report.lexical_metrics.rouge2,
                "rougeL": report.lexical_metrics.rougeL,
                "bleu": report.lexical_metrics.bleu,
            }
            if report.lexical_metrics
            else None
        ),
    )
