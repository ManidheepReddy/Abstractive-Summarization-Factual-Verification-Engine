"""End-to-end orchestration: preprocess -> generate summary -> extract &
verify claims via NLI -> score factuality -> (optionally) score against a
reference summary and the Lead-N baseline -> assemble a `SummaryReport`.

This is the single entry point the API layer calls; it composes every other
module in this package without containing generation/verification logic itself.
"""
from __future__ import annotations

from summarization_engine.evaluation.baseline import lead_n_summary
from summarization_engine.evaluation.factuality import factuality_score
from summarization_engine.evaluation.lexical_metrics import compute_lexical_metrics
from summarization_engine.evaluation.semantic_metrics import bertscore
from summarization_engine.ingestion.preprocessor import LengthFilterResult, filter_by_length
from summarization_engine.models import DecodingConfig, SummaryReport
from summarization_engine.summarization.model import Seq2SeqSummarizer
from summarization_engine.verification.nli_verifier import NLIVerifier


class SummarizationPipeline:
    def __init__(
        self,
        summarizer: Seq2SeqSummarizer,
        verifier: NLIVerifier,
        min_input_tokens: int = 50,
        max_input_tokens: int = 100_000,
    ):
        self.summarizer = summarizer
        self.verifier = verifier
        self.min_input_tokens = min_input_tokens
        self.max_input_tokens = max_input_tokens

    def summarize_and_verify(
        self,
        source_document: str,
        decoding_config: DecodingConfig | None = None,
        reference_summary: str | None = None,
        compute_baseline: bool = True,
        compute_semantic: bool = False,
    ) -> SummaryReport:
        decoding_config = decoding_config or DecodingConfig()

        length_check = self._check_length(source_document)
        if not length_check.keep:
            raise ValueError(f"Document rejected by length filter: {length_check.reason}")

        summary_text = self.summarizer.summarize(source_document, decoding_config)

        claims = self.verifier.verify_claims(summary_text, source_document)

        report = SummaryReport(
            source_document=source_document,
            summary_text=summary_text,
            decoding_config=decoding_config,
            claims=claims,
            factuality_score=factuality_score(claims),
        )

        if compute_baseline:
            report.baseline_summary = lead_n_summary(source_document, n=3)

        if reference_summary:
            report.lexical_metrics = compute_lexical_metrics(reference_summary, summary_text)
            if compute_semantic:
                report.semantic_metrics = bertscore(reference_summary, summary_text)

        return report

    def _check_length(self, text: str) -> LengthFilterResult:
        return filter_by_length(text, min_tokens=self.min_input_tokens, max_tokens=self.max_input_tokens)
