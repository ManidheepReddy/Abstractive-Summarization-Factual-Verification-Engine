# Abstractive Summarization & Factual Verification Engine

**Domain-Adapted Abstractive Summarization & Hallucination Auditing Engine**

A production-shaped NLP system that fine-tunes a T5/BART encoder-decoder for
domain-specific abstractive summarization (incident post-mortems, research
papers, legal briefs), gives explicit control over decoding behavior, and
verifies every generated claim against the source document with a DistilBERT
NLI classifier — turning "fluent but maybe wrong" into an auditable output
with per-sentence factuality labels.

## Why this exists

Generic LLM summarization is fluent but opaque: no control over decoding
behavior beyond default API settings, and no signal on which sentences are
actually supported by the source vs. quietly hallucinated. This system
addresses both:

| Gap | This system's answer |
|---|---|
| No control over generation mechanics | `DecodingConfig` + explicit beam/sampling/n-gram-blocking implementation |
| No hallucination signal | Per-claim NLI verification (entailment / neutral / contradiction) against the source |
| No benchmark beyond "looks good" | ROUGE/BLEU/BERTScore + factuality score, benchmarked against a Lead-3 baseline |

## Architecture

```
[Raw Long-Form Text]
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ 1. Ingestion & Preprocessing                            │
│    • Length filtering, sentence-aligned chunking         │
│    • Subword tokenization (WordPiece/BPE via HF)         │
└────────────────────────┬───────────────────────────────┘
                          ▼
┌────────────────────────────────────────────────────────┐
│ 2. Fine-Tuned Encoder-Decoder (T5 / BART)                │
│    • Cross-attention + self-attention                    │
│    • Configurable decoding: beam search / sampling /     │
│      n-gram blocking / repetition penalty                │
└────────────────────────┬───────────────────────────────┘
                          ▼
┌────────────────────────────────────────────────────────┐
│ 3. NLI Hallucination Detector (DistilBERT)                │
│    • Claim extraction (sentence-level)                   │
│    • Evidence-sentence selection from source              │
│    • Entailment / Neutral / Contradiction classification │
└────────────────────────┬───────────────────────────────┘
                          ▼
┌────────────────────────────────────────────────────────┐
│ 4. Evaluation Suite                                       │
│    • ROUGE-1/2/L, BLEU (lexical)                          │
│    • BERTScore (semantic)                                  │
│    • Factuality score = % entailed claims                  │
│    • Benchmarked against Lead-3 extractive baseline        │
└────────────────────────────────────────────────────────┘
```

## Repository layout

```
abstractive-summarization-factual-verification/
├── config/settings.py                     # central config (env-driven)
├── src/summarization_engine/
│   ├── models.py                          # shared dataclasses (DecodingConfig, ClaimVerification, SummaryReport, ...)
│   ├── ingestion/
│   │   └── preprocessor.py                # length filtering, sentence-aligned chunking, tokenizer wrapper
│   ├── summarization/
│   │   ├── model.py                       # Seq2SeqSummarizer (T5/BART wrapper)
│   │   ├── decoding.py                    # n-gram blocking, repetition penalty, top-k/top-p — implemented explicitly
│   │   └── trainer.py                     # HF Trainer-based fine-tuning harness
│   ├── verification/
│   │   ├── claim_extraction.py            # splits summary into verifiable claim sentences
│   │   └── nli_verifier.py                # DistilBERT NLI classifier + evidence-sentence selection
│   ├── evaluation/
│   │   ├── lexical_metrics.py             # ROUGE-1/2/L, BLEU (pure Python)
│   │   ├── semantic_metrics.py            # BERTScore (optional dependency)
│   │   ├── factuality.py                  # % entailed claims + full label breakdown
│   │   └── baseline.py                    # Lead-N extractive baseline
│   └── pipeline.py                        # end-to-end orchestration
├── api/main.py                            # FastAPI service (/summarize)
├── tests/                                 # unit tests (42 passing, no model downloads required)
└── docs/ARCHITECTURE.md                   # deeper design notes
```

## Objectives → implementation mapping

**A. Fine-tuned encoder-decoder** → `summarization/model.py` wraps a T5/BART
checkpoint; `summarization/trainer.py` fine-tunes one via Hugging Face
`Seq2SeqTrainer` with ROUGE as the validation metric.

**B. Controlled decoding & inference tuning** → `models.DecodingConfig` +
`summarization/decoding.py`. Beam width/length penalty flow through to HF's
`generate()`; n-gram blocking, repetition penalty, and top-k/top-p filtering
are implemented explicitly at the logit level (not just passed as kwargs) so
the mechanics are visible and unit-tested independent of any model.

**C. Automated NLI hallucination detector** → `verification/claim_extraction.py`
splits a summary into claims; `verification/nli_verifier.py` selects the most
relevant source sentence(s) per claim and classifies the pair with a
DistilBERT NLI model into entailment/neutral/contradiction.

**D. Evaluation suite** → `evaluation/lexical_metrics.py` (ROUGE/BLEU, pure
Python), `evaluation/semantic_metrics.py` (BERTScore, optional),
`evaluation/factuality.py` (% entailed), `evaluation/baseline.py` (Lead-3) —
all wired together in `pipeline.py`'s `SummaryReport`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Running the API

```bash
uvicorn api.main:app --reload
```

`POST /summarize` — send `{"document": "...", "reference_summary": "..." }`,
get back the generated summary, the Lead-3 baseline, per-claim NLI labels,
a factuality score, and (if a reference was provided) ROUGE/BLEU.

## Running tests

```bash
pytest -q
```

42 tests cover preprocessing, decoding mechanics, lexical metrics,
factuality scoring, the Lead-N baseline, claim extraction, and NLI
evidence-sentence selection — all without downloading any model weights.

## Status

Interfaces, decoding mechanics, chunking, lexical/factuality metrics,
evidence-sentence selection, and orchestration are implemented and tested.
The model-backed stages (T5/BART generation, DistilBERT NLI classification,
BERTScore) are wired against real Hugging Face APIs but need model weights
and a domain dataset to run end-to-end. See `docs/ARCHITECTURE.md` for the
implemented-vs-stubbed breakdown and suggested next steps.
