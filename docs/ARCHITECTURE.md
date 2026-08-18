# Architecture Notes

## What's fully implemented (runs, tested, no external model needed)

- **Preprocessing** (`ingestion/preprocessor.py`) — length filtering, sentence
  splitting, sentence-aligned chunking with overlap. Covered by
  `tests/test_preprocessor.py`.
- **Decoding mechanics** (`summarization/decoding.py`) — n-gram repetition
  blocking (mirrors HF's `NoRepeatNGramLogitsProcessor`), CTRL-style
  repetition penalty, top-k filtering, nucleus (top-p) filtering, temperature
  scaling — all implemented at the logit level and tested against tiny fake
  vocabularies. Covered by `tests/test_decoding.py`.
- **`DecodingConfig` → HF `generate()` kwargs mapping** (`models.py`) —
  covered by `tests/test_decoding_config.py`.
- **Lexical metrics** (`evaluation/lexical_metrics.py`) — ROUGE-1/2 (n-gram
  F1), ROUGE-L (LCS-based F1), and BLEU (n-gram precision + brevity penalty),
  all pure Python. Covered by `tests/test_lexical_metrics.py`.
- **Factuality scoring** (`evaluation/factuality.py`) — % entailed + full
  label breakdown. Covered by `tests/test_evaluation.py`.
- **Lead-N baseline** (`evaluation/baseline.py`) — covered by
  `tests/test_evaluation.py`.
- **Claim extraction & evidence-sentence selection**
  (`verification/claim_extraction.py`, the Jaccard-overlap ranking in
  `verification/nli_verifier.py`) — covered by `tests/test_verification.py`.
- **Pipeline orchestration** (`pipeline.py`) — wires all of the above into
  one `summarize_and_verify()` call.
- **FastAPI service** (`api/main.py`) — `/summarize`, `/health`.

## What's implemented but needs external models/weights to execute

Written against real Hugging Face APIs (not mocked), not run end-to-end here
since that requires downloading model weights and, for fine-tuning, a
domain-specific dataset:

- `summarization/model.py` — `Seq2SeqSummarizer` wraps
  `AutoModelForSeq2SeqLM` + `AutoTokenizer` for a T5/BART checkpoint.
- `summarization/trainer.py` — fine-tuning harness via `Seq2SeqTrainer`,
  `Seq2SeqTrainingArguments`, ROUGE-based `compute_metrics`.
- `verification/nli_verifier.py`'s `NLIVerifier.classify()` — wraps a
  Hugging Face `text-classification` pipeline for the DistilBERT NLI model.
- `evaluation/semantic_metrics.py` — `bert_score.score()`; returns `None`
  gracefully if `bert-score` isn't installed, same pattern as the sibling
  financial-RAG project's optional RAGAS wrapper.
- `ingestion/preprocessor.py`'s `SubwordTokenizer` — real WordPiece/BPE
  encode/decode via `AutoTokenizer`, separate from the word-count heuristic
  (`approx_token_count`) used for testable length filtering/chunking.

## Suggested next build steps

1. **Pick a domain dataset and fine-tune.** The trainer harness expects a
   `datasets.Dataset` with `document`/`summary` columns — something like
   incident post-mortems, PubMed abstracts, or a legal-brief summarization
   set, depending on which domain you want the portfolio piece to showcase.
2. **Calibrate the NLI evidence window.** `select_evidence_sentences` uses
   lexical (Jaccard) overlap as a cheap, dependency-free proxy for
   relevance — once real embeddings are available elsewhere in your stack,
   swapping in cosine similarity over sentence embeddings will likely
   produce a tighter premise, especially for claims that paraphrase the
   source heavily.
3. **Validate NLI label mapping against the chosen checkpoint.** Different
   MNLI-tuned checkpoints order their labels differently (`LABEL_0/1/2` vs.
   named labels); `_normalize_label()` handles the common cases but should be
   spot-checked against whatever checkpoint you actually load.
4. **Build the adversarial hallucination test set.** To claim "near-zero
   hallucination" credibly, you need summaries with deliberately injected
   false claims and confirm the NLI verifier actually flags them as
   contradictions — right now the detector is implemented but its catch rate
   is unmeasured.
5. **Decide on chunked-document summarization strategy.** For inputs beyond
   the encoder's context window, `chunk_long_document` produces
   sentence-aligned chunks, but there's no map-reduce summarization step yet
   (summarize each chunk, then summarize the summaries) — needed for
   documents longer than ~1024 tokens.
6. **UI for contradiction highlighting.** The project description calls for
   highlighting contradicted sentences in an output UI; `SummaryReport.
   contradicted_claims` already surfaces exactly that list — the rendering
   layer doesn't exist yet.
