"""Fine-tuning harness for adapting a T5/BART checkpoint to a specialized
domain (incident post-mortems, biomedical papers, legal briefs, ...).

Wraps `Seq2SeqTrainer` rather than reimplementing the training loop — the
value-add here is the domain-appropriate defaults (label smoothing, ROUGE as
the validation metric, generation-time eval) rather than the loop mechanics
themselves.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FineTuningConfig:
    base_model: str = "facebook/bart-base"
    output_dir: str = "./checkpoints/domain-adapted-summarizer"
    learning_rate: float = 3e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    num_train_epochs: int = 3
    max_input_length: int = 1024
    max_target_length: int = 256
    label_smoothing_factor: float = 0.1
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50


def preprocess_for_training(examples: dict, tokenizer, config: FineTuningConfig) -> dict:
    """Maps a batch of {"document": [...], "summary": [...]} to tokenized
    model inputs, to be used with `datasets.Dataset.map(..., batched=True)`.
    """
    model_inputs = tokenizer(
        examples["document"],
        max_length=config.max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=config.max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_compute_metrics(tokenizer):
    """Returns a `compute_metrics` callable for `Seq2SeqTrainer` that reports
    ROUGE on decoded predictions vs. references during evaluation."""

    def compute_metrics(eval_pred):
        from summarization_engine.evaluation.lexical_metrics import rouge_1, rouge_2, rouge_l

        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # -100 is the ignore-index the Trainer uses for padded label positions.
        labels = [[tok if tok != -100 else tokenizer.pad_token_id for tok in seq] for seq in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        r1 = sum(rouge_1(ref, pred) for ref, pred in zip(decoded_labels, decoded_preds)) / len(decoded_preds)
        r2 = sum(rouge_2(ref, pred) for ref, pred in zip(decoded_labels, decoded_preds)) / len(decoded_preds)
        rl = sum(rouge_l(ref, pred) for ref, pred in zip(decoded_labels, decoded_preds)) / len(decoded_preds)
        return {"rouge1": r1, "rouge2": r2, "rougeL": rl}

    return compute_metrics


def run_finetuning(config: FineTuningConfig, train_dataset, eval_dataset):
    """Fine-tunes `config.base_model` on `train_dataset`/`eval_dataset`
    (each a `datasets.Dataset` with "document"/"summary" columns) and saves
    the resulting checkpoint to `config.output_dir`.
    """
    try:
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as e:
        raise RuntimeError("transformers is required for fine-tuning. `pip install transformers datasets`.") from e

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model)

    train_tokenized = train_dataset.map(
        lambda ex: preprocess_for_training(ex, tokenizer, config), batched=True
    )
    eval_tokenized = eval_dataset.map(
        lambda ex: preprocess_for_training(ex, tokenizer, config), batched=True
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        label_smoothing_factor=config.label_smoothing_factor,
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=build_compute_metrics(tokenizer),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return trainer
