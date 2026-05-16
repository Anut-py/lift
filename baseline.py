#!/usr/bin/env python3
"""
Standalone baseline SFT trainer (v2 — uses model.load/save_baseline_model).

Trains purely on (prompt -> ground-truth answer) pairs with uniform cross-entropy
and no KL regularisation — a clean accuracy-only baseline to compare against
more sophisticated runs.

Imports from your existing project files (config, datagen, model/) but makes
no changes to any of them.

Usage:
    python train_baseline.py

Output model is saved via save_baseline_model() to a checkpoint directory
distinct from the aligned model so neither can clobber the other.
"""

import os
import random
import gc

import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer

# ── project imports (read-only) ───────────────────────────────────────────────
from config import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    GRAD_ACCUM_STEPS,
    MAX_LENGTH,
    out_subdir,
)
from datagen import dataset_sources, format_entry, format_answer
from model.model import load_tokenizer, load_baseline_model, save_baseline_model

# ── baseline-specific config ──────────────────────────────────────────────────

# How many examples to sample per dataset. Tune to taste.
BASELINE_PER_DATASET = 500

# Prompt format. "answer_only" trains directly on the final answer token(s).
# "cot" uses the dataset's reasoning trace (gsm8k only; others fall back to answer_only).
PROMPT_FORMAT = "answer_only"  # "answer_only" | "cot"

# ─────────────────────────────────────────────────────────────────────────────


def clear_cuda():
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


def build_examples(tokenizer):
    """
    Sample from every dataset source and return a list of
    {"input_ids": [...], "labels": [...]} dicts ready for Dataset.from_list().

    Labels are -100 for the prompt portion (no loss) and token ids for the
    target portion (loss computed here).
    """
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    examples = []
    skipped = 0

    for dataset_name, dataset in dataset_sources.items():
        n = min(BASELINE_PER_DATASET, len(dataset))
        indices = random.sample(range(len(dataset)), k=n)

        for i in indices:
            entry = dataset[i]
            prompt = format_entry(entry, dataset_name)
            answer = str(format_answer(entry, dataset_name)).strip()

            # ── build input / target strings ──────────────────────────────
            if PROMPT_FORMAT == "cot" and dataset_name == "gsm8k":
                # gsm8k stores the full reasoning in entry["answer"] before "####"
                reasoning = entry["answer"].split("####")[0].strip()
                inp = f"Q: {prompt}\nReasoning:\n"
                tgt = f"{reasoning}\nAnswer: {answer}"
            else:
                inp = f"Q: {prompt}\nAnswer: "
                tgt = answer

            # ── tokenise ─────────────────────────────────────────────────
            inp_ids = tokenizer.encode(inp, add_special_tokens=False)
            tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)

            if len(inp_ids) + len(tgt_ids) > MAX_LENGTH:
                skipped += 1
                continue

            input_ids = inp_ids + tgt_ids
            labels    = [-100] * len(inp_ids) + tgt_ids

            examples.append({"input_ids": input_ids, "labels": labels})

    print(f"[Baseline] built {len(examples)} examples ({skipped} skipped for length)")
    return examples


def data_collator(tokenizer):
    """Returns a collate_fn closed over the tokenizer."""
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = [
            x["input_ids"] + [pad_id] * (max_len - len(x["input_ids"]))
            for x in batch
        ]
        labels = [
            x["labels"] + [-100] * (max_len - len(x["labels"]))
            for x in batch
        ]
        attn_mask = [
            [1] * len(x["input_ids"]) + [0] * (max_len - len(x["input_ids"]))
            for x in batch
        ]
        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask,  dtype=torch.long),
            "labels":         torch.tensor(labels,     dtype=torch.long),
        }

    return collate


def run_baseline():
    print("[Baseline] Loading tokenizer and model …")
    tokenizer = load_tokenizer()
    model     = load_baseline_model(trainable=True)

    print("[Baseline] Building dataset …")
    examples = build_examples(tokenizer)
    if not examples:
        raise ValueError("No training examples were produced — check dataset sources.")

    hf_ds = Dataset.from_list(examples)

    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_ds,
        data_collator=data_collator(tokenizer),
        tokenizer=tokenizer,
    )

    print("[Baseline] Starting training …")
    trainer.train()
    print("[Baseline] Training complete.")

    save_baseline_model(model)
    print("[Baseline] Model saved.")

    del tokenizer, model, trainer
    clear_cuda()


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    run_baseline()
    print("[Baseline] Done.")