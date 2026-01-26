#!/usr/bin/env python3
from config import BATCH_SIZE, EPOCHS, LR, GRAD_ACCUM_STEPS, MAX_LENGTH, KL_LAMBDA, ANSWER_WEIGHT, SAMPLE_WEIGHT, SCORE_THRESHOLD, SCORE_ANSWER_THRESHOLD, DATA_DIR, out_subdir, _model_tag
from datagen import dataset_sources, format_entry, format_answer
from copy import deepcopy
import pickle
from model.model import load_tokenizer, load_aligned_model, load_base_model
import random
import os
import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from torch.nn import functional as F
from torch.optim import AdamW
import argparse

def run_sft(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    data = [({**y, 'samples': [(z, w, i) for i, (z, w) in enumerate(y['samples']) if len(z) != i + 1]}) for y in data if len(y['samples'])]

    pmap = dict()
    dsmap = dict()
    
    for k, v in dataset_sources.items():
        for vv in v:
            dsmap[format_entry(vv, k)] = k
            pmap[format_entry(vv, k)] = (
                format_answer(vv, k),
                (ANSWER_WEIGHT[0] if "gsm" in k else ANSWER_WEIGHT[1] if "qa" in k else ANSWER_WEIGHT[2]),
                (SAMPLE_WEIGHT[0] if "gsm" in k else SAMPLE_WEIGHT[1] if "qa" in k else SAMPLE_WEIGHT[2])
            )

    kept = []
    set_aside = []
    l, h = SCORE_THRESHOLD
    
    from collections import defaultdict
    for entry in data:
        samples = entry.get('samples', [])

        outside = [s for s in samples if s[1] < l or s[1] > h]

        if outside:
            set_aside.append({
                'prompt': entry['prompt'],
                'original': entry['original'],
                'samples': outside
            })

        inside = [s for s in samples if s[1] >= l and s[1] <= h]

        if inside:
            new_entry = deepcopy(entry)
            new_entry['samples'] = inside
            new_entry['answer'] = pmap[entry['prompt']][0] if entry['prompt'] in pmap else None
            new_entry['answer_weight'] = pmap[entry['prompt']][1] if entry['prompt'] in pmap else 0
            new_entry['mult'] = pmap[entry['prompt']][2] if entry['prompt'] in pmap else 1.0
            kept.append(new_entry)

    random.seed(42)
    random.shuffle(kept) # make sure we don't have long stretches of data from the same dataset
    
    tokenizer = load_tokenizer()
    model = load_aligned_model()
    ref_model = load_base_model()
    
    model.train()
    ref_model.eval()
    
    device = next(model.parameters()).device
    
    def _join_trace(trace):
        if isinstance(trace, (list, tuple)):
            return "\n".join(s.strip() for s in trace if s is not None)
        return str(trace)

    examples = []
    raw_scores = [float(sc) for e in kept for _, sc, _ in e.get("samples", [])]
    if not raw_scores:
        raise ValueError("kept contains no samples")
    mn, mx = min(raw_scores), max(raw_scores)
    denom = max(1e-9, mx - mn)
    eos = tokenizer.eos_token or ""

    for e in kept:
        prompt = e["prompt"].strip()
        avg = 0
        for trace, score, step in e.get("samples", []):
            weight = (float(score) - mn) / denom
            weight = (0.05 + 0.95 * weight) ** (1/2)
            add_ans = e['answer_weight'] > 0 and weight >= SCORE_ANSWER_THRESHOLD
            weight *= e['mult']
            avg += weight
            inp = f"Q: {prompt}\nReasoning:\n{_join_trace(trace[:step+1])}\n"
            tgt = f"{_join_trace(trace[step+1:])}"
            ans = None
            if add_ans and e['answer'] is not None:
                tgt += "\n"
                ans = f"Answer: {e['answer']}\n\n"
            inp_ids = tokenizer.encode(inp, add_special_tokens=False)
            tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)
            token_weights = [0] * len(inp_ids) + [weight] * len(tgt_ids)
            if ans is not None:
                ans_ids = tokenizer.encode(ans, add_special_tokens=False)
                tgt_ids += ans_ids
                token_weights += [e['answer_weight'] * weight] * len(ans_ids)
            if len(inp_ids) + len(tgt_ids) > MAX_LENGTH:
                continue
            input_ids = inp_ids + tgt_ids
            labels = [-100] * len(inp_ids) + tgt_ids
            examples.append({"input_ids": input_ids, "labels": labels, "token_weights": token_weights})
    hf_ds = Dataset.from_list(examples)

    def data_collator(batch):
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = [x["input_ids"] + [pad_id] * (max_len - len(x["input_ids"])) for x in batch]
        labels = [x["labels"] + [-100] * (max_len - len(x["labels"])) for x in batch]
        attention_mask = [[1] * len(x["input_ids"]) + [0] * (max_len - len(x["input_ids"])) for x in batch]
        token_weights = [x["token_weights"] + [0] * (max_len - len(x["token_weights"])) for x in batch]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_weights": torch.tensor(token_weights, dtype=torch.float)
        }

    from torch.nn import functional as F
    import gc
    import torch
    
    class WeightedSFTTrainer(Trainer):
        def __init__(self, ref_model=None, kl_lambda=0.5, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ref_model = ref_model
            self.kl_lambda = kl_lambda
            if self.ref_model is not None:
                self.ref_model.to(self.model.device)
                self.ref_model.eval()
                for p in self.ref_model.parameters():
                    p.requires_grad = False
            self.step = 0
    
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            self.step += 1
            if (self.step % 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
            token_weights = inputs.pop("token_weights", None)
            device = self.model.device
            tensor_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
            token_weights = token_weights.to(device).float()[..., 1:].contiguous()
        
            labels = tensor_inputs["labels"]
            outputs = model(**tensor_inputs)
            logits = outputs.logits  # (B, S, V)
        
            # --- SHIFT for causal LM: predict token t using logits at t-1 ---
            shift_logits = logits[..., :-1, :].contiguous()          # (B, S-1, V)
            shift_labels = labels[..., 1:].contiguous()             # (B, S-1)
            mask = (shift_labels != -100).float()                   # (B, S-1)
        
            vocab = shift_logits.size(-1)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            flat_logits = shift_logits.view(-1, vocab)
            flat_labels = shift_labels.view(-1)
            token_losses = loss_fct(flat_logits, flat_labels).view(shift_labels.size(0), -1) * mask
        
            token_loss_sum = (token_losses * token_weights).sum(dim=1)
            denom = token_weights.sum(dim=1).clamp(min=1.0)
            per_sample_ce = (token_losses * token_weights).sum(dim=1) / denom
            weighted_ce = per_sample_ce.mean()
            total_loss = weighted_ce
        
            # --- KL (compare next-token distributions) ---
            if self.ref_model is not None and self.kl_lambda > 0:
                with torch.no_grad():
                    ref_logits = self.ref_model(
                        input_ids=tensor_inputs["input_ids"],
                        attention_mask=tensor_inputs.get("attention_mask", None)
                    ).logits
                ref_shift = ref_logits[..., :-1, :].contiguous()
                ref_logp = F.log_softmax(ref_shift, dim=-1)
                model_logp = F.log_softmax(shift_logits, dim=-1)
                ref_p = torch.exp(ref_logp)
                per_token_kl = (ref_p * (ref_logp - model_logp)).sum(dim=-1)    # (B, S-1)
                per_sample_kl = (per_token_kl * token_weights).sum(dim=1) / denom
                weighted_kl = per_sample_kl.mean()
                total_loss = total_loss + self.kl_lambda * weighted_kl
    
            del token_weights, tensor_inputs, logits, \
                shift_logits, labels, shift_labels, \
                mask, vocab, flat_logits, flat_labels, \
                token_losses
        
            return (total_loss, outputs) if return_outputs else total_loss
    
    training_args = TrainingArguments(
        output_dir=out_subdir + "/training-output",
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",
        logging_steps=5,
    )
    
    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        ref_model=ref_model,
        kl_lambda=KL_LAMBDA
    )
    
    trainer.train()
    
    from model.model import save_aligned_model
    save_aligned_model(model)

    del tokenizer, model, ref_model, trainer

# ---------- main ----------
if __name__ == "__main__":
    run_sft(f"{DATA_DIR}/{_model_tag}-datagen.pkl")