#!/usr/bin/env python
"""
HyenaDNA token-classification training script
 — now enhanced to
   • save train / val / test splits
   • save per-token truth + prediction csv
   • report total tokens, TP, FP, FN, TN
"""

import argparse
import os                      # NEW
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EarlyStoppingCallback
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,          # NEW
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import ModelOutput


print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.1f} GB")
###############################################################################
# 1) Custom Output Class
###############################################################################
@dataclass
class CustomTokenClassifierOutput(ModelOutput):
    """
    Mirrors TokenClassifierOutput (but compatible with older transformers).
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

###############################################################################
# 2) HyenaDNAForTokenClassification
###############################################################################
class HyenaDNAForTokenClassification(PreTrainedModel):
    base_model_prefix = "hyena"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.hyena = None  # will attach after loading
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        self.post_init()

    def forward(self, input_ids=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("attention_mask", None)

        outputs = self.hyena(input_ids=input_ids, **kwargs)
        seq_output = self.dropout(outputs[0])
        logits = self.classifier(seq_output)

        loss = None
        if labels is not None:
            class_weights = torch.tensor([0.2, 0.8],
                                         device=logits.device,
                                         dtype=torch.float)
            loss_fct = nn.CrossEntropyLoss(weight=class_weights,
                                           ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1))

        return CustomTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=getattr(outputs, "attentions", None),
        )

    @classmethod
    def from_pretrained_hyena(cls, pretrained_name_or_path,
                              config=None, trust_remote_code=True, **kwargs):
        if config is None:
            config = AutoConfig.from_pretrained(
                pretrained_name_or_path,
                trust_remote_code=trust_remote_code,
                **kwargs
            )

        print("[INFO] Initializing HyenaDNAForTokenClassification with config:",
              config)

        model = cls(config)

        print("[INFO] Loading HyenaDNA base model from:",
              pretrained_name_or_path)
        base_model = AutoModel.from_pretrained(
            pretrained_name_or_path,
            config=config,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        model.hyena = base_model
        return model

###############################################################################
# 3) CSV-based dataset
###############################################################################
class DNADataset(torch.utils.data.Dataset):
    MAX_LENGTH = 31000

    def __init__(self, df, tokenizer):
        self.tokenizer = tokenizer
        print("[INFO] Original dataset size:", len(df))

        filtered_df = df[df["sequence"].str.len() <= self.MAX_LENGTH] \
                        .copy().reset_index(drop=True)
        print(f"[INFO] Discarded {len(df) - len(filtered_df)} over-long seqs.")

        self.sequences = filtered_df["sequence"].tolist()
        self.labels = [list(map(int, s.split()))
                       for s in filtered_df["labels"].tolist()]
        print("[INFO] Remaining dataset size:", len(self.sequences))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_str = self.sequences[idx]
        labs = self.labels[idx]
        encoding = self.tokenizer(list(seq_str),
                                  is_split_into_words=True,
                                  return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "labels": torch.tensor(labs, dtype=torch.long),
        }

###############################################################################
# 4) Metric function (now with confusion-matrix counts)
###############################################################################
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    probs_pos = probs[:, :, 1]
    preds = (probs_pos >= 0.3).astype(int)

    valid_p, valid_l, valid_prob = [], [], []
    for p_row, l_row, pr_row in zip(preds, labels, probs_pos):
        mask = l_row != -100
        valid_p.extend(p_row[mask])
        valid_l.extend(l_row[mask])
        valid_prob.extend(pr_row[mask])

    valid_p = np.array(valid_p)
    valid_l = np.array(valid_l)

    tn, fp, fn, tp = confusion_matrix(valid_l, valid_p, labels=[0, 1]).ravel()
    total_tokens = len(valid_l)

    precision = precision_score(valid_l, valid_p, zero_division=0)
    recall = recall_score(valid_l, valid_p, zero_division=0)
    f1 = f1_score(valid_l, valid_p, zero_division=0)
    accuracy = (valid_p == valid_l).mean()
    try:
        auc = roc_auc_score(valid_l, np.array(valid_prob))
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "total_tokens": total_tokens,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

###############################################################################
# 5) Main
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--merged_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── 1) read CSV
    print("[INFO] Reading", args.merged_csv)
    df = pd.read_csv(args.merged_csv)

    # ── 2) shuffle & 60-20-20 split
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n = len(df)
    train_end, val_end = int(0.6*n), int(0.8*n)
    train_df, val_df, test_df = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    print(f"[INFO] splits — train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    # save splits
    os.makedirs(args.output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.output_dir, "train_split.csv"), index=False)
    val_df  .to_csv(os.path.join(args.output_dir, "val_split.csv"),   index=False)
    test_df .to_csv(os.path.join(args.output_dir, "test_split.csv"),  index=False)
    print(f"[INFO] Saved split CSVs to {args.output_dir}")

    # ── 3) config + model
    config = AutoConfig.from_pretrained(args.base_model_path, trust_remote_code=True)
    config.num_labels = args.num_labels
    model = HyenaDNAForTokenClassification.from_pretrained_hyena(
        args.base_model_path, config=config, trust_remote_code=True)

    # ── 4) tokenizer & datasets
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    train_ds = DNADataset(train_df, tokenizer)
    val_ds   = DNADataset(val_df,   tokenizer)
    test_ds  = DNADataset(test_df,  tokenizer)

    # # ── 5) trainer
    # targs = TrainingArguments(
    # output_dir="runs/exp1",
    # per_device_train_batch_size=2,   # ← training batch
    # per_device_eval_batch_size=2,    # ← ***NEW: eval batch = 1***
    # num_train_epochs=5,
    # evaluation_strategy="steps",
    # eval_steps=500,
    # save_strategy="steps",
    # save_steps=500,
    # load_best_model_at_end=True,
    # metric_for_best_model="f1",
    # )
    targs = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,      # Very small
    per_device_eval_batch_size=1,       # Very small  
    # gradient_accumulation_steps=8,      # Maintain effective batch size
    num_train_epochs=5,
    # max_steps=1000,                     # Test with fewer steps first
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    fp16=False,                          # Use mixed precision
    dataloader_pin_memory=False,        # Disable pin memory
    # gradient_checkpointing=True,        # Save memory
    save_safetensors=False,  # Add this line
    # Early stopping configuration
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # or "eval_loss" 
    greater_is_better=True,      # True for f1, False for loss
    evaluation_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        
        
    )

    # ── 6) train
    trainer.train()
    print("[INFO] Training complete.")

    # ── 7) validation metrics
    val_res = trainer.predict(val_ds)
    print("Validation metrics:", val_res.metrics)

    # ── 8) test metrics
    test_res = trainer.predict(test_ds)
    print("Test metrics:", test_res.metrics)

    # ── 9) save per-token predictions on test set
    logits = test_res.predictions
    probs_pos = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()[:, :, 1]
    preds = (probs_pos >= 0.3).astype(int)

    rows = []
    for seq_idx, (p_row, l_row) in enumerate(zip(preds, test_res.label_ids)):
        mask = l_row != -100
        for tok_idx in np.where(mask)[0]:
            rows.append({
                "sequence_id": seq_idx,
                "token_pos":   int(tok_idx),
                "true_label":  int(l_row[tok_idx]),
                "pred_label":  int(p_row[tok_idx]),
                "pred_prob":   float(probs_pos[seq_idx, tok_idx]),
            })

    tok_df = pd.DataFrame(rows)
    tok_path = os.path.join(args.output_dir, "test_token_predictions.csv")
    tok_df.to_csv(tok_path, index=False)
    print(f"[INFO] Saved token-level predictions → {tok_path}")

    # ── save final fine-tuned model & tokenizer ──
    final_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_dir, safe_serialization=False)
    if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(final_dir)  # No extra parameter needed
    print(f"[INFO] Final model + tokenizer saved to {final_dir}")

    # ── 10) summary banner
    print("\n================ FINAL SUMMARY ================")
    for k, v in test_res.metrics.items():
        print(f"{k:15s}: {v:.4f}" if isinstance(v, float) else f"{k:15s}: {v}")
    print("===============================================")


if __name__ == "__main__":
    main()
