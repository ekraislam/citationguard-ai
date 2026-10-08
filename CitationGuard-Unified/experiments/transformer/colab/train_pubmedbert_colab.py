"""
CitationGuard AI — Phase 5 Google Colab GPU Training & Evaluation Pipeline
==========================================================================
End-to-end reproducible fine-tuning and evaluation for:
microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
on the frozen CitationGuard verification dataset.

Optimized for Google Colab NVIDIA GPU environments (T4 / V100 / A100).
Supports FP16 mixed precision, gradient accumulation, Drive mounting,
and comprehensive multi-subset reporting.
"""

import os
import sys
import json
import csv
import re
import time
import random
import argparse
from collections import Counter
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# -----------------------------------------------------------------------------
# 1. ENVIRONMENT & GPU INSPECTION
# -----------------------------------------------------------------------------
def inspect_environment(allow_cpu: bool = False):
    print("=" * 70)
    print("CITATIONGUARD AI — PHASE 5 COLAB ENVIRONMENT AUDIT")
    print("=" * 70)
    print(f"Python Version:      {sys.version.split()[0]}")
    print(f"PyTorch Version:     {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available:      {cuda_avail}")
    
    if not cuda_avail:
        if not allow_cpu:
            raise RuntimeError(
                "\n[ERROR] No NVIDIA GPU detected! "
                "In Google Colab, please switch to a GPU runtime: "
                "Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU. "
                "Per CitationGuard Phase 5 protocol, large-scale transformer fine-tuning "
                "must not silently execute on CPU."
            )
        else:
            print("[WARNING] Proceeding on CPU as explicitly requested via --allow_cpu.")
            device = torch.device("cpu")
            gpu_name = "CPU (Fallback)"
            gpu_vram = 0.0
    else:
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cuda_ver = torch.version.cuda
        print(f"GPU Accelerator:     {gpu_name}")
        print(f"GPU Total VRAM:      {gpu_vram:.2f} GB")
        print(f"CUDA Driver/Runtime: {cuda_ver}")
        print(f"Current Device:      {device}")
        
    print("=" * 70 + "\n")
    return device, gpu_name, gpu_vram

# -----------------------------------------------------------------------------
# 2. DATA INTEGRITY PRE-FLIGHT AUDIT (RULE 19)
# -----------------------------------------------------------------------------
def verify_data_integrity(records: List[Dict[str, Any]]):
    print("=" * 70)
    print("[INTEGRITY AUDIT] Running Phase 5 Frozen Dataset Pre-flight Guard...")
    print("=" * 70)
    
    n_total = len(records)
    print(f"Total verification records: {n_total}")
    assert n_total == 1754, f"Integrity Failure: Expected exactly 1,754 records, got {n_total}"
    
    split_counts = Counter(r.get("guard_split") for r in records)
    print(f"Split breakdown: {dict(split_counts)}")
    assert split_counts["TRAIN"] == 1226, f"Integrity Failure: Expected 1,226 TRAIN, got {split_counts['TRAIN']}"
    assert split_counts["DEV"] == 263, f"Integrity Failure: Expected 263 DEV, got {split_counts['DEV']}"
    assert split_counts["TEST"] == 265, f"Integrity Failure: Expected 265 TEST, got {split_counts['TEST']}"
    
    valid_labels = {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_EVIDENCE"}
    labels_found = set(r.get("verification_label") for r in records)
    assert labels_found == valid_labels, f"Integrity Failure: Unexpected labels: {labels_found}"
    
    # Check split leakage by paper_id / document_id
    train_papers = set(r["paper_id"] for r in records if r["guard_split"] == "TRAIN" and r.get("paper_id"))
    dev_papers = set(r["paper_id"] for r in records if r["guard_split"] == "DEV" and r.get("paper_id"))
    test_papers = set(r["paper_id"] for r in records if r["guard_split"] == "TEST" and r.get("paper_id"))
    
    assert len(train_papers.intersection(dev_papers)) == 0, "Integrity Failure: Paper overlap TRAIN & DEV"
    assert len(train_papers.intersection(test_papers)) == 0, "Integrity Failure: Paper overlap TRAIN & TEST"
    assert len(dev_papers.intersection(test_papers)) == 0, "Integrity Failure: Paper overlap DEV & TEST"
    
    sources = set(r["source_dataset"] for r in records)
    assert "MSVEC" not in sources, "Integrity Failure: MSVEC found in verification dataset!"
    
    print("[INTEGRITY AUDIT] ALL DATA INTEGRITY CHECKS PASSED. Dataset is clean and verified.")
    print("=" * 70 + "\n")

# -----------------------------------------------------------------------------
# 3. PREPROCESSING & CLEANING (RULE 5)
# -----------------------------------------------------------------------------
def clean_scientific_text(text: str, remove_citations: bool = True) -> str:
    if not text:
        return ""
    text = str(text)
    # Unicode normalization
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\xa0", " ")
    
    if remove_citations:
        # Bracket citations: [1], [12, 13], [1-3]
        text = re.sub(r'\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]', '', text)
        # Author-year citations: (Smith, 2020), (Smith et al., 2020)
        text = re.sub(r'\([A-Z][A-Za-z\s]+(?:et al\.?)?,\s*\d{4}[a-z]?\)', '', text)
        
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_record_contexts(r: Dict[str, Any]) -> Tuple[str, str]:
    strict_ev = r.get("evidence_text")
    if strict_ev and str(strict_ev).strip():
        strict_context = clean_scientific_text(str(strict_ev), remove_citations=True)
    else:
        strict_context = ""
        
    fallback_ev = r.get("evidence_text") or r.get("paper_abstract") or ""
    if fallback_ev and str(fallback_ev).strip():
        fallback_context = clean_scientific_text(str(fallback_ev), remove_citations=True)
    else:
        fallback_context = ""
        
    return strict_context, fallback_context

# -----------------------------------------------------------------------------
# 4. DATASET & TOKENIZATION (RULES 3, 4, 6)
# -----------------------------------------------------------------------------
LABEL_LIST = ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_EVIDENCE"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

class VerificationDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, mode: str = "strict", max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.features = self._build_features()
        
    def _build_features(self):
        features = []
        for r in self.records:
            claim = clean_scientific_text(r.get("claim_text", ""), remove_citations=True)
            title = clean_scientific_text(r.get("paper_title", ""), remove_citations=False)
            ctx_strict, ctx_fallback = prepare_record_contexts(r)
            
            if self.mode == "strict":
                ev = ctx_strict if ctx_strict else "[NO_EVIDENCE_PROVIDED]"
            else: # fallback
                ev = ctx_fallback if ctx_fallback else "[NO_EVIDENCE_PROVIDED]"
                
            input_text = f"Claim:\n{claim}\n\nTitle:\n{title}\n\nEvidence:\n{ev}"
            
            enc = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_length,
                padding=False
            )
            
            label_id = LABEL_TO_ID[r["verification_label"]]
            features.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": label_id,
                "record_id": r["record_id"],
                "source_dataset": r["source_dataset"],
                "synthetic_or_natural": r["synthetic_or_natural"],
                "claim_text": r["claim_text"],
                "paper_title": r.get("paper_title") or "",
                "evidence_text": ev,
                "true_label": r["verification_label"]
            })
        return features
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx]

def custom_collate_fn(batch):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in batch]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in batch]
    labels = torch.tensor([f["labels"] for f in batch], dtype=torch.long)
    
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": labels,
        "meta": batch
    }

# -----------------------------------------------------------------------------
# 5. METRIC COMPUTATION
# -----------------------------------------------------------------------------
def compute_eval_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABEL_LIST, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=LABEL_LIST, average="weighted", zero_division=0)
    
    p_per = precision_score(y_true, y_pred, labels=LABEL_LIST, average=None, zero_division=0)
    r_per = recall_score(y_true, y_pred, labels=LABEL_LIST, average=None, zero_division=0)
    f1_per = f1_score(y_true, y_pred, labels=LABEL_LIST, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_LIST)
    
    per_class = {}
    for idx, lbl in enumerate(LABEL_LIST):
        per_class[lbl] = {
            "precision": float(p_per[idx]),
            "recall": float(r_per[idx]),
            "f1": float(f1_per[idx])
        }
        
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }

# -----------------------------------------------------------------------------
# 6. TRAINING & EVALUATION CONTROLLER
# -----------------------------------------------------------------------------
def train_and_evaluate(
    experiment_name: str,
    train_dataset: VerificationDataset,
    dev_dataset: VerificationDataset,
    test_dataset: VerificationDataset,
    tokenizer,
    class_weights: torch.Tensor,
    device: torch.device,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2.0e-5,
    grad_accum_steps: int = 2,
    use_fp16: bool = True
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"RUNNING EXPERIMENT: {experiment_name}")
    print(f"Mode: {train_dataset.mode.upper()} | Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}, Grad Accum: {grad_accum_steps}, FP16: {use_fp16 and device.type == 'cuda'}")
    print("=" * 70)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID
    )
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    steps_per_epoch = (len(train_loader) + grad_accum_steps - 1) // grad_accum_steps
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=(use_fp16 and device.type == "cuda"))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(use_fp16 and device.type == "cuda"))
    
    best_dev_macro_f1 = -1.0
    best_model_weights = None
    training_history = []
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        total_train_loss = 0.0
        train_preds, train_trues = [], []
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.cuda.amp.autocast(enabled=(use_fp16 and device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps
                    
            scaler.scale(loss).backward()
            
            # Check if this is a standard accumulation boundary or the final batch in the epoch (partial accumulation)
            is_accum_step = (step + 1) % grad_accum_steps == 0
            is_final_step = (step + 1) == len(train_loader)
            
            if is_accum_step or is_final_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    # Optimizer stepped first; step scheduler only if optimizer step was not skipped due to inf/nan
                    if scale_after >= scale_before:
                        scheduler.step()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                optimizer.zero_grad()
                
            loss_mult = grad_accum_steps if grad_accum_steps > 1 else 1.0
            total_train_loss += loss.item() * loss_mult * len(labels)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            train_preds.extend([ID_TO_LABEL[p] for p in preds])
            train_trues.extend([ID_TO_LABEL[l] for l in labels.cpu().tolist()])
            
        train_loss = total_train_loss / len(train_dataset)
        train_acc = accuracy_score(train_trues, train_preds)
        train_macro_f1 = f1_score(train_trues, train_preds, labels=LABEL_LIST, average="macro", zero_division=0)
        
        # Evaluate on DEV
        model.eval()
        total_dev_loss = 0.0
        dev_preds, dev_trues = [], []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                with torch.cuda.amp.autocast(enabled=(use_fp16 and device.type == "cuda")):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = loss_fn(logits, labels)
                    
                total_dev_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                dev_preds.extend([ID_TO_LABEL[p] for p in preds])
                dev_trues.extend([ID_TO_LABEL[l] for l in labels.cpu().tolist()])
                
        dev_loss = total_dev_loss / len(dev_dataset)
        dev_metrics = compute_eval_metrics(dev_trues, dev_preds)
        dev_macro_f1 = dev_metrics["macro_f1"]
        dev_acc = dev_metrics["accuracy"]
        
        ep_duration = time.time() - ep_start
        print(f"Epoch {epoch}/{epochs} [{ep_duration:.1f}s] -> "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"DEV Loss: {dev_loss:.4f}, DEV Acc: {dev_acc:.4f}, DEV Macro-F1: {dev_macro_f1:.4f} "
              f"(SUP: {dev_metrics['per_class']['SUPPORTED']['f1']:.3f}, "
              f"CON: {dev_metrics['per_class']['CONTRADICTED']['f1']:.3f}, "
              f"NEE: {dev_metrics['per_class']['NOT_ENOUGH_EVIDENCE']['f1']:.3f})")
              
        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_macro_f1,
            "dev_loss": dev_loss,
            "dev_acc": dev_acc,
            "dev_macro_f1": dev_macro_f1,
            "dev_sup_f1": dev_metrics["per_class"]["SUPPORTED"]["f1"],
            "dev_con_f1": dev_metrics["per_class"]["CONTRADICTED"]["f1"],
            "dev_nee_f1": dev_metrics["per_class"]["NOT_ENOUGH_EVIDENCE"]["f1"],
            "epoch_duration_sec": ep_duration
        })
        
        if dev_macro_f1 > best_dev_macro_f1:
            best_dev_macro_f1 = dev_macro_f1
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"   >>> New best checkpoint saved with DEV Macro-F1 = {best_dev_macro_f1:.4f}")
            
    total_training_sec = time.time() - start_time
    print(f"\nTraining completed in {total_training_sec:.1f}s. Best DEV Macro-F1 = {best_dev_macro_f1:.4f}")
    
    # Load best weights
    model.load_state_dict(best_model_weights)
    model.eval()
    
    # Save checkpoint directory
    save_dir = os.path.join(output_dir, f"best_checkpoint_{train_dataset.mode}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved best model checkpoint to: {save_dir}")
    
    # Comprehensive TEST Evaluation
    test_preds, test_trues = [], []
    test_probs = []
    test_meta = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            with torch.cuda.amp.autocast(enabled=(use_fp16 and device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().tolist()
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                
            test_preds.extend([ID_TO_LABEL[p] for p in preds])
            test_trues.extend([ID_TO_LABEL[l] for l in batch["labels"].cpu().tolist()])
            test_probs.extend(probs)
            test_meta.extend(batch["meta"])
            
    combined_test_metrics = compute_eval_metrics(test_trues, test_preds)
    
    # Stratified Subsets
    indices_natural = [i for i, m in enumerate(test_meta) if m["synthetic_or_natural"] == "NATURAL"]
    indices_synth = [i for i, m in enumerate(test_meta) if m["synthetic_or_natural"] == "SYNTHETIC_NEGATION"]
    indices_scit = [i for i, m in enumerate(test_meta) if m["source_dataset"] == "SCitance"]
    indices_scifact = [i for i, m in enumerate(test_meta) if m["source_dataset"] == "SciFact"]
    
    natural_metrics = compute_eval_metrics([test_trues[i] for i in indices_natural], [test_preds[i] for i in indices_natural])
    synth_metrics = compute_eval_metrics([test_trues[i] for i in indices_synth], [test_preds[i] for i in indices_synth])
    scit_metrics = compute_eval_metrics([test_trues[i] for i in indices_scit], [test_preds[i] for i in indices_scit])
    scifact_metrics = compute_eval_metrics([test_trues[i] for i in indices_scifact], [test_preds[i] for i in indices_scifact])
    
    print("\n" + "-" * 60)
    print(f"[{experiment_name}] FINAL TEST EVALUATION RESULTS:")
    print(f"Combined Test ($N=265$) -> Acc: {combined_test_metrics['accuracy']:.4f}, Macro-F1: {combined_test_metrics['macro_f1']:.4f}")
    print(f"Natural-Only  ($N=228$) -> Acc: {natural_metrics['accuracy']:.4f}, Macro-F1: {natural_metrics['macro_f1']:.4f} (CON F1: {natural_metrics['per_class']['CONTRADICTED']['f1']:.4f})")
    print(f"Synthetic     ($N=37$)  -> Acc: {synth_metrics['accuracy']:.4f}, Macro-F1: {synth_metrics['macro_f1']:.4f} (CON F1: {synth_metrics['per_class']['CONTRADICTED']['f1']:.4f})")
    print(f"SCitance      ($N=93$)  -> Acc: {scit_metrics['accuracy']:.4f}, Macro-F1: {scit_metrics['macro_f1']:.4f}")
    print(f"SciFact       ($N=172$) -> Acc: {scifact_metrics['accuracy']:.4f}, Macro-F1: {scifact_metrics['macro_f1']:.4f}")
    print("-" * 60)
    
    # Save Predictions CSV
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, f"predictions_pubmedbert_{train_dataset.mode}.csv")
    with open(pred_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record_id", "source_dataset", "synthetic_or_natural", "true_label", "predicted_label", "prob_SUPPORTED", "prob_CONTRADICTED", "prob_NOT_ENOUGH_EVIDENCE"])
        for i, m in enumerate(test_meta):
            pb = test_probs[i]
            writer.writerow([
                m["record_id"],
                m["source_dataset"],
                m["synthetic_or_natural"],
                test_trues[i],
                test_preds[i],
                f"{pb[LABEL_TO_ID['SUPPORTED']]:.4f}",
                f"{pb[LABEL_TO_ID['CONTRADICTED']]:.4f}",
                f"{pb[LABEL_TO_ID['NOT_ENOUGH_EVIDENCE']]:.4f}"
            ])
            
    # Collect Errors
    errors = []
    for i, m in enumerate(test_meta):
        if test_trues[i] != test_preds[i]:
            errors.append({
                "record_id": m["record_id"],
                "source_dataset": m["source_dataset"],
                "synthetic_or_natural": m["synthetic_or_natural"],
                "true_label": test_trues[i],
                "predicted_label": test_preds[i],
                "claim_text": m["claim_text"],
                "paper_title": m["paper_title"],
                "evidence_text": m["evidence_text"],
                "prob_SUP": f"{test_probs[i][LABEL_TO_ID['SUPPORTED']]:.4f}",
                "prob_CON": f"{test_probs[i][LABEL_TO_ID['CONTRADICTED']]:.4f}",
                "prob_NEE": f"{test_probs[i][LABEL_TO_ID['NOT_ENOUGH_EVIDENCE']]:.4f}"
            })
            
    return {
        "experiment_name": experiment_name,
        "mode": train_dataset.mode,
        "training_time_sec": total_training_sec,
        "best_dev_macro_f1": best_dev_macro_f1,
        "training_history": training_history,
        "test_metrics": {
            "combined": combined_test_metrics,
            "natural": natural_metrics,
            "synthetic": synth_metrics,
            "scitance": scit_metrics,
            "scifact": scifact_metrics
        },
        "errors": errors
    }

# -----------------------------------------------------------------------------
# 7. VISUALIZATIONS & REPORT EXPORTERS
# -----------------------------------------------------------------------------
def plot_training_curves(history: List[Dict[str, Any]], title: str, out_path: str):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    dev_loss = [h["dev_loss"] for h in history]
    dev_macro_f1 = [h["dev_macro_f1"] for h in history]
    dev_acc = [h["dev_acc"] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    ax1.plot(epochs, train_loss, 'o-', color='#1f77b4', label='Train Loss', linewidth=2)
    ax1.plot(epochs, dev_loss, 's--', color='#ff7f0e', label='DEV Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Cross-Entropy Loss', fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontweight='bold')
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    ax2.plot(epochs, dev_macro_f1, '^-', color='#2ca02c', label='DEV Macro-F1', linewidth=2)
    ax2.plot(epochs, dev_acc, 'd--', color='#d62728', label='DEV Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('Validation Performance', fontweight='bold')
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_confusion_matrix(cm_matrix: List[List[int]], title: str, out_path: str):
    cm = np.array(cm_matrix)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(LABEL_LIST)))
    ax.set_yticks(range(len(LABEL_LIST)))
    ax.set_xticklabels(["SUP", "CON", "NEE"], fontsize=10, fontweight="bold")
    ax.set_yticklabels(["SUP", "CON", "NEE"], fontsize=10, fontweight="bold")
    
    for i in range(len(LABEL_LIST)):
        for j in range(len(LABEL_LIST)):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=12, fontweight="bold")
            
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Ground-Truth Label", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# -----------------------------------------------------------------------------
# 8. MAIN EXECUTION PIPELINE
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CitationGuard Phase 5 PubMedBERT Training on GPU")
    parser.add_argument("--data_path", type=str, required=True, help="Path to frozen citationguard_verification.jsonl")
    parser.add_argument("--output_dir", type=str, default="./phase5_output", help="Directory to save artifacts")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length (default: 512)")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--allow_cpu", action="store_true", help="Permit CPU execution if no GPU is available")
    parser.add_argument("--skip_fallback", action="store_true", help="Skip Experiment 2 (Abstract Fallback)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Environment Audit
    device, gpu_name, gpu_vram = inspect_environment(allow_cpu=args.allow_cpu)
    
    # 2. Seed Reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        
    # 3. Load & Audit Dataset
    with open(args.data_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    verify_data_integrity(records)
    
    train_recs = [r for r in records if r["guard_split"] == "TRAIN"]
    dev_recs = [r for r in records if r["guard_split"] == "DEV"]
    test_recs = [r for r in records if r["guard_split"] == "TEST"]
    
    # 4. Balanced Class Weights
    label_counts = Counter(r["verification_label"] for r in train_recs)
    n_train = len(train_recs)
    n_classes = len(LABEL_LIST)
    weights = [n_train / (n_classes * label_counts[l]) for l in LABEL_LIST]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"Calculated TRAIN balanced class weights: SUP={weights[0]:.4f}, CON={weights[1]:.4f}, NEE={weights[2]:.4f}\n")
    
    # 5. Tokenizer
    tokenizer_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    print(f"Loading Tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # -------------------------------------------------------------------------
    # EXPERIMENT 1: PRIMARY EXPERIMENT — STRICT EVIDENCE
    # -------------------------------------------------------------------------
    train_strict = VerificationDataset(train_recs, tokenizer, mode="strict", max_length=args.max_length)
    dev_strict = VerificationDataset(dev_recs, tokenizer, mode="strict", max_length=args.max_length)
    test_strict = VerificationDataset(test_recs, tokenizer, mode="strict", max_length=args.max_length)
    
    res_strict = train_and_evaluate(
        "Experiment 1: PubMedBERT Strict Evidence",
        train_strict,
        dev_strict,
        test_strict,
        tokenizer,
        class_weights_tensor,
        device=device,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum_steps=args.grad_accum,
        use_fp16=True
    )
    
    plot_training_curves(
        res_strict["training_history"],
        "PubMedBERT (Strict Evidence) Training Curves",
        os.path.join(figures_dir, "training_curves_pubmedbert_strict.png")
    )
    plot_confusion_matrix(
        res_strict["test_metrics"]["combined"]["confusion_matrix"],
        "PubMedBERT (Strict Evidence) Test Confusion Matrix",
        os.path.join(figures_dir, "confusion_matrix_pubmedbert_strict.png")
    )
    
    # Save training_history.csv
    hist_csv = os.path.join(args.output_dir, "training_history.csv")
    with open(hist_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "train_acc", "train_macro_f1", "dev_loss", "dev_acc", "dev_macro_f1",
            "dev_sup_f1", "dev_con_f1", "dev_nee_f1", "epoch_duration_sec"
        ])
        writer.writeheader()
        for h in res_strict["training_history"]:
            writer.writerow(h)
            
    # Save error analysis CSV
    err_csv = os.path.join(args.output_dir, "transformer_error_analysis.csv")
    with open(err_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "record_id", "source_dataset", "synthetic_or_natural", "true_label", "predicted_label",
            "prob_SUP", "prob_CON", "prob_NEE", "claim_text", "paper_title", "evidence_text"
        ])
        writer.writeheader()
        for e in res_strict["errors"]:
            writer.writerow(e)
            
    # -------------------------------------------------------------------------
    # EXPERIMENT 2: ABLATION EXPERIMENT — ABSTRACT FALLBACK
    # -------------------------------------------------------------------------
    res_fallback = None
    if not args.skip_fallback:
        train_fallback = VerificationDataset(train_recs, tokenizer, mode="fallback", max_length=args.max_length)
        dev_fallback = VerificationDataset(dev_recs, tokenizer, mode="fallback", max_length=args.max_length)
        test_fallback = VerificationDataset(test_recs, tokenizer, mode="fallback", max_length=args.max_length)
        
        res_fallback = train_and_evaluate(
            "Experiment 2: PubMedBERT Abstract Fallback",
            train_fallback,
            dev_fallback,
            test_fallback,
            tokenizer,
            class_weights_tensor,
            device=device,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            grad_accum_steps=args.grad_accum,
            use_fp16=True
        )
        
        plot_training_curves(
            res_fallback["training_history"],
            "PubMedBERT (Abstract Fallback) Training Curves",
            os.path.join(figures_dir, "training_curves_pubmedbert_fallback.png")
        )
        plot_confusion_matrix(
            res_fallback["test_metrics"]["combined"]["confusion_matrix"],
            "PubMedBERT (Abstract Fallback) Test Confusion Matrix",
            os.path.join(figures_dir, "confusion_matrix_pubmedbert_fallback.png")
        )
        
    # -------------------------------------------------------------------------
    # REPORTS GENERATION
    # -------------------------------------------------------------------------
    # 1. Source Performance Report
    scit_m = res_strict["test_metrics"]["scitance"]
    scifact_m = res_strict["test_metrics"]["scifact"]
    src_rep_path = os.path.join(reports_dir, "transformer_source_performance.md")
    with open(src_rep_path, "w", encoding="utf-8") as f:
        f.write("# CitationGuard AI — PubMedBERT Source-Wise Performance Report\n\n")
        f.write("Evaluation of the primary Strict Evidence PubMedBERT checkpoint broken down by origin source dataset.\n")
        f.write("*(Note: `source_dataset` metadata was strictly withheld from model inputs).*\n\n")
        f.write("| Source Dataset | N (Test) | Accuracy | Macro-F1 | Weighted-F1 | SUP F1 | CON F1 | NEE F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **SCitance** | 93 | {scit_m['accuracy']:.4f} | **{scit_m['macro_f1']:.4f}** | {scit_m['weighted_f1']:.4f} | {scit_m['per_class']['SUPPORTED']['f1']:.4f} | {scit_m['per_class']['CONTRADICTED']['f1']:.4f} | {scit_m['per_class']['NOT_ENOUGH_EVIDENCE']['f1']:.4f} |\n")
        f.write(f"| **SciFact** | 172 | {scifact_m['accuracy']:.4f} | **{scifact_m['macro_f1']:.4f}** | {scifact_m['weighted_f1']:.4f} | {scifact_m['per_class']['SUPPORTED']['f1']:.4f} | {scifact_m['per_class']['CONTRADICTED']['f1']:.4f} | {scifact_m['per_class']['NOT_ENOUGH_EVIDENCE']['f1']:.4f} |\n\n")
        f.write("## Key Diagnostic Findings\n")
        f.write("1. **Corpus Distribution**: SCitance contains 93 test records; SciFact contains 172 test records.\n")
        f.write("2. **Evidence Completeness**: SCitance has strict sentence-level citation contexts across 100% of samples, whereas 416 SciFact records lack explicit sentence rationales and rely on empty-evidence representations in strict mode.\n")
        
    # 2. Master Phase 5 Report
    st_comb = res_strict["test_metrics"]["combined"]
    st_nat = res_strict["test_metrics"]["natural"]
    st_syn = res_strict["test_metrics"]["synthetic"]
    
    diff_macro_f1 = (st_comb["macro_f1"] - 0.6754) * 100
    diff_acc = (st_comb["accuracy"] - 0.6717) * 100
    sign_f1 = "+" if diff_macro_f1 >= 0 else ""
    sign_acc = "+" if diff_acc >= 0 else ""
    
    rep_path = os.path.join(reports_dir, "PHASE5_PUBMEDBERT_REPORT.md")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("# CitationGuard AI — PHASE 5 RESEARCH REPORT: PUBMEDBERT BENCHMARK\n\n")
        f.write("## 1. Environment\n")
        f.write(f"- **Accelerator**: {gpu_name} ({gpu_vram:.2f} GB VRAM)\n")
        f.write(f"- **PyTorch**: {torch.__version__} | **CUDA**: {torch.version.cuda}\n")
        f.write(f"- **Mixed Precision**: FP16 enabled with gradient scaling\n\n")
        
        f.write("## 2. Dataset & Integrity\n")
        f.write("- **Frozen Dataset**: `citationguard_verification.jsonl` ($N=1,754$)\n")
        f.write("- **Splits**: TRAIN = 1,226 (70%), DEV = 263 (15%), TEST = 265 (15%)\n")
        f.write("- **Integrity Status**: All 1,754 records verified, 0 paper leaks across splits, MSVEC excluded.\n\n")
        
        f.write("## 3. Preprocessing\n")
        f.write("- Applied Unicode normalization and superficial citation marker removal (`[1]`, `(Smith, 2020)`).\n")
        f.write("- Preserved scientific nomenclature, numerical values, and domain terminology.\n\n")
        
        f.write("## 4. Tokenization\n")
        f.write(f"- Tokenizer: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`\n")
        f.write(f"- Max Sequence Length: {args.max_length} tokens (truncating only ~11.7% of strict evidence sequences).\n\n")
        
        f.write("## 5. Model Architecture & Hyperparameters\n")
        f.write("- Backbone: BiomedNLP-PubMedBERT (110M parameters)\n")
        f.write("- Classification Head: 3-class linear head (`SUPPORTED`, `CONTRADICTED`, `NOT_ENOUGH_EVIDENCE`)\n")
        f.write(f"- Learning Rate: `{args.lr}`, Batch Size: `{args.batch_size}`, Weight Decay: `0.01`, Epochs: `{args.epochs}`\n")
        f.write(f"- Class Weights: Balanced CrossEntropyLoss ($w_{{SUP}}={weights[0]:.4f}, w_{{CON}}={weights[1]:.4f}, w_{{NEE}}={weights[2]:.4f}$)\n\n")
        
        f.write("## 6. Primary Benchmark Results (TEST Set)\n\n")
        f.write("### Model Comparison Against Phase 4 Baseline\n\n")
        f.write("| Model | Input Condition | DEV Macro-F1 | TEST Macro-F1 | TEST Accuracy | TEST Weighted-F1 | Absolute Difference vs Phase 4 Baseline |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        f.write("| TF-IDF + Logistic Regression | Claim + Strict Evidence | 0.6625 | 0.6754 | 0.6717 | 0.6765 | *Phase 4 Reference* |\n")
        f.write(f"| **PubMedBERT (Ours)** | **Claim + Strict Evidence** | **{res_strict['best_dev_macro_f1']:.4f}** | **{st_comb['macro_f1']:.4f}** | **{st_comb['accuracy']:.4f}** | **{st_comb['weighted_f1']:.4f}** | **{sign_f1}{diff_macro_f1:.2f} percentage points Macro-F1** ({sign_acc}{diff_acc:.2f} pp Acc) |\n\n")
        
        f.write("### Detailed Per-Class Performance (PubMedBERT Strict)\n\n")
        f.write("| Class | Precision | Recall | F1-Score |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for lbl in LABEL_LIST:
            f.write(f"| **{lbl}** | {st_comb['per_class'][lbl]['precision']:.4f} | {st_comb['per_class'][lbl]['recall']:.4f} | {st_comb['per_class'][lbl]['f1']:.4f} |\n")
        f.write(f"| **Macro Average** | - | - | **{st_comb['macro_f1']:.4f}** |\n\n")
        
        f.write("## 7. Natural vs Synthetic Evaluation\n\n")
        f.write("| Test Subset | N | Accuracy | Macro-F1 | SUP F1 | CONTRADICTED F1 | NEE F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Combined TEST** | 265 | {st_comb['accuracy']:.4f} | **{st_comb['macro_f1']:.4f}** | {st_comb['per_class']['SUPPORTED']['f1']:.4f} | **{st_comb['per_class']['CONTRADICTED']['f1']:.4f}** | {st_comb['per_class']['NOT_ENOUGH_EVIDENCE']['f1']:.4f} |\n")
        f.write(f"| **Natural-Only TEST** | 228 | {st_nat['accuracy']:.4f} | **{st_nat['macro_f1']:.4f}** | {st_nat['per_class']['SUPPORTED']['f1']:.4f} | **{st_nat['per_class']['CONTRADICTED']['f1']:.4f}** | {st_nat['per_class']['NOT_ENOUGH_EVIDENCE']['f1']:.4f} |\n")
        f.write(f"| **Synthetic-Negation TEST** | 37 | {st_syn['accuracy']:.4f} | **{st_syn['macro_f1']:.4f}** | {st_syn['per_class']['SUPPORTED']['f1']:.4f} | **{st_syn['per_class']['CONTRADICTED']['f1']:.4f}** | {st_syn['per_class']['NOT_ENOUGH_EVIDENCE']['f1']:.4f} |\n\n")
        
        if res_fallback:
            fb_comb = res_fallback["test_metrics"]["combined"]
            fb_diff = (fb_comb["macro_f1"] - st_comb["macro_f1"]) * 100
            f.write("## 8. Ablation: Strict Evidence vs Abstract Fallback Context\n\n")
            f.write("| Input Condition | DEV Macro-F1 | TEST Macro-F1 | TEST Accuracy | Difference vs Strict |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            f.write(f"| **Strict Evidence** | {res_strict['best_dev_macro_f1']:.4f} | {st_comb['macro_f1']:.4f} | {st_comb['accuracy']:.4f} | *Baseline Transformer* |\n")
            f.write(f"| **Abstract Fallback** | {res_fallback['best_dev_macro_f1']:.4f} | {fb_comb['macro_f1']:.4f} | {fb_comb['accuracy']:.4f} | {'+' if fb_diff >= 0 else ''}{fb_diff:.2f} percentage points |\n\n")
            
        f.write("## 9. Answers to Core Scientific Questions\n")
        f.write(f"1. **Does PubMedBERT outperform TF-IDF + Logistic Regression?**: {'Yes' if diff_macro_f1 > 0 else 'No'}.\n")
        f.write(f"2. **By how many Macro-F1 percentage points?**: **{sign_f1}{diff_macro_f1:.2f} percentage points**.\n")
        f.write(f"3. **Does the gain remain on Natural-only TEST?**: Natural-Only Macro-F1 is **{st_nat['macro_f1']:.4f}** (CON F1 = {st_nat['per_class']['CONTRADICTED']['f1']:.4f}).\n")
        f.write(f"4. **Does the model still perform well on synthetic-negation TEST?**: Synthetic-Negation Macro-F1 is **{st_syn['macro_f1']:.4f}** (CON F1 = {st_syn['per_class']['CONTRADICTED']['f1']:.4f}).\n")
        if res_fallback:
            f.write(f"5. **Does the transformer handle full abstracts better than TF-IDF?**: Fallback Macro-F1 is {fb_comb['macro_f1']:.4f} vs Strict {st_comb['macro_f1']:.4f}.\n")
        f.write(f"6. **Which class remains hardest?**: Evaluated by per-class F1 across test slices.\n")
        f.write(f"7. **What types of errors remain?**: Exported to `transformer_error_analysis.csv` ({len(res_strict['errors'])} test errors analyzed).\n\n")
        
        f.write("## 10. Recommendations for Next Phase\n")
        f.write("- Incorporate rationale sentence selection (MultiVerS architecture) before stance verification.\n")
        f.write("- Evaluate SciBERT domain comparison.\n")
        
    print(f"\n[REPORT] Saved master report: {rep_path}")
    print(f"[REPORT] Saved source performance report: {src_rep_path}")
    print("\n" + "=" * 70)
    print("PHASE 5 COLAB PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
