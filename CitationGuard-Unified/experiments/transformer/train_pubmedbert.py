"""
CitationGuard AI — Phase 5 Scientific Transformer Training Pipeline
===================================================================
Fine-tunes microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract for 3-class
scientific citation and claim verification under strict CPU-safe settings.

Includes:
- Rule 19 Data Integrity Pre-flight Check
- Deterministic Preprocessing and Safe Citation Marker Removal
- Primary Experiment: Strict Evidence (Claim + Title + Strict Evidence)
- Ablation Experiment: Abstract Fallback (Claim + Title + Full Abstract)
- Balanced Class Weighted Cross-Entropy Loss
- Dynamic Batch Padding with PyTorch OpenMP Multi-threading
- Per-epoch Evaluation on DEV with Early Stopping on DEV Macro-F1
- Stratified Evaluation on TEST: Combined, Natural-Only, Synthetic-Negation
- Diagnostic Source Evaluation: SCitance vs SciFact
- Export of Metrics, Logs, Confusion Matrices, Training Curves, and Error CSV
"""

import os
import sys
import json
import csv
import re
import time
import random
from collections import defaultdict, Counter
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
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# -----------------------------------------------------------------------------
# CONFIGURATION & REPRODUCIBILITY SEED
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Optimize PyTorch CPU execution
CPU_THREADS = 8
torch.set_num_threads(CPU_THREADS)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(CURRENT_DIR)
BASE_UNIFIED = os.path.dirname(EXPERIMENTS_DIR)

DATA_PATH = os.path.join(BASE_UNIFIED, "citationguard_verification.jsonl")
REPORTS_DIR = os.path.join(EXPERIMENTS_DIR, "reports")
FIGURES_DIR = os.path.join(EXPERIMENTS_DIR, "figures")
PREDICTIONS_DIR = os.path.join(EXPERIMENTS_DIR, "predictions")
TRANSFORMER_DIR = os.path.join(EXPERIMENTS_DIR, "transformer")

for d in [REPORTS_DIR, FIGURES_DIR, PREDICTIONS_DIR, TRANSFORMER_DIR]:
    os.makedirs(d, exist_ok=True)

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
LABEL_LIST = ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_EVIDENCE"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

# Hyperparameters (Optimized for 12th Gen Intel Core i5 CPU)
MAX_SEQ_LENGTH = 384
BATCH_SIZE = 16
LEARNING_RATE = 2.5e-5
WEIGHT_DECAY = 0.01
EPOCHS = 3
WARMUP_RATIO = 0.1

# Import preprocessing
sys.path.append(os.path.join(EXPERIMENTS_DIR, "preprocessing"))
from preprocess import clean_scientific_text, prepare_record_contexts

# -----------------------------------------------------------------------------
# 1. PRE-FLIGHT DATA INTEGRITY CHECK (RULE 19)
# -----------------------------------------------------------------------------
def verify_data_integrity(records: List[Dict[str, Any]]):
    """Rigorous verification before running training."""
    print("=" * 65)
    print("[INTEGRITY AUDIT] Running Phase 5 Data Integrity Pre-flight Guard...")
    print("=" * 65)
    
    assert len(records) == 1754, f"Integrity Failure: Expected 1,754 records, got {len(records)}"
    
    train_recs = [r for r in records if r["guard_split"] == "TRAIN"]
    dev_recs = [r for r in records if r["guard_split"] == "DEV"]
    test_recs = [r for r in records if r["guard_split"] == "TEST"]
    
    assert len(train_recs) == 1226, f"Integrity Failure: Expected 1,226 TRAIN, got {len(train_recs)}"
    assert len(dev_recs) == 263, f"Integrity Failure: Expected 263 DEV, got {len(dev_recs)}"
    assert len(test_recs) == 265, f"Integrity Failure: Expected 265 TEST, got {len(test_recs)}"
    
    labels = set(r["verification_label"] for r in records)
    assert labels == set(LABEL_LIST), f"Integrity Failure: Invalid labels {labels}"
    
    norm_claims = [re.sub(r'[^\w\s]', '', r["claim_text"].lower()) for r in records]
    assert len(set(norm_claims)) == 1754, "Integrity Failure: Duplicate normalized claims detected!"
    
    train_papers = set(str(r["paper_id"]) for r in train_recs)
    dev_papers = set(str(r["paper_id"]) for r in dev_recs)
    test_papers = set(str(r["paper_id"]) for r in test_recs)
    
    assert len(train_papers & dev_papers) == 0, "Integrity Failure: Paper leakage between TRAIN and DEV!"
    assert len(train_papers & test_papers) == 0, "Integrity Failure: Paper leakage between TRAIN and TEST!"
    assert len(dev_papers & test_papers) == 0, "Integrity Failure: Paper leakage between DEV and TEST!"
    
    sources = set(r["source_dataset"] for r in records)
    assert "MSVEC" not in sources, "Integrity Failure: MSVEC found in verification dataset!"
    
    print("[INTEGRITY AUDIT] ALL DATA INTEGRITY CHECKS PASSED. Ready for training.")
    print("=" * 65 + "\n")

# -----------------------------------------------------------------------------
# 2. DATASET CLASS & INPUT BUILDER
# -----------------------------------------------------------------------------
class VerificationDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, mode: str = "strict", max_length: int = 384):
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
            ctx_strict, ctx_fallback, _ = prepare_record_contexts(r)
            
            if self.mode == "strict":
                ev = ctx_strict if ctx_strict else "[NO_EVIDENCE_PROVIDED]"
            else: # fallback
                ev = ctx_fallback if ctx_fallback else "[NO_EVIDENCE_PROVIDED]"
                
            input_text = f"Claim:\n{claim}\n\nTitle:\n{title}\n\nEvidence:\n{ev}"
            
            enc = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_length,
                padding=False # dynamically padded in collator
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
# 3. METRIC COMPUTATION
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
# 4. TRAINING & EVALUATION LOOP
# -----------------------------------------------------------------------------
def train_and_evaluate(
    experiment_name: str,
    train_dataset: VerificationDataset,
    dev_dataset: VerificationDataset,
    test_dataset: VerificationDataset,
    tokenizer,
    class_weights: torch.Tensor
) -> Dict[str, Any]:
    print("\n" + "=" * 65)
    print(f"RUNNING EXPERIMENT: {experiment_name}")
    print(f"Mode: {train_dataset.mode.upper()} EVIDENCE")
    print(f"Train samples: {len(train_dataset)}, Dev samples: {len(dev_dataset)}, Test samples: {len(test_dataset)}")
    print("=" * 65)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    
    # Initialize fresh model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID
    )
    device = torch.device("cpu")
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    best_dev_macro_f1 = -1.0
    best_model_weights = None
    training_history = []
    
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()
        model.train()
        total_train_loss = 0.0
        train_preds, train_trues = [], []
        
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item() * len(labels)
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
        print(f"Epoch {epoch}/{EPOCHS} [{ep_duration:.1f}s] -> "
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
    save_dir = os.path.join(TRANSFORMER_DIR, f"best_checkpoint_{train_dataset.mode}")
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
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            
            test_preds.extend([ID_TO_LABEL[p] for p in preds])
            test_trues.extend([ID_TO_LABEL[l] for l in batch["labels"].cpu().tolist()])
            test_probs.extend(probs)
            test_meta.extend(batch["meta"])
            
    # Full Test Metrics
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
    
    print("\n" + "-" * 50)
    print(f"[{experiment_name}] FINAL TEST EVALUATION RESULTS:")
    print(f"Combined Test ($N=265$) -> Acc: {combined_test_metrics['accuracy']:.4f}, Macro-F1: {combined_test_metrics['macro_f1']:.4f}")
    print(f"Natural-Only  ($N=228$) -> Acc: {natural_metrics['accuracy']:.4f}, Macro-F1: {natural_metrics['macro_f1']:.4f} (CON F1: {natural_metrics['per_class']['CONTRADICTED']['f1']:.4f})")
    print(f"Synthetic     ($N=37$)  -> Acc: {synth_metrics['accuracy']:.4f}, Macro-F1: {synth_metrics['macro_f1']:.4f} (CON F1: {synth_metrics['per_class']['CONTRADICTED']['f1']:.4f})")
    print(f"SCitance      ($N=93$)  -> Acc: {scit_metrics['accuracy']:.4f}, Macro-F1: {scit_metrics['macro_f1']:.4f}")
    print(f"SciFact       ($N=172$) -> Acc: {scifact_metrics['accuracy']:.4f}, Macro-F1: {scifact_metrics['macro_f1']:.4f}")
    print("-" * 50)
    
    # Save Predictions CSV
    pred_path = os.path.join(PREDICTIONS_DIR, f"predictions_pubmedbert_{train_dataset.mode}.csv")
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
    print(f"Saved predictions: {pred_path}")
    
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
# 5. FIGURES & REPORTS GENERATION
# -----------------------------------------------------------------------------
def plot_training_curves(history: List[Dict[str, Any]], title: str, filename: str):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    dev_loss = [h["dev_loss"] for h in history]
    dev_macro_f1 = [h["dev_macro_f1"] for h in history]
    dev_acc = [h["dev_acc"] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    # Loss plot
    ax1.plot(epochs, train_loss, 'o-', color='#1f77b4', label='Train Loss', linewidth=2)
    ax1.plot(epochs, dev_loss, 's--', color='#ff7f0e', label='DEV Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Cross-Entropy Loss', fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontweight='bold')
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    # Metrics plot
    ax2.plot(epochs, dev_macro_f1, '^-', color='#2ca02c', label='DEV Macro-F1', linewidth=2)
    ax2.plot(epochs, dev_acc, 'd--', color='#d62728', label='DEV Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('Validation Performance', fontweight='bold')
    ax2.set_xticks(epochs)
    ax2.set_ylim(0.4, 0.9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved figure: {out_path}")

def plot_confusion_matrix(cm_matrix: List[List[int]], title: str, filename: str):
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
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved figure: {out_path}")

# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION CONTROLLER
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("CITATIONGUARD AI — PHASE 5 PUBMEDBERT TRAINING & EVALUATION")
    print("=" * 70)
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    verify_data_integrity(records)
    
    train_recs = [r for r in records if r["guard_split"] == "TRAIN"]
    dev_recs = [r for r in records if r["guard_split"] == "DEV"]
    test_recs = [r for r in records if r["guard_split"] == "TEST"]
    
    # Calculate balanced class weights on TRAIN
    label_counts = Counter(r["verification_label"] for r in train_recs)
    n_train = len(train_recs)
    n_classes = len(LABEL_LIST)
    weights = [n_train / (n_classes * label_counts[l]) for l in LABEL_LIST]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"Calculated TRAIN balanced class weights: SUP={weights[0]:.4f}, CON={weights[1]:.4f}, NEE={weights[2]:.4f}")
    
    # Load Tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # -------------------------------------------------------------------------
    # EXPERIMENT 1: PRIMARY EXPERIMENT — STRICT EVIDENCE
    # -------------------------------------------------------------------------
    train_strict = VerificationDataset(train_recs, tokenizer, mode="strict", max_length=MAX_SEQ_LENGTH)
    dev_strict = VerificationDataset(dev_recs, tokenizer, mode="strict", max_length=MAX_SEQ_LENGTH)
    test_strict = VerificationDataset(test_recs, tokenizer, mode="strict", max_length=MAX_SEQ_LENGTH)
    
    res_strict = train_and_evaluate(
        "Experiment 1: PubMedBERT Strict Evidence",
        train_strict,
        dev_strict,
        test_strict,
        tokenizer,
        class_weights_tensor
    )
    
    plot_training_curves(res_strict["training_history"], "PubMedBERT (Strict Evidence) Training Curves", "training_curves_pubmedbert_strict.png")
    plot_confusion_matrix(res_strict["test_metrics"]["combined"]["confusion_matrix"], "PubMedBERT (Strict Evidence) Test Confusion Matrix", "confusion_matrix_pubmedbert_strict.png")
    
    # -------------------------------------------------------------------------
    # EXPERIMENT 2: ABLATION EXPERIMENT — ABSTRACT FALLBACK
    # -------------------------------------------------------------------------
    train_fallback = VerificationDataset(train_recs, tokenizer, mode="fallback", max_length=MAX_SEQ_LENGTH)
    dev_fallback = VerificationDataset(dev_recs, tokenizer, mode="fallback", max_length=MAX_SEQ_LENGTH)
    test_fallback = VerificationDataset(test_recs, tokenizer, mode="fallback", max_length=MAX_SEQ_LENGTH)
    
    res_fallback = train_and_evaluate(
        "Experiment 2: PubMedBERT Abstract Fallback",
        train_fallback,
        dev_fallback,
        test_fallback,
        tokenizer,
        class_weights_tensor
    )
    
    plot_training_curves(res_fallback["training_history"], "PubMedBERT (Abstract Fallback) Training Curves", "training_curves_pubmedbert_fallback.png")
    plot_confusion_matrix(res_fallback["test_metrics"]["combined"]["confusion_matrix"], "PubMedBERT (Abstract Fallback) Test Confusion Matrix", "confusion_matrix_pubmedbert_fallback.png")
    
    # -------------------------------------------------------------------------
    # EXPORT METRICS & ERROR CSV
    # -------------------------------------------------------------------------
    # Save training_history.csv for primary strict model
    hist_csv = os.path.join(TRANSFORMER_DIR, "training_history.csv")
    with open(hist_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "train_acc", "train_macro_f1", "dev_loss", "dev_acc", "dev_macro_f1",
            "dev_sup_f1", "dev_con_f1", "dev_nee_f1", "epoch_duration_sec"
        ])
        writer.writeheader()
        for h in res_strict["training_history"]:
            writer.writerow(h)
    print(f"Saved {hist_csv}")
    
    # Save training_log.md
    log_md = os.path.join(TRANSFORMER_DIR, "training_log.md")
    with open(log_md, "w", encoding="utf-8") as f:
        f.write("# CitationGuard AI — PubMedBERT Training Log\n\n")
        f.write("## Primary Experiment: Strict Evidence Training Progression\n\n")
        f.write("| Epoch | Train Loss | Train Acc | DEV Loss | DEV Acc | DEV Macro-F1 | SUP F1 | CON F1 | NEE F1 | Duration |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for h in res_strict["training_history"]:
            f.write(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['train_acc']:.4f} | {h['dev_loss']:.4f} | {h['dev_acc']:.4f} | **{h['dev_macro_f1']:.4f}** | {h['dev_sup_f1']:.3f} | {h['dev_con_f1']:.3f} | {h['dev_nee_f1']:.3f} | {h['epoch_duration_sec']:.1f}s |\n")
        f.write("\n## Ablation Experiment: Abstract Fallback Progression\n\n")
        f.write("| Epoch | Train Loss | Train Acc | DEV Loss | DEV Acc | DEV Macro-F1 | SUP F1 | CON F1 | NEE F1 | Duration |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for h in res_fallback["training_history"]:
            f.write(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['train_acc']:.4f} | {h['dev_loss']:.4f} | {h['dev_acc']:.4f} | **{h['dev_macro_f1']:.4f}** | {h['dev_sup_f1']:.3f} | {h['dev_con_f1']:.3f} | {h['dev_nee_f1']:.3f} | {h['epoch_duration_sec']:.1f}s |\n")
    print(f"Saved {log_md}")

    # Save Error Analysis CSV (for primary strict model)
    err_csv = os.path.join(TRANSFORMER_DIR, "transformer_error_analysis.csv")
    with open(err_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "record_id", "source_dataset", "synthetic_or_natural", "true_label", "predicted_label",
            "prob_SUP", "prob_CON", "prob_NEE", "claim_text", "paper_title", "evidence_text"
        ])
        writer.writeheader()
        for e in res_strict["errors"]:
            writer.writerow(e)
    print(f"Saved {err_csv} ({len(res_strict['errors'])} errors)")

    # Save transformer_config.json
    cfg_json = os.path.join(TRANSFORMER_DIR, "transformer_config.json")
    config_dict = {
        "model_name": MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "seed": SEED,
        "class_weights": weights,
        "primary_experiment_strict": {
            "best_dev_macro_f1": res_strict["best_dev_macro_f1"],
            "test_accuracy": res_strict["test_metrics"]["combined"]["accuracy"],
            "test_macro_f1": res_strict["test_metrics"]["combined"]["macro_f1"],
            "test_weighted_f1": res_strict["test_metrics"]["combined"]["weighted_f1"],
            "natural_macro_f1": res_strict["test_metrics"]["natural"]["macro_f1"],
            "synthetic_macro_f1": res_strict["test_metrics"]["synthetic"]["macro_f1"]
        },
        "ablation_experiment_fallback": {
            "best_dev_macro_f1": res_fallback["best_dev_macro_f1"],
            "test_accuracy": res_fallback["test_metrics"]["combined"]["accuracy"],
            "test_macro_f1": res_fallback["test_metrics"]["combined"]["macro_f1"],
            "test_weighted_f1": res_fallback["test_metrics"]["combined"]["weighted_f1"],
            "natural_macro_f1": res_fallback["test_metrics"]["natural"]["macro_f1"],
            "synthetic_macro_f1": res_fallback["test_metrics"]["synthetic"]["macro_f1"]
        }
    }
    with open(cfg_json, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved {cfg_json}")
    
    # Save transformer_environment.txt
    env_txt = os.path.join(TRANSFORMER_DIR, "transformer_environment.txt")
    with open(env_txt, "w", encoding="utf-8") as f:
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        f.write(f"CPU Threads Allocated: {CPU_THREADS}\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"Transformers Version: {AutoModelForSequenceClassification.__module__}\n")
    print(f"Saved {env_txt}")
    
    # Save master cache json
    master_cache = os.path.join(TRANSFORMER_DIR, "phase5_master_results.json")
    with open(master_cache, "w", encoding="utf-8") as f:
        json.dump({"strict": res_strict, "fallback": res_fallback}, f, indent=2)
    print(f"Saved {master_cache}")

    print("\n" + "=" * 70)
    print("PHASE 5 PUBMEDBERT TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
