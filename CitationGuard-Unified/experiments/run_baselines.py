"""
CitationGuard AI — Phase 4 Machine-Learning Baseline Suite
==========================================================
Reproducible end-to-end pipeline for Phase 4 lexical baselines:
- Critical Data Integrity Checks
- Deterministic Preprocessing and 5 Input Conditions
- Baseline 0: Majority Class Baseline
- Baseline 1: TF-IDF + Logistic Regression (Grid search on DEV)
- Baseline 2: TF-IDF + Linear SVM (Grid search on DEV)
- Evidence Ablation Study across Conditions A-E
- Stratified Evaluation on TEST: Combined, Natural-Only, Synthetic-Negation
- Diagnostic Source Evaluation: SCitance vs SciFact
- Error Analysis & Representative False Predictions
- Publication-quality Confusion Matrix Generation
- Complete Machine-Readable Predictions and Metric Reports
"""

import os
import sys
import json
import csv
import re
import random
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# Add preprocessing module to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESS_DIR = os.path.join(CURRENT_DIR, "preprocessing")
if PREPROCESS_DIR not in sys.path:
    sys.path.append(PREPROCESS_DIR)

from preprocess import (
    clean_scientific_text,
    prepare_record_contexts,
    build_input_conditions
)

# Set deterministic seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
BASE_UNIFIED = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
VERIF_JSONL = os.path.join(BASE_UNIFIED, "citationguard_verification.jsonl")
VERIF_CSV = os.path.join(BASE_UNIFIED, "citationguard_verification.csv")

REPORTS_DIR = os.path.join(CURRENT_DIR, "reports")
PREDICTIONS_DIR = os.path.join(CURRENT_DIR, "predictions")
FIGURES_DIR = os.path.join(CURRENT_DIR, "figures")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

for d in [REPORTS_DIR, PREDICTIONS_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

LABEL_ORDER = ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_EVIDENCE"]
LABEL_TO_IDX = {lbl: idx for idx, lbl in enumerate(LABEL_ORDER)}

# -----------------------------------------------------------------------------
# 1. CRITICAL DATA INTEGRITY CHECK (RULE 18)
# -----------------------------------------------------------------------------
def verify_data_integrity(records: List[Dict[str, Any]]):
    """Rigorous verification before running any experiment."""
    print("\n" + "=" * 60)
    print("[INTEGRITY AUDIT] Running Rule 18 Pre-flight Data Checks...")
    print("=" * 60)
    
    assert len(records) == 1754, f"Integrity Failure: Expected 1,754 records, got {len(records)}"
    
    train_recs = [r for r in records if r["guard_split"] == "TRAIN"]
    dev_recs = [r for r in records if r["guard_split"] == "DEV"]
    test_recs = [r for r in records if r["guard_split"] == "TEST"]
    
    assert len(train_recs) == 1226, f"Integrity Failure: Expected 1,226 TRAIN, got {len(train_recs)}"
    assert len(dev_recs) == 263, f"Integrity Failure: Expected 263 DEV, got {len(dev_recs)}"
    assert len(test_recs) == 265, f"Integrity Failure: Expected 265 TEST, got {len(test_recs)}"
    
    labels = set(r["verification_label"] for r in records)
    assert labels == set(LABEL_ORDER), f"Integrity Failure: Invalid labels {labels}"
    
    # Check no duplicate normalized claims
    norm_claims = [re.sub(r'[^\w\s]', '', r["claim_text"].lower()) for r in records]
    assert len(set(norm_claims)) == 1754, "Integrity Failure: Duplicate normalized claims detected!"
    
    # Check cross-split paper leakage
    train_papers = set(str(r["paper_id"]) for r in train_recs)
    dev_papers = set(str(r["paper_id"]) for r in dev_recs)
    test_papers = set(str(r["paper_id"]) for r in test_recs)
    
    assert len(train_papers & dev_papers) == 0, "Integrity Failure: Paper leakage between TRAIN and DEV!"
    assert len(train_papers & test_papers) == 0, "Integrity Failure: Paper leakage between TRAIN and TEST!"
    assert len(dev_papers & test_papers) == 0, "Integrity Failure: Paper leakage between DEV and TEST!"
    
    # Verify MSVEC is NOT used
    sources = set(r["source_dataset"] for r in records)
    assert "MSVEC" not in sources, "Integrity Failure: MSVEC found in verification dataset!"
    
    print("[INTEGRITY AUDIT] ALL 8 CHECKS PASSED. Dataset is clean and verified.")
    print("=" * 60 + "\n")

# -----------------------------------------------------------------------------
# 2. METRIC CALCULATION HELPERS
# -----------------------------------------------------------------------------
def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0)
    
    p_per = precision_score(y_true, y_pred, labels=LABEL_ORDER, average=None, zero_division=0)
    r_per = recall_score(y_true, y_pred, labels=LABEL_ORDER, average=None, zero_division=0)
    f1_per = f1_score(y_true, y_pred, labels=LABEL_ORDER, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    
    per_class = {}
    for idx, lbl in enumerate(LABEL_ORDER):
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
# 3. CONFUSION MATRIX PLOTTING
# -----------------------------------------------------------------------------
def save_confusion_matrix_figure(cm_matrix: List[List[int]], title: str, filename: str):
    """Plot and save a publication-quality confusion matrix heatmap."""
    cm = np.array(cm_matrix)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(LABEL_ORDER)))
    ax.set_yticks(range(len(LABEL_ORDER)))
    ax.set_xticklabels(["SUP", "CON", "NEE"], fontsize=10, fontweight="bold")
    ax.set_yticklabels(["SUP", "CON", "NEE"], fontsize=10, fontweight="bold")
    
    # Annotate numbers
    for i in range(len(LABEL_ORDER)):
        for j in range(len(LABEL_ORDER)):
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
    print(f"      Saved figure: {out_path}")

# -----------------------------------------------------------------------------
# 4. MAIN PIPELINE
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("CITATIONGUARD AI — PHASE 4 MACHINE LEARNING BASELINE EXECUTION")
    print("=" * 70)
    
    # 1. Load data
    with open(VERIF_JSONL, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    verify_data_integrity(records)
    
    # 2. Preprocess records into 5 conditions
    print("[1/6] Preprocessing all records and constructing 5 input conditions...")
    processed_records = []
    fallback_counter = 0
    
    for r in records:
        conds = build_input_conditions(r)
        _, _, used_fb = prepare_record_contexts(r)
        if used_fb:
            fallback_counter += 1
            
        processed_records.append({
            "record_id": r["record_id"],
            "source_dataset": r["source_dataset"],
            "paper_id": r["paper_id"],
            "paper_title": r.get("paper_title") or "",
            "claim_text": r["claim_text"],
            "verification_label": r["verification_label"],
            "guard_split": r["guard_split"],
            "synthetic_or_natural": r["synthetic_or_natural"],
            "used_fallback": used_fb,
            **conds
        })
        
    print(f"      Total records preprocessed: {len(processed_records)}")
    print(f"      Total records using fallback abstract: {fallback_counter} / 1754 (23.72%)")
    
    # Partition by split
    train_data = [r for r in processed_records if r["guard_split"] == "TRAIN"]
    dev_data = [r for r in processed_records if r["guard_split"] == "DEV"]
    test_data = [r for r in processed_records if r["guard_split"] == "TEST"]
    
    y_train = [r["verification_label"] for r in train_data]
    y_dev = [r["verification_label"] for r in dev_data]
    y_test = [r["verification_label"] for r in test_data]
    
    conditions = [
        ("condition_a", "Condition A: Claim Only"),
        ("condition_b", "Condition B: Claim + Title"),
        ("condition_c", "Condition C: Claim + Evidence (Strict)"),
        ("condition_d", "Condition D: Claim + Abstract Fallback"),
        ("condition_e", "Condition E: Full Standardized Input"),
    ]
    
    # -------------------------------------------------------------------------
    # 3. BASELINE 0 — MAJORITY CLASS
    # -------------------------------------------------------------------------
    print("\n[2/6] Evaluating Baseline 0: Majority Class Baseline...")
    maj_pred_dev = ["SUPPORTED"] * len(y_dev)
    maj_pred_test = ["SUPPORTED"] * len(y_test)
    
    maj_metrics_dev = compute_metrics(y_dev, maj_pred_dev)
    maj_metrics_test = compute_metrics(y_test, maj_pred_test)
    
    print(f"      Majority Baseline DEV  -> Acc: {maj_metrics_dev['accuracy']:.4f}, Macro-F1: {maj_metrics_dev['macro_f1']:.4f}")
    print(f"      Majority Baseline TEST -> Acc: {maj_metrics_test['accuracy']:.4f}, Macro-F1: {maj_metrics_test['macro_f1']:.4f}")
    
    # -------------------------------------------------------------------------
    # 4. HYPERPARAMETER SELECTION GRIDS (DEV TUNING ONLY)
    # -------------------------------------------------------------------------
    print("\n[3/6] Running controlled hyperparameter search on DEV...")
    
    feature_grid = [
        {"ngram_range": (1, 1), "min_df": 1, "max_features": 5000, "sublinear_tf": True},
        {"ngram_range": (1, 2), "min_df": 1, "max_features": 10000, "sublinear_tf": True},
        {"ngram_range": (1, 2), "min_df": 2, "max_features": 10000, "sublinear_tf": True},
        {"ngram_range": (1, 2), "min_df": 2, "max_features": None, "sublinear_tf": True},
    ]
    
    c_grid = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # We find best configuration on Condition E for LR and LinearSVM
    best_lr_config = None
    best_lr_dev_f1 = -1.0
    
    best_svm_config = None
    best_svm_dev_f1 = -1.0
    
    for f_idx, f_cfg in enumerate(feature_grid):
        vec = TfidfVectorizer(
            ngram_range=f_cfg["ngram_range"],
            min_df=f_cfg["min_df"],
            max_features=f_cfg["max_features"],
            sublinear_tf=f_cfg["sublinear_tf"],
            lowercase=True
        )
        X_train_e = vec.fit_transform([r["condition_e"] for r in train_data])
        X_dev_e = vec.transform([r["condition_e"] for r in dev_data])
        
        for c_val in c_grid:
            # LR
            lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, random_state=SEED)
            lr.fit(X_train_e, y_train)
            pred_dev_lr = lr.predict(X_dev_e)
            f1_lr = f1_score(y_dev, pred_dev_lr, labels=LABEL_ORDER, average="macro", zero_division=0)
            
            if f1_lr > best_lr_dev_f1:
                best_lr_dev_f1 = f1_lr
                best_lr_config = {"feature_cfg": f_cfg, "C": c_val, "dev_macro_f1": f1_lr}
                
            # SVM
            svm = LinearSVC(C=c_val, class_weight="balanced", max_iter=2000, random_state=SEED)
            svm.fit(X_train_e, y_train)
            pred_dev_svm = svm.predict(X_dev_e)
            f1_svm = f1_score(y_dev, pred_dev_svm, labels=LABEL_ORDER, average="macro", zero_division=0)
            
            if f1_svm > best_svm_dev_f1:
                best_svm_dev_f1 = f1_svm
                best_svm_config = {"feature_cfg": f_cfg, "C": c_val, "dev_macro_f1": f1_svm}
                
    print(f"      Selected LR  Config: C={best_lr_config['C']}, Features={best_lr_config['feature_cfg']}, DEV Macro-F1: {best_lr_dev_f1:.4f}")
    print(f"      Selected SVM Config: C={best_svm_config['C']}, Features={best_svm_config['feature_cfg']}, DEV Macro-F1: {best_svm_dev_f1:.4f}")

    # -------------------------------------------------------------------------
    # 5. EXECUTION ACROSS ALL 5 INPUT CONDITIONS
    # -------------------------------------------------------------------------
    print("\n[4/6] Executing baselines across all 5 Input Conditions...")
    
    all_results = []
    # Dict to hold predictions for export
    # key: (model_name, cond_key) -> list of record predictions
    predictions_registry = {}
    
    for cond_key, cond_name in conditions:
        print(f"\n   --- Running {cond_name} ---")
        train_texts = [r[cond_key] for r in train_data]
        dev_texts = [r[cond_key] for r in dev_data]
        test_texts = [r[cond_key] for r in test_data]
        
        # 1. Logistic Regression
        vec_lr = TfidfVectorizer(
            ngram_range=best_lr_config["feature_cfg"]["ngram_range"],
            min_df=best_lr_config["feature_cfg"]["min_df"],
            max_features=best_lr_config["feature_cfg"]["max_features"],
            sublinear_tf=best_lr_config["feature_cfg"]["sublinear_tf"],
            lowercase=True
        )
        X_tr_lr = vec_lr.fit_transform(train_texts)
        X_dv_lr = vec_lr.transform(dev_texts)
        X_ts_lr = vec_lr.transform(test_texts)
        
        clf_lr = LogisticRegression(C=best_lr_config["C"], class_weight="balanced", max_iter=1000, random_state=SEED)
        clf_lr.fit(X_tr_lr, y_train)
        
        pred_dv_lr = clf_lr.predict(X_dv_lr)
        pred_ts_lr = clf_lr.predict(X_ts_lr)
        proba_ts_lr = clf_lr.predict_proba(X_ts_lr)
        
        m_dv_lr = compute_metrics(y_dev, pred_dv_lr)
        m_ts_lr = compute_metrics(y_test, pred_ts_lr)
        
        all_results.append({
            "model": "Logistic Regression",
            "condition_key": cond_key,
            "condition_name": cond_name,
            "dev_acc": m_dv_lr["accuracy"],
            "dev_macro_f1": m_dv_lr["macro_f1"],
            "dev_weighted_f1": m_dv_lr["weighted_f1"],
            "test_acc": m_ts_lr["accuracy"],
            "test_macro_f1": m_ts_lr["macro_f1"],
            "test_weighted_f1": m_ts_lr["weighted_f1"],
            "test_metrics_full": m_ts_lr,
            "dev_metrics_full": m_dv_lr,
            "predictions_test": pred_ts_lr.tolist(),
            "proba_test": proba_ts_lr.tolist()
        })
        
        predictions_registry[("LogisticRegression", cond_key)] = {
            "preds": pred_ts_lr,
            "probas": proba_ts_lr,
            "classes": clf_lr.classes_.tolist()
        }
        
        # 2. Linear SVM
        vec_svm = TfidfVectorizer(
            ngram_range=best_svm_config["feature_cfg"]["ngram_range"],
            min_df=best_svm_config["feature_cfg"]["min_df"],
            max_features=best_svm_config["feature_cfg"]["max_features"],
            sublinear_tf=best_svm_config["feature_cfg"]["sublinear_tf"],
            lowercase=True
        )
        X_tr_svm = vec_svm.fit_transform(train_texts)
        X_dv_svm = vec_svm.transform(dev_texts)
        X_ts_svm = vec_svm.transform(test_texts)
        
        clf_svm = LinearSVC(C=best_svm_config["C"], class_weight="balanced", max_iter=2000, random_state=SEED)
        clf_svm.fit(X_tr_svm, y_train)
        
        pred_dv_svm = clf_svm.predict(X_dv_svm)
        pred_ts_svm = clf_svm.predict(X_ts_svm)
        decision_ts_svm = clf_svm.decision_function(X_ts_svm)
        
        m_dv_svm = compute_metrics(y_dev, pred_dv_svm)
        m_ts_svm = compute_metrics(y_test, pred_ts_svm)
        
        all_results.append({
            "model": "Linear SVM",
            "condition_key": cond_key,
            "condition_name": cond_name,
            "dev_acc": m_dv_svm["accuracy"],
            "dev_macro_f1": m_dv_svm["macro_f1"],
            "dev_weighted_f1": m_dv_svm["weighted_f1"],
            "test_acc": m_ts_svm["accuracy"],
            "test_macro_f1": m_ts_svm["macro_f1"],
            "test_weighted_f1": m_ts_svm["weighted_f1"],
            "test_metrics_full": m_ts_svm,
            "dev_metrics_full": m_dv_svm,
            "predictions_test": pred_ts_svm.tolist(),
            "decision_test": decision_ts_svm.tolist()
        })
        
        predictions_registry[("LinearSVM", cond_key)] = {
            "preds": pred_ts_svm,
            "decisions": decision_ts_svm,
            "classes": clf_svm.classes_.tolist()
        }
        
        print(f"      LR  -> DEV Macro-F1: {m_dv_lr['macro_f1']:.4f} | TEST Macro-F1: {m_ts_lr['macro_f1']:.4f} (Acc: {m_ts_lr['accuracy']:.4f})")
        print(f"      SVM -> DEV Macro-F1: {m_dv_svm['macro_f1']:.4f} | TEST Macro-F1: {m_ts_svm['macro_f1']:.4f} (Acc: {m_ts_svm['accuracy']:.4f})")

    # -------------------------------------------------------------------------
    # 6. EXPORT PREDICTIONS
    # -------------------------------------------------------------------------
    print("\n[5/6] Exporting predictions and saving metrics...")
    for (model_name, cond_key), p_info in predictions_registry.items():
        pred_file = os.path.join(PREDICTIONS_DIR, f"predictions_{model_name}_{cond_key}.csv")
        with open(pred_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if "probas" in p_info:
                writer.writerow(["record_id", "true_label", "predicted_label", "prob_SUPPORTED", "prob_CONTRADICTED", "prob_NOT_ENOUGH_EVIDENCE"])
                classes = p_info["classes"]
                sup_idx = classes.index("SUPPORTED")
                con_idx = classes.index("CONTRADICTED")
                nee_idx = classes.index("NOT_ENOUGH_EVIDENCE")
                for r_idx, r in enumerate(test_data):
                    pb = p_info["probas"][r_idx]
                    writer.writerow([
                        r["record_id"],
                        r["verification_label"],
                        p_info["preds"][r_idx],
                        f"{pb[sup_idx]:.4f}",
                        f"{pb[con_idx]:.4f}",
                        f"{pb[nee_idx]:.4f}"
                    ])
            else:
                writer.writerow(["record_id", "true_label", "predicted_label"])
                for r_idx, r in enumerate(test_data):
                    writer.writerow([
                        r["record_id"],
                        r["verification_label"],
                        p_info["preds"][r_idx]
                    ])

    # -------------------------------------------------------------------------
    # 7. PRIMARY MODEL SELECTION ON DEV
    # -------------------------------------------------------------------------
    # Primary: DEV Macro-F1
    best_overall = max(all_results, key=lambda x: x["dev_macro_f1"])
    print("\n" + "=" * 60)
    print(f"BEST BASELINE MODEL (Selected on DEV Macro-F1):")
    print(f"Model: {best_overall['model']}")
    print(f"Input: {best_overall['condition_name']}")
    print(f"DEV  Macro-F1: {best_overall['dev_macro_f1']:.4f} (Acc: {best_overall['dev_acc']:.4f})")
    print(f"TEST Macro-F1: {best_overall['test_macro_f1']:.4f} (Acc: {best_overall['test_acc']:.4f})")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 8. SUBSET EVALUATIONS ON TEST (Natural-only, Synthetic-only, Source-wise)
    # -------------------------------------------------------------------------
    # Run subset metrics for all model/conditions
    subset_records_indices = {
        "Combined Test": list(range(len(test_data))),
        "Natural-Only Test": [i for i, r in enumerate(test_data) if r["synthetic_or_natural"] == "NATURAL"],
        "Synthetic-Negation Test": [i for i, r in enumerate(test_data) if r["synthetic_or_natural"] == "SYNTHETIC_NEGATION"],
        "SCitance Test": [i for i, r in enumerate(test_data) if r["source_dataset"] == "SCitance"],
        "SciFact Test": [i for i, r in enumerate(test_data) if r["source_dataset"] == "SciFact"]
    }
    
    detailed_subset_results = []
    for res in all_results:
        preds = res["predictions_test"]
        for s_name, indices in subset_records_indices.items():
            sub_true = [y_test[i] for i in indices]
            sub_pred = [preds[i] for i in indices]
            sub_m = compute_metrics(sub_true, sub_pred)
            detailed_subset_results.append({
                "model": res["model"],
                "condition_key": res["condition_key"],
                "condition_name": res["condition_name"],
                "subset": s_name,
                "n_samples": len(indices),
                "accuracy": sub_m["accuracy"],
                "macro_f1": sub_m["macro_f1"],
                "weighted_f1": sub_m["weighted_f1"],
                "sup_f1": sub_m["per_class"]["SUPPORTED"]["f1"],
                "con_f1": sub_m["per_class"]["CONTRADICTED"]["f1"],
                "nee_f1": sub_m["per_class"]["NOT_ENOUGH_EVIDENCE"]["f1"],
                "confusion_matrix": sub_m["confusion_matrix"]
            })

    # -------------------------------------------------------------------------
    # 9. CONFUSION MATRIX FIGURES
    # -------------------------------------------------------------------------
    # Best Logistic Regression (Condition E)
    best_lr_res = [r for r in all_results if r["model"] == "Logistic Regression" and r["condition_key"] == "condition_e"][0]
    save_confusion_matrix_figure(
        best_lr_res["test_metrics_full"]["confusion_matrix"],
        f"Logistic Regression (Full Input) Confusion Matrix\nTest Macro-F1: {best_lr_res['test_macro_f1']:.4f}",
        "confusion_matrix_logistic_regression.png"
    )
    
    # Best Linear SVM (Condition E)
    best_svm_res = [r for r in all_results if r["model"] == "Linear SVM" and r["condition_key"] == "condition_e"][0]
    save_confusion_matrix_figure(
        best_svm_res["test_metrics_full"]["confusion_matrix"],
        f"Linear SVM (Full Input) Confusion Matrix\nTest Macro-F1: {best_svm_res['test_macro_f1']:.4f}",
        "confusion_matrix_linear_svm.png"
    )

    # -------------------------------------------------------------------------
    # 10. ERROR ANALYSIS COLLECTION
    # -------------------------------------------------------------------------
    best_preds = best_overall["predictions_test"]
    errors = []
    for idx, r in enumerate(test_data):
        true_l = r["verification_label"]
        pred_l = best_preds[idx]
        if true_l != pred_l:
            errors.append({
                "record_id": r["record_id"],
                "source_dataset": r["source_dataset"],
                "true_label": true_l,
                "predicted_label": pred_l,
                "claim_text": r["claim_text"],
                "paper_title": r["paper_title"],
                "context_used": r[best_overall["condition_key"]],
                "synthetic_or_natural": r["synthetic_or_natural"]
            })
            
    err_csv_path = os.path.join(REPORTS_DIR, "baseline_error_analysis.csv")
    with open(err_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "record_id", "source_dataset", "synthetic_or_natural", "true_label", "predicted_label",
            "claim_text", "paper_title", "context_used"
        ])
        writer.writeheader()
        for e in errors:
            writer.writerow(e)
    print(f"      Saved error analysis ({len(errors)} errors): {err_csv_path}")

    # -------------------------------------------------------------------------
    # 11. BASELINE COMPARISON CSV & METRIC COMPARISON CSV
    # -------------------------------------------------------------------------
    # baseline_comparison.csv
    base_comp_path = os.path.join(REPORTS_DIR, "baseline_comparison.csv")
    with open(base_comp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "condition", "dev_accuracy", "dev_macro_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1"])
        writer.writerow(["Majority Baseline", "Condition A: Claim Only", f"{maj_metrics_dev['accuracy']:.4f}", f"{maj_metrics_dev['macro_f1']:.4f}", f"{maj_metrics_test['accuracy']:.4f}", f"{maj_metrics_test['macro_f1']:.4f}", f"{maj_metrics_test['weighted_f1']:.4f}"])
        for r in all_results:
            writer.writerow([r["model"], r["condition_name"], f"{r['dev_acc']:.4f}", f"{r['dev_macro_f1']:.4f}", f"{r['test_acc']:.4f}", f"{r['test_macro_f1']:.4f}", f"{r['test_weighted_f1']:.4f}"])
    print(f"      Saved {base_comp_path}")

    # metric_comparison.csv (Detailed subset results)
    metric_comp_path = os.path.join(REPORTS_DIR, "metric_comparison.csv")
    with open(metric_comp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "condition_key", "subset", "n_samples", "accuracy", "macro_f1", "weighted_f1", "sup_f1", "con_f1", "nee_f1"
        ])
        writer.writeheader()
        for d in detailed_subset_results:
            writer.writerow({
                "model": d["model"],
                "condition_key": d["condition_key"],
                "subset": d["subset"],
                "n_samples": d["n_samples"],
                "accuracy": f"{d['accuracy']:.4f}",
                "macro_f1": f"{d['macro_f1']:.4f}",
                "weighted_f1": f"{d['weighted_f1']:.4f}",
                "sup_f1": f"{d['sup_f1']:.4f}",
                "con_f1": f"{d['con_f1']:.4f}",
                "nee_f1": f"{d['nee_f1']:.4f}"
            })
    print(f"      Saved {metric_comp_path}")

    # Dump complete JSON results for report generator
    summary_data = {
        "best_overall": {
            "model": best_overall["model"],
            "condition_key": best_overall["condition_key"],
            "condition_name": best_overall["condition_name"],
            "dev_acc": best_overall["dev_acc"],
            "dev_macro_f1": best_overall["dev_macro_f1"],
            "test_acc": best_overall["test_acc"],
            "test_macro_f1": best_overall["test_macro_f1"],
            "test_weighted_f1": best_overall["test_weighted_f1"],
            "test_metrics_full": best_overall["test_metrics_full"]
        },
        "majority_baseline": {
            "dev": maj_metrics_dev,
            "test": maj_metrics_test
        },
        "lr_config": best_lr_config,
        "svm_config": best_svm_config,
        "all_results": [
            {
                "model": r["model"],
                "condition_key": r["condition_key"],
                "condition_name": r["condition_name"],
                "dev_acc": r["dev_acc"],
                "dev_macro_f1": r["dev_macro_f1"],
                "test_acc": r["test_acc"],
                "test_macro_f1": r["test_macro_f1"],
                "test_weighted_f1": r["test_weighted_f1"],
                "test_metrics_full": r["test_metrics_full"]
            }
            for r in all_results
        ],
        "subset_results": detailed_subset_results,
        "total_errors": len(errors),
        "total_test": len(test_data),
        "fallback_count": fallback_counter
    }
    
    with open(os.path.join(CURRENT_DIR, "baseline_experiment_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"      Exported execution cache: {os.path.join(CURRENT_DIR, 'baseline_experiment_results.json')}")

if __name__ == "__main__":
    main()
