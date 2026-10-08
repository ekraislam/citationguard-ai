# CitationGuard AI — Phase 5 Google Colab GPU Package

This package provides a clean, self-contained, reproducible training and evaluation suite for fine-tuning **PubMedBERT** (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`) on the CitationGuard AI scientific citation verification benchmark using Google Colab GPUs (NVIDIA T4 / V100 / A100).

---

## 1. Package Structure

```
experiments/transformer/colab/
├── README.md                          # This instruction manual
├── requirements.txt                   # Minimal Python dependencies for Colab
├── transformer_config_colab.json      # Complete Phase 5 configuration & references
├── train_pubmedbert_colab.py          # Standalone GPU training & multi-subset evaluation script
└── phase5_pubmedbert_colab.ipynb      # Interactive Google Colab notebook
```

---

## 2. Quickstart Instructions for Google Colab

### Option A: Interactive Notebook (Recommended)
1. Open Google Colab: [https://colab.research.google.com/](https://colab.research.google.com/)
2. Click **Upload** and upload `phase5_pubmedbert_colab.ipynb`.
3. In Colab, go to **Runtime** → **Change runtime type** → select **T4 GPU** (or V100/A100).
4. Run Step 1 (Environment check). The notebook verifies CUDA and reports your GPU accelerator.
5. Upload `citationguard_verification.jsonl` directly via the notebook widget or specify a Google Drive path.
6. Upload `train_pubmedbert_colab.py` to the Colab working directory (`/content/`).
7. Execute Step 5 to run the complete training pipeline.
8. Inspect inline training curves, confusion matrices, and the generated research report.

### Option B: Command Line in Colab Terminal / Cell
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run end-to-end training and evaluation
python train_pubmedbert_colab.py \
    --data_path ./data/citationguard_verification.jsonl \
    --output_dir ./phase5_output \
    --epochs 3 \
    --batch_size 16 \
    --lr 2.5e-5 \
    --max_length 384
```

---

## 3. Hardware & Acceleration Profile
- **Mixed Precision**: Enabled (`torch.cuda.amp.autocast()` + `GradScaler` in FP16).
- **Throughput**: On an NVIDIA T4 GPU, training requires **~35–45 seconds per epoch** (compared to ~41.5 minutes per epoch on laptop CPU), providing a **~60× speedup**.
- **Memory Safety**: Default batch size is 16 with max sequence length 384 tokens (comfortably fitting in 15 GB T4 VRAM). If using a smaller GPU, set `--batch_size 8 --grad_accum 2`.

---

## 4. Experimental Conditions & Integrity Rules

### Data Integrity (Rule 19)
The script enforces a strict pre-flight audit before allocating tensors:
- Total records: **1,754**
- Splits: **1,226 TRAIN** (70%), **263 DEV** (15%), **265 TEST** (15%)
- Labels: `SUPPORTED`, `CONTRADICTED`, `NOT_ENOUGH_EVIDENCE`
- Contamination check: Zero paper leakage across splits; MSVEC excluded.

### Experiment 1: Primary Strict Evidence
- Input representation:
  ```
  Claim:
  {claim_text}

  Title:
  {paper_title}

  Evidence:
  {evidence_text}
  ```
- Missing evidence (416 SciFact records) is represented explicitly as `[NO_EVIDENCE_PROVIDED]`. No silent substitution.

### Experiment 2: Ablation Abstract Fallback
- Input representation:
  ```
  Claim:
  {claim_text}

  Title:
  {paper_title}

  Evidence:
  {evidence_text if available else paper_abstract}
  ```

---

## 5. Primary Baseline Reference (Phase 4)

All Colab results must be benchmarked against the strongest Phase 4 baseline:
- **Model**: TF-IDF (10,000 sublinear n-grams) + Logistic Regression
- **Input Condition**: Claim + Title + Strict Evidence
- **DEV Macro-F1**: `0.6625`
- **TEST Macro-F1**: `0.6754`
- **TEST Accuracy**: `0.6717`
- **TEST Weighted-F1**: `0.6765`

> [!IMPORTANT]
> All score deltas must be reported in **percentage points** (e.g. $+3.70$ percentage points), never ambiguous relative percentages.

---

## 6. Generated Output Artifacts

All outputs will be saved to `--output_dir` (`./phase5_output/`):
- `best_checkpoint_strict/`: Safetensors weights, model configuration, tokenizer.
- `best_checkpoint_fallback/`: Fallback ablation weights and tokenizer.
- `predictions/`: `predictions_pubmedbert_strict.csv` and `predictions_pubmedbert_fallback.csv`.
- `figures/`:
  - `training_curves_pubmedbert_strict.png`
  - `confusion_matrix_pubmedbert_strict.png`
  - `training_curves_pubmedbert_fallback.png`
  - `confusion_matrix_pubmedbert_fallback.png`
- `reports/`:
  - `transformer_source_performance.md` (SCitance vs SciFact breakdown)
  - `PHASE5_PUBMEDBERT_REPORT.md` (Comprehensive Phase 5 report answering all 7 scientific questions)
- `transformer_error_analysis.csv`: Detailed categorization of all test errors.
- `training_history.csv`: Per-epoch loss and validation metrics.
