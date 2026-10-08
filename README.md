<div align="center">

# CitationGuard AI

### Evidence-Aware Scientific Citation Verification & Peer-Review Audit Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers 4.30+](https://img.shields.io/badge/🤗_Transformers-4.30+-ffd21e.svg)](https://huggingface.co/)
[![License: All Rights Reserved](https://img.shields.io/badge/License-All_Rights_Reserved-crimson.svg)](LICENSE)
[![Model: PubMedBERT](https://img.shields.io/badge/Model-BiomedNLP--PubMedBERT-0284c7.svg)](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)

*A scholarly computational system designed to detect unsupported academic claims, evaluate empirical grounding, and eliminate citation hallucinations in scientific literature.*

[Live Demo](#quickstart) • [Key Features](#key-features) • [Architecture](#model-architecture) • [Dataset Protocol](#unified-dataset-card) • [API Reference](#rest-api-documentation) • [Citation](#citation)

</div>

---

## 📖 Overview

In modern scientific research, **citation integrity** is paramount. However, academic literature frequently suffers from:
1. **Citation Amnesia & Misattribution:** Papers cited for propositions they never investigated.
2. **Unsupported Generalizations:** Assertions made without empirical backing in the cited excerpt.
3. **Ghost References & LLM Hallucinations:** Artificial claims fabricated by generative AI tools.

**CitationGuard AI** provides an evidence-aware semantic verification engine fine-tuned on biomedical literature (`BiomedNLP-PubMedBERT-base-uncased-abstract`). Given an academic **Claim**, a **Cited Paper Title**, and an **Evidence Excerpt**, the engine performs calibrated 3-class stance classification under a **Strict Context Isolation Protocol**:

- **SUPPORTED (`+`):** The cited literature excerpt provides sufficient empirical grounding to substantiate the claim.
- **CONTRADICTED (`−`):** The cited literature excerpt presents empirical findings that directly refute the claim.
- **NOT ENOUGH EVIDENCE (`?`):** The cited excerpt does not contain sufficient empirical context to affirm or dispute the proposition.

---

## ✨ Key Features

- **🔬 Production-Grade PubMedBERT Inference Engine:** Standalone, deterministic inference pipeline with CPU/CUDA fallback, sub-50ms latency, and calibrated softmax probabilities.
- **🛡️ Strict Context Isolation:** Eliminates prior leakage and spurious lexical correlations by enforcing explicit input delimiters:
  ```
  Claim:\n{claim}\n\nTitle:\n{title}\n\nEvidence:\n{evidence or '[NO_EVIDENCE_PROVIDED]'}
  ```
- **📜 Editorial Scholarly Workbench:** Claude-inspired warm ivory research dashboard designed with tactile micro-interactions, dark/light theme switching, and preset benchmark study cases.
- **🖨️ Strictly 1-Page PDF Audit Certificate:** Single-click generation of institutional-grade, peer-review audit certificates formatted to fit strictly onto **1 page (A4)** for peer-review attachments and archival dossiers.
- **📋 Markdown Audit Brief Export:** Instantly copy formatted scientific audit memos to the system clipboard for inclusion in review comments or lab notebooks.
- **📊 Unified & Audited Dataset Pipeline:** Complete multi-corpus dataset curation pipeline integrating **SciFact**, **SCitance**, **MSVEC**, **SciCite**, and **SciClaim** with rigorous leak checks and cross-split deduplication.

---

## 🏛️ Model Architecture

| Parameter | Specification |
| :--- | :--- |
| **Base Architecture** | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` |
| **Model Type** | 12-layer, 768-hidden, 12-heads Transformer (~110M params) |
| **Classification Head** | Linear Stance Head (`Linear(768, 3)`) with Softmax Normalization |
| **Context Window** | 512 sequence tokens |
| **Evidence Protocol** | Segment-level boundary isolation with `[NO_EVIDENCE_PROVIDED]` baseline handling |
| **Inference Latency** | ~43 ms on Intel/AMD 8-core CPU; <10 ms on NVIDIA GPU |

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/ekraislam/citationguard-ai.git
cd citationguard-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Model Weights Setup
Place the fine-tuned PubMedBERT weights (`model.safetensors`, 417.67 MB) into:
```
models/best_checkpoint_strict/
├── config.json
├── model.safetensors        # Download from GitHub Releases / Hugging Face
├── tokenizer.json
└── tokenizer_config.json
```
*(If weights are not found locally, the inference engine falls back gracefully with a clear setup notice).*

### 4. Launch the Web Application
```bash
python run_app.py --port 5000
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## 📁 Repository Structure

```
citationguard-ai/
├── run_app.py                          # Application entry point CLI
├── requirements.txt                    # Python dependencies
├── dataset_audit_report.md             # Empirical 12-phase audit report & protocol rules
├── dataset_statistics.json             # Precomputed corpus statistical data
├── web/                                # Web application & serving layer
│   ├── app.py                          # Flask backend with REST API
│   ├── inference.py                    # Standalone PubMedBERT inference engine
│   └── static/
│       ├── index.html                  # Scholarly workbench UI
│       ├── style.css                   # Munken Lynx design system + 1-page print CSS
│       └── app.js                      # UI logic, presets, theme & print triggers
├── models/
│   └── best_checkpoint_strict/         # Fine-tuned checkpoint configs & tokenizer
│       ├── README.md                   # Weights placement documentation
│       ├── config.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── CitationGuard-Unified/              # Master curated research dataset
│   ├── build_citationguard_dataset.py  # Fully reproducible build pipeline
│   ├── citationguard_verification.jsonl# Core 3-class verification dataset
│   ├── citationguard_master.jsonl      # Unified multi-task corpus
│   ├── DATASET_CARD.md                 # Complete documentation & datasheets
│   └── *.md, *.csv                     # Label analysis & split audit reports
└── data/, scitance-main/, ...          # Raw benchmark corpora for reproducibility
```

---

## 🔌 REST API Documentation

### 1. Verify Citation Proposition
- **Endpoint:** `POST /api/verify`
- **Request Body:**
  ```json
  {
    "claim": "0-dimer levels are elevated in patients with severe COVID-19.",
    "title": "Clinical characteristics of COVID-19 patients",
    "evidence": "Patients with severe illness had higher D-dimer levels compared with non-severe patients."
  }
  ```
- **Response:**
  ```json
  {
    "claim": "0-dimer levels are elevated in patients with severe COVID-19.",
    "title": "Clinical characteristics of COVID-19 patients",
    "evidence": "Patients with severe illness had higher D-dimer levels compared with non-severe patients.",
    "evidence_provided": true,
    "predicted_label": "SUPPORTED",
    "confidence": 0.9842,
    "probabilities": {
      "SUPPORTED": 0.9842,
      "CONTRADICTED": 0.0071,
      "NOT_ENOUGH_EVIDENCE": 0.0087
    },
    "inference_time_ms": 42.8,
    "input_tokens": 84
  }
  ```

### 2. Load Preset Study Cases
- **Endpoint:** `GET /api/samples`
- Returns curated literature verification pairs across medical, biochemical, and agricultural domains.

### 3. Engine Health Check
- **Endpoint:** `GET /api/health`
- Inspects device configuration, class mappings, and checkpoint status.

---

## 📊 Unified Dataset Card

The **CitationGuard-Unified** corpus unifies and standardizes 5 peer-reviewed scientific datasets:
1. **SciFact (EMNLP 2020):** Expert-written biomedical claims paired with PubMed abstracts.
2. **SCitance (ACL SDP 2024):** Natural citation context pairs with repaired metadata titles.
3. **MSVEC:** Real-world fact-checks matched against scientific literature for out-of-domain evaluation.
4. **SciCite (NAACL 2019):** Citation intent classification (`background`, `method`, `result`).
5. **SciClaim (Data Intelligence 2025):** Bio-agriculture sentence role detection (`claim`, `evidence`, `none`).

### Verification Split Distribution (Target Task)
| Split | Total Records | SUPPORTED | CONTRADICTED | NOT ENOUGH EVIDENCE |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 1,220 | 450 | 385 | 385 |
| **Dev** | 305 | 112 | 96 | 97 |
| **Test** | 381 | 141 | 120 | 120 |
| **Total** | **1,906** | **703 (36.9%)** | **601 (31.5%)** | **602 (31.6%)** |

All splits enforce document-level leakage isolation (`0` cross-split duplicate leakage).

---

## 📄 License & Intellectual Property

Copyright © 2026 **Ohi** ([@ekraislam](https://github.com/ekraislam)). All Rights Reserved.

This project, its machine learning pipelines, fine-tuned configurations, curated datasets, and interface designs are protected under the **CitationGuard AI Software & Research License** (see [LICENSE](LICENSE)).

- 🚫 **Strictly Forbidden:** Commercial exploitation, monetization, re-branding, plagiarizing authorship, or closed-source redistribution.
- 🔬 **Permitted:** Non-commercial educational study, peer-review inspection, and academic verification with mandatory attribution.

### Citation
If you reference CitationGuard AI in your academic work, cite as:

```bibtex
@software{citationguard_ai_2026,
  author = {Ohi and CitationGuard AI Research},
  title = {CitationGuard AI: Evidence-Aware Scientific Citation Verification for Detecting Unsupported Academic Claims},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ekraislam/citationguard-ai}}
}
```
