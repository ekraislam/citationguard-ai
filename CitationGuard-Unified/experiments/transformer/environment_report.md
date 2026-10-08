# CitationGuard AI — Hardware & Execution Environment Audit Report
**Project:** CitationGuard AI — Evidence-Aware Citation Verification for Detecting Unsupported Academic Claims  
**Phase:** Phase 5 — Scientific Transformer Evaluation (PubMedBERT)  
**Document:** `CitationGuard-Unified/experiments/transformer/environment_report.md`  
**Date:** October 8, 2026  

---

## 1. Executive Summary

Prior to initiating Phase 5 deep learning experiments, an automated audit of the local hardware architecture and Python execution runtime was conducted. This assessment establishes operating constraints to guarantee that scientific transformer fine-tuning executes stably, deterministically, and safely on consumer laptop hardware without memory exhaustion or system instability.

---

## 2. Hardware Resource Specifications

| Hardware Component | Detected System Parameter | Operational Assessment |
| :--- | :--- | :--- |
| **CPU Model** | 12th Gen Intel(R) Core(TM) i5-1240P | High-performance hybrid CPU (12 cores, 16 logical threads) |
| **Physical CPU Cores** | 12 physical cores (4 Performance-cores, 8 Efficient-cores) | Capable of high-throughput multi-threaded tokenization & inference |
| **Logical Processors** | 16 execution threads | Set `torch.set_num_threads(8)` to optimize throughput without starvation |
| **Total Physical RAM** | 12.54 GB (11.67 GiB) | Safe memory envelope; batch sizes $\le 16$ ensure memory usage $< 4\text{ GB}$ |
| **Primary Storage (C:)** | 54.2 GB free space | Ample storage for model weights (~440 MB) and checkpoint caches |
| **GPU / Accelerators** | None detected (CUDA Available = `False`) | CPU-only PyTorch execution mode |
| **Operating System** | Microsoft Windows 11 Home (x64, Build 26200) | Fully supported runtime environment |

---

## 3. Software & Framework Stack

| Software Package | Installed Version | Purpose in Pipeline |
| :--- | :--- | :--- |
| **Python** | 3.12.10 (AMD64) | Primary language runtime |
| **PyTorch** | 2.14.1+cpu | CPU-optimized deep learning tensor execution engine |
| **Hugging Face Transformers**| 5.19.0 | Scientific model backbones, tokenizers & heads |
| **Accelerate** | 1.15.0 | Efficient execution orchestration |
| **scikit-learn** | 1.6.1 | Deterministic metric computation & baseline comparison |
| **Matplotlib** | 3.11.2 | Training curve and confusion matrix rendering |

---

## 4. CPU-Safe Training Strategy & Memory Constraints

Because execution runs on a 12th Gen Intel Core i5 processor without dedicated CUDA acceleration, the following **conservative, stability-first hyperparameter envelope** is mandated:

1. **Backbone Selection:**  
   `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` (110M parameters).  
   *PubMedBERT-base* fits comfortably in ~440 MB of RAM, allowing fast forward-backward passes on modern multicore CPUs.
2. **Batch Sizing & Gradient Accumulation:**  
   * Per-device Batch Size: $B = 8$ or $16$.  
   * Gradient Accumulation Steps: $2$ (effective batch size = 16 or 32).  
   * Keeps peak process memory well below 4 GB (leaving $> 8\text{ GB}$ system headroom).
3. **Sequence Length Optimization:**  
   * Sequence length dynamically analyzed via tokenizer inspection (Step 6) to cap maximum length at 256 or 384 tokens, avoiding quadratic attention overhead over padded tokens.
4. **Thread Allocation:**  
   * Explicitly allocate 8 compute threads (`torch.set_num_threads(8)`) for optimized Intel OpenMP parallel matrix multiplication.
5. **Early Stopping & Checkpoint Pruning:**  
   * Monitor `DEV Macro-F1` after each epoch; retain only the single best model weights (`best_checkpoint/`) to prevent disk bloat.
