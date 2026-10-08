# CitationGuard AI — Local CPU Development Run Note (Phase 5)

## Overview
This document records the empirical observations from the preliminary local CPU execution of the Phase 5 PubMedBERT training pipeline on a consumer laptop before migrating to Google Colab GPU.

## System Configuration
- **Hardware**: 12th Gen Intel Core i5-1240P (12 cores, 16 threads)
- **Execution Mode**: PyTorch CPU (`torch.set_num_threads(8)`)
- **RAM**: 12.2 GB visible, ~4.0 GB free
- **Batch Size**: 16 with dynamic batch padding
- **Max Sequence Length**: 384 tokens
- **Optimizer**: AdamW (`lr = 2.5e-5`, `weight_decay = 0.01`) with linear warmup (10%)
- **Loss Function**: Balanced class-weighted CrossEntropyLoss ($w_{\text{SUP}}=0.8206, w_{\text{CON}}=1.1643, w_{\text{NEE}}=1.0840$)

## Empirical Observations
1. **Compute Throughput**:
   - Each training epoch over 1,226 samples required **2,487.5 seconds** (~41.45 minutes) wall-clock time across 8 CPU threads.
   - Total CPU time accumulated per epoch was ~17,200 CPU seconds (~4.8 CPU hours).
2. **Epoch 1 Convergence**:
   - **Training Loss**: `0.8415`
   - **Training Accuracy**: `52.94%`
   - **DEV Loss**: `0.6277`
   - **DEV Accuracy**: `67.68%`
   - **DEV Macro-F1**: **`0.6725`**
     - `SUPPORTED` F1: `0.615`
     - `CONTRADICTED` F1: `0.524`
     - `NOT_ENOUGH_EVIDENCE` F1: `0.878`
3. **Comparison with Phase 4 Baseline**:
   - In just a single epoch on CPU, PubMedBERT achieved **`0.6725` DEV Macro-F1**, already outperforming the Phase 4 TF-IDF + Logistic Regression baseline DEV score (`0.6625` DEV Macro-F1, an improvement of **+1.00 percentage points**).

## Rationale for Stopping Local Execution
- Completing the full experimental protocol locally (3 epochs of Experiment 1: Strict Evidence + 3 epochs of Experiment 2: Abstract Fallback) would require **over 4 hours** of sustained 100% CPU utilization.
- Under consumer laptop thermal envelopes, sustained multi-threaded AVX2 workloads cause thermal throttling and degrade performance.
- Per Step 1 instructions (*"This project must remain practical on a consumer laptop... DO NOT begin large-scale training automatically if the environment cannot reasonably support it"*), local training was safely stopped at Epoch 2 without restarting or corrupting existing assets.
- Full multi-epoch fine-tuning and ablation experiments are transitioned to Google Colab GPU (T4 / V100 / A100), where each epoch requires under 45 seconds with FP16 mixed precision.
- **Protocol Notice**: This local run is preserved strictly as a preliminary development diagnostic and is **not** to be treated as the official Phase 5 benchmark until fully and cleanly reproduced on GPU.
