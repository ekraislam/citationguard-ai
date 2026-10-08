# CitationGuard AI — PubMedBERT Token Length & Sequence Truncation Report
**Project:** CitationGuard AI — Evidence-Aware Citation Verification for Detecting Unsupported Academic Claims  
**Phase:** Phase 5 — Scientific Transformer Evaluation (PubMedBERT)  
**Document:** `CitationGuard-Unified/experiments/transformer/transformer_token_length_report.md`  
**Tokenizer Model:** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` (WordPiece vocab: 30,522)  
**Date:** October 8, 2026  

---

## 1. Executive Summary

Prior to fine-tuning `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`, we performed a token-length audit across the complete CitationGuard verification dataset ($N=1,754$). The objective is to determine an optimal `max_seq_length` that balances **evidence context preservation** against **CPU quadratic computational complexity** ($O(L^2)$ attention).

We evaluate three input regimes:
1. **Strict Evidence Input:** `Claim: {c}\n\nTitle: {t}\n\nEvidence: {e_strict}` (Primary Experiment).
2. **Abstract Fallback Input:** `Claim: {c}\n\nTitle: {t}\n\nEvidence: {e_fallback}` (Ablation Experiment).
3. **Claim Text Alone:** `{claim_text}`.

---

## 2. Token Length Distribution Table

| Input Representation | Sample Count ($N$) | Mean Tokens | Median Tokens | Min | Max | 95th Percentile | 99th Percentile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strict Evidence Input** | 1,754 | **191.77** | **129.0** | 16 | 1,400 | 525.3 | 744.3 |
| **Abstract Fallback Input** | 1,754 | **251.68** | **234.0** | 16 | 1,400 | 574.4 | 804.5 |
| **Claim Text Alone** | 1,754 | **26.06** | **21.0** | 5 | 149 | 57.0 | 84.9 |

---

## 3. Sequence Truncation Impact Across Ceilings

We audited the proportion of samples that would undergo sequence truncation at standard transformer context boundaries:

| Context Ceiling ($L_{\text{max}}$) | Strict Evidence Input Truncation | Abstract Fallback Input Truncation | Claim-Only Truncation |
| :---: | :---: | :---: | :---: |
| **$L = 256$** | 489 samples (**27.88%**) | 691 samples (**39.40%**) | 0 samples (**0.00%**) |
| **$L = 384$** | **206 samples (11.74%)** | **276 samples (15.74%)** | 0 samples (**0.00%**) |
| **$L = 512$** | 98 samples (**5.59%**) | 136 samples (**7.75%**) | 0 samples (**0.00%**) |

---

## 4. Architectural Sequence Length Selection

### Selected Setting: $\mathbf{L_{\text{max}} = 384\text{ tokens}}$

**Scientific & Computational Justification:**
1. **Evidence Coverage:** Setting $L_{\text{max}} = 384$ preserves **88.26% of all strict evidence samples with zero truncation**. For the remaining 11.74% of samples (which derive from lengthy multi-sentence SCitance abstracts), the critical introductory and methodology rationale sentences are captured within the first 384 tokens.
2. **CPU Compute Viability:** On the 12th Gen Intel Core i5 processor, $384^2 = 147,456$ attention operations versus $512^2 = 262,144$ operations represents a **43.7% reduction in quadratic attention compute and memory overhead**.
3. **Dynamic Padding:** Tokenized batches will be dynamically padded to the batch maximum (rather than fixed 384 padding), ensuring that batches with concise SciFact sentence rationales (median length 129 tokens) process rapidly without idle compute.
