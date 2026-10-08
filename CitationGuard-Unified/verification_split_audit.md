# CitationGuard AI — Verification Split Quality & Leakage Audit
**Project:** CitationGuard AI — Evidence-Aware Citation Verification for Detecting Unsupported Academic Claims  
**Phase:** Phase 3 — Verification Quality Audit & ML-Readiness Protocol  
**Document:** `CitationGuard-Unified/verification_split_audit.md`  
**Date:** October 8, 2026  

---

## 1. Executive Summary

Data leakage between training and evaluation splits is a fatal methodological flaw in scientific NLP. When identical research papers, full abstracts, or evidence rationales appear in both `TRAIN` and `TEST`, models memorize paper-specific terminology, entities, and citations rather than learning verifiable evidence reasoning.

This audit provides **rigorous, deterministic mathematical proof** of split isolation for the primary CitationGuard verification dataset (`citationguard_verification.jsonl`, $N=1,754$). We evaluate paper identifiers, integer document identifiers, raw abstract strings, evidence rationale passages, and normalized claim propositions across all partition boundaries (`TRAIN`, `DEV`, `TEST`).

---

## 2. Quantitative Split Partition Matrix

The core verification view is partitioned into three splits:

| Split Name | Record Count ($N$) | Percentage of Total | Unique Papers (`paper_id`) | Unique Documents (`document_id`) | Unique Abstracts |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`TRAIN`** | 1,226 | 69.90% | 416 | 416 | 416 |
| **`DEV`** | 263 | 14.99% | 89 | 89 | 89 |
| **`TEST`** | 265 | 15.11% | 97 | 97 | 97 |
| **Total Set Cardinality** | **1,754** | **100.00%** | **602** | **602** | **602** |

*Note on Document Count Discrepancy:* Prior documentation (`leakage_report.md` and `final_statistics.json`) cited 603 total paper components (417 train, 89 dev, 97 test). Our independent audit reveals that document `22406695` was associated exclusively with record `CG_000710`, which was removed from the clean verification view as an `EXACT_DUPLICATE`. The active verification view contains exactly **602 unique documents** (416 train, 89 dev, 97 test).

---

## 3. Mathematical Proof of Zero Leakage

Let $D_{\text{train}}$, $D_{\text{dev}}$, and $D_{\text{test}}$ represent the sets of integer document identifiers (`document_id`) in each split:
* $|D_{\text{train}}| = 416$
* $|D_{\text{dev}}| = 89$
* $|D_{\text{test}}| = 97$

### 3.1. Document and Paper Identifier Isolation
We compute the pairwise set intersections:
$$\text{Overlap}(D_{\text{train}}, D_{\text{dev}}) = |D_{\text{train}} \cap D_{\text{dev}}| = 0$$
$$\text{Overlap}(D_{\text{train}}, D_{\text{test}}) = |D_{\text{train}} \cap D_{\text{test}}| = 0$$
$$\text{Overlap}(D_{\text{dev}}, D_{\text{test}}) = |D_{\text{dev}} \cap D_{\text{test}}| = 0$$

$$\mathbf{D_{\text{train}} \cap D_{\text{dev}} \cap D_{\text{test}} = \emptyset \quad (\text{Cross-Split Document Leakage} = 0.00\%)}$$

Every individual paper is strictly quarantined into exactly one split.

### 3.2. Raw Abstract Text Isolation
To guard against hash collisions or divergent document ID schemas across corpora, we compared raw text strings of all abstracts (`paper_abstract`):
* $|A_{\text{train}}| = 416$, $|A_{\text{dev}}| = 89$, $|A_{\text{test}}| = 97$
* $|A_{\text{train}} \cap A_{\text{dev}}| = 0$
* $|A_{\text{train}} \cap A_{\text{test}}| = 0$
* $|A_{\text{dev}} \cap A_{\text{test}}| = 0$

$$\mathbf{\text{Cross-Split Abstract Leakage} = 0.00\%}$$

### 3.3. Ground-Truth Evidence Passage Isolation
We evaluated ground-truth sentence rationales (`evidence_text` non-empty strings):
* $|E_{\text{train}}| = 604$, $|E_{\text{dev}}| = 127$, $|E_{\text{test}}| = 138$
* $|E_{\text{train}} \cap E_{\text{dev}}| = 0$
* $|E_{\text{train}} \cap E_{\text{test}}| = 0$
* $|E_{\text{dev}} \cap E_{\text{test}}| = 0$

$$\mathbf{\text{Cross-Split Evidence Rationale Overlap} = 0.00\%}$$

Zero evidence sentences from the training split appear in either the validation or test splits.

### 3.4. Normalized Claim Text Isolation
We compared normalized claim text strings (`claim_text`):
* $|C_{\text{train}}| = 1,226$, $|C_{\text{dev}}| = 263$, $|C_{\text{test}}| = 265$
* $|C_{\text{train}} \cap C_{\text{dev}}| = 0$
* $|C_{\text{train}} \cap C_{\text{test}}| = 0$
* $|C_{\text{dev}} \cap C_{\text{test}}| = 0$

$$\mathbf{\text{Cross-Split Exact Claim Leakage} = 0.00\%}$$

---

## 4. Semantic Claim Proximity Across Splits

While exact claim leakage is **0.00%**, our near-duplicate audit (documented in [`verification_duplicate_recheck.md`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/verification_duplicate_recheck.md)) detected **7 pairs** of claims across splits exhibiting Jaccard token similarity $\ge 0.70$:
* **Pair 1 & 2 (CRISPR immunity bias):** `CG_000206` (DEV) / `CG_000461` (DEV) vs. `CG_000683` (TRAIN) ($J = 0.957$).
* **Pair 3 (Homocysteine levels):** `CG_000673` (TRAIN) vs. `CG_001463` (TEST) ($J = 0.700$).
* **Pair 4 (Paraquat poisoning):** `CG_000792` (TRAIN) vs. `CG_000914` (DEV) ($J = 0.727$).
* **Pair 5 (EAM susceptibility):** `CG_001105` (DEV) vs. `CG_001627` (TRAIN) ($J = 0.800$).
* **Pair 6 & 7 (HPV cytology screening):** `CG_001208` (TEST) / `CG_001668` (TEST) vs. `CG_001209` (TRAIN) ($J = 0.810$).

### Scientific Evaluation:
1. All 7 pairs reference **completely distinct cited documents** (different `paper_id` and distinct abstracts).
2. The graph-based document component splitting operated flawlessly, as no document was shared between splits.
3. However, because different scientific papers investigated identical or closely related biomedical questions, high-level semantic proposition overlap exists across splits for these specific concepts.
4. **Conclusion:** Split isolation at the document and evidence level is **100% airtight**. Model evaluation should retain these 7 pairs in test/dev as a test of cross-paper concept generalization, but flag them for sensitivity analysis.
