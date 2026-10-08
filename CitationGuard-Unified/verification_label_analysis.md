# CitationGuard AI — Verification Label Distribution & Stratification Analysis
**Project:** CitationGuard AI — Evidence-Aware Citation Verification for Detecting Unsupported Academic Claims  
**Phase:** Phase 3 — Verification Quality Audit & ML-Readiness Protocol  
**Document:** `CitationGuard-Unified/verification_label_analysis.md`  
**Associated Artifact:** [`verification_label_distribution.csv`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/verification_label_distribution.csv)  
**Date:** October 8, 2026  

---

## 1. Executive Summary

This report delivers an independent, mathematically validated audit of the label space for the primary CitationGuard verification dataset (`citationguard_verification.jsonl`, $N=1,754$). The target task is **evidence-aware 3-class scientific citation and claim verification**, categorizing claims into `SUPPORTED`, `CONTRADICTED`, and `NOT_ENOUGH_EVIDENCE`.

We evaluate the global label distribution, stratify by source dataset (`SCitance` vs. `SciFact`), isolate the influence of synthetic negations (`NATURAL` vs. `SYNTHETIC_NEGATION`), and audit distribution stability across partition splits (`TRAIN`, `DEV`, `TEST`).

---

## 2. Comprehensive Label Distribution Matrix

The following table summarizes the verified sample counts and percentages across all relevant dataset strata:

| Stratum / Subset | Total Records ($N$) | SUPPORTED ($n$, %) | CONTRADICTED ($n$, %) | NOT_ENOUGH_EVIDENCE ($n$, %) |
| :--- | :---: | :---: | :---: | :---: |
| **ALL (Verification Core)** | **1,754** | **706 (40.25%)** | **488 (27.82%)** | **560 (31.93%)** |
| **SCitance** | 649 | 251 (38.67%) | 251 (38.67%) | 147 (22.65%) |
| **SciFact** | 1,105 | 455 (41.18%) | 237 (21.45%) | 413 (37.38%) |
| **NATURAL** | 1,503 | 706 (46.97%) | 237 (15.77%) | 560 (37.26%) |
| **SYNTHETIC_NEGATION** | 251 | 0 (0.00%) | 251 (100.00%) | 0 (0.00%) |
| **TRAIN** | 1,226 | 498 (40.62%) | 351 (28.63%) | 377 (30.75%) |
| **DEV** | 263 | 95 (36.12%) | 67 (25.48%) | 101 (38.40%) |
| **TEST** | 265 | 113 (42.64%) | 70 (26.42%) | 82 (30.94%) |

*Note: Machine-readable data is archived in [`verification_label_distribution.csv`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/verification_label_distribution.csv).*

---

## 3. Stratum-by-Stratum Epistemic Breakdown

### 3.1. Source Disparity: SCitance vs. SciFact
* **SCitance ($N=649$):** Demonstrates an artificially symmetric distribution between `SUPPORTED` (38.67%) and `CONTRADICTED` (38.67%). This symmetry is directly induced by GPT-3.5 synthetic negations, which constitute exactly 251 out of the 649 claims. `NOT_ENOUGH_EVIDENCE` accounts for only 22.65% of SCitance.
* **SciFact ($N=1,105$):** Reflects genuine human-annotated scientific claim distributions: `SUPPORTED` (41.18%) is the primary class, followed by `NOT_ENOUGH_EVIDENCE` (37.38%). Natural `CONTRADICTED` instances form only 21.45% of the data.

### 3.2. Natural vs. Synthetic Stratification
* **Natural Claims ($N=1,503$):** In the unmanipulated scientific literature, explicit refutations are rare. Natural `CONTRADICTED` samples represent only **15.77%** (237 samples), while `SUPPORTED` comprises **46.97%** and `NOT_ENOUGH_EVIDENCE` comprises **37.26%**.
* **Synthetic Negation Claims ($N=251$):** 100% of synthetic negations are assigned to `CONTRADICTED`. There are zero synthetic samples in `SUPPORTED` or `NOT_ENOUGH_EVIDENCE`.
* **Impact:** Synthetic negations account for **51.43%** ($251 / 488$) of all contradicted training/testing samples in the core verification dataset.

### 3.3. Cross-Split Distribution Stability (Train vs. Dev vs. Test)
The deterministic group partitioning protocol preserved remarkable label stability across splits:
* `SUPPORTED`: Train (40.62%), Dev (36.12%), Test (42.64%).
* `CONTRADICTED`: Train (28.63%), Dev (25.48%), Test (26.42%).
* `NOT_ENOUGH_EVIDENCE`: Train (30.75%), Dev (38.40%), Test (30.94%).

Maximum deviation across splits for any class is under 7.7%, confirming that the paper-level graph clustering did not introduce pathological label skew.

---

## 4. Discrepancy Reconciliation with Phase 2 Records

1. **`final_statistics.json` Discrepancy:**
   `final_statistics.json` reported `records_per_verification_label`: `SUPPORTED: 726`, `CONTRADICTED: 505`, `NOT_ENOUGH_EVIDENCE: 583` (Total: 1,814).
   * **Root Cause:** That calculation aggregated the master dataset before filtering out the 60 `EXACT_DUPLICATE` records. The true verification view contains exactly **1,754** unique, clean records (706 Supported, 488 Contradicted, 560 Not Enough Evidence).
2. **`leakage_report.md` Draft Table Discrepancy:**
   The intermediate table in `leakage_report.md` Section 4 listed: 705 Supported, 486 Contradicted, 563 Not Enough Evidence.
   * **Root Cause:** A slight drift during an intermediate graph component iteration. The recalculated numbers above represent the verified ground truth of `citationguard_verification.jsonl`.
3. **The "51.6%" Synthetic Percentage Mistake:**
   The Phase 2 documentation cited $51.6\%$ based on $251 / 486 = 51.65\%$. On the actual verified dataset ($N=488$ Contradicted), the exact value is **51.43%** ($251 / 488$).

---

## 5. Audit Recommendations for Model Training

1. **Retain 3-Class Formulation:** The distribution supports a healthy 3-way classification problem with approximately a 40:28:32 split.
2. **Loss Weighting:** To prevent models from underpredicting the minority class `CONTRADICTED`, apply inverse class weighting during training:
   $$w_{\text{SUP}} = \frac{1754}{3 \times 706} \approx 0.828, \quad w_{\text{CON}} = \frac{1754}{3 \times 488} \approx 1.198, \quad w_{\text{NEE}} = \frac{1754}{3 \times 560} \approx 1.044$$
3. **Dual Evaluation Protocol:** Benchmark performance must be computed separately on `NATURAL` test samples ($N=228$) and `ALL` test samples ($N=265$) to detect synthetic negation overfitting.
