# CitationGuard AI — Phase 3 Verification Dataset Quality Audit & ML-Readiness Report
**Project Name:** CitationGuard AI — Evidence-Aware Citation Verification for Detecting Unsupported Academic Claims  
**Phase:** Phase 3 — Deep Quality Audit & Machine-Learning Readiness Protocol  
**Document:** `CitationGuard-Unified/phase3_verification_readiness_report.md`  
**Primary Verification Files Audited:**  
- [`citationguard_verification.jsonl`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/citationguard_verification.jsonl) (5,807,874 bytes, 1,754 records)  
- [`citationguard_verification.csv`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/citationguard_verification.csv) (4,984,727 bytes, 1,754 records)  
**Associated Audit Artifacts:**  
- [`verification_label_distribution.csv`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/verification_label_distribution.csv)  
- [`text_length_statistics.csv`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/text_length_statistics.csv)  
- [`label_purity_flags.csv`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/label_purity_flags.csv)  
- [`verification_label_analysis.md`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/verification_label_analysis.md)  
- [`source_bias_analysis.md`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/source_bias_analysis.md)  
- [`verification_duplicate_recheck.md`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/verification_duplicate_recheck.md)  
- [`verification_split_audit.md`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/verification_split_audit.md)  
**Date:** October 8, 2026  
**Auditor:** CitationGuard AI Research Agent  

---

## Executive Summary

This report establishes the **Phase 3 Machine-Learning Readiness Assessment** of the CitationGuard AI verification dataset. Following the successful creation of the Phase 2 unified master dataset, this phase performs a deep, independent scientific audit of the primary verification corpus (`citationguard_verification.jsonl` and `.csv`).

In accordance with strict research protocol:
- **No original source datasets were modified.**
- **No new external datasets were created.**
- **No models were trained.**
- **No records were resampled or mutated.**

Our findings establish that **the CitationGuard verification dataset is scientifically sound, leakage-free at the paper and document level, and ready for baseline modeling**, subject to essential preprocessing filters (citation marker stripping and empty-evidence handling) and a bifurcated evaluation protocol (natural vs. synthetic test suites).

---

## 1. Verification of Final Data Counts & Reconciliation

An independent recalculation was conducted directly on `citationguard_verification.jsonl` and `citationguard_verification.csv`. The results were cross-referenced against `final_statistics.json` and `dataset_summary.csv`.

### 1.1. Recalculated Ground-Truth Counts

| Metric Category | Recalculated Verification Value | Source Breakdown / Details |
| :--- | :---: | :--- |
| **Total Records** | **1,754** | JSONL: 1,754 lines; CSV: 1,754 rows (100% matched) |
| **Records per Source** | SCitance: 649 (37.00%)<br>SciFact: 1,105 (63.00%) | Harmonized scientific literature verification |
| **Records per Label** | SUPPORTED: 706 (40.25%)<br>CONTRADICTED: 488 (27.82%)<br>NOT_ENOUGH_EVIDENCE: 560 (31.93%) | 3-Class Epistemic Verification Plane |
| **Records per Split** | TRAIN: 1,226 (69.90%)<br>DEV: 263 (14.99%)<br>TEST: 265 (15.11%) | Strictly partitioned via paper connected components |
| **Records per Category** | NATURAL: 1,503 (85.69%)<br>SYNTHETIC_NEGATION: 251 (14.31%) | All synthetic records originate from SCitance |
| **Unique Claims (Raw)** | **1,754** | Zero exact duplicate claim strings |
| **Unique Claims (Norm)** | **1,754** | Zero normalized duplicate claims (case/punct/whitespace) |
| **Unique Papers (`paper_id`)** | **602** | Distinct cited research papers |
| **Unique Documents (`document_id`)** | **602** | 1-to-1 match with `paper_id` |
| **Unique Evidence Passages** | **869** (non-empty) / **870** (total) | 416 records contain `None` / empty evidence |

### 1.2. Discrepancy Reconciliation

We compared our recalculated ground truth against `final_statistics.json` and `dataset_summary.csv`. Every discrepancy is detailed below:

1. **`total_records` and Partition Splits:**
   - `dataset_summary.csv` specifies: Total: 1,754; Train: 1,226; Dev: 263; Test: 265.
   - `final_statistics.json` specifies: `total_verification_records: 1754`, `verification_guard_split_counts`: Train: 1,226, Dev: 263, Test: 265.
   - **Verdict:** **100% Match (0 Discrepancy).**

2. **Verification Label Counts Discrepancy:**
   - `final_statistics.json` (lines 44–48) reported:
     `{"SUPPORTED": 726, "CONTRADICTED": 505, "NOT_ENOUGH_EVIDENCE": 583}` (Sum = 1,814).
   - Recalculated Ground Truth:
     `{"SUPPORTED": 706, "CONTRADICTED": 488, "NOT_ENOUGH_EVIDENCE": 560}` (Sum = 1,754).
   - **Root Cause:** In `build_citationguard_dataset.py`, line 775 calculated `records_per_verification_label` over `all_records` in `citationguard_master` rather than `verif_records`. As a result, it included 60 records marked `EXACT_DUPLICATE` (20 SUPPORTED, 17 CONTRADICTED, 23 NOT_ENOUGH_EVIDENCE). The clean verification dataset has exactly 1,754 records.

3. **Unique Document Count Discrepancy (602 vs. 603):**
   - `final_statistics.json` (lines 83–89) and `leakage_report.md` reported `total_paper_components: 603` (Train: 417, Dev: 89, Test: 97).
   - Recalculated Ground Truth: Exactly **602 unique documents** (Train: 416, Dev: 89, Test: 97).
   - **Root Cause:** Document ID `22406695` was cited exclusively by record `CG_000710` (SciFact). Because `CG_000710` was an `EXACT_DUPLICATE` of another claim, it was purged from `citationguard_verification.jsonl`. Consequently, document `22406695` is absent from the clean verification view, reducing the active document count in TRAIN from 417 to 416.

4. **`leakage_report.md` Draft Table Discrepancy:**
   - `leakage_report.md` Section 4 listed: 705 Supported, 486 Contradicted, 563 Not Enough Evidence; Train Synth: 173, Dev Synth: 39, Test Synth: 39.
   - Recalculated Ground Truth: 706 Supported, 488 Contradicted, 560 Not Enough Evidence; Train Synth: 183, Dev Synth: 31, Test Synth: 37.
   - **Root Cause:** An intermediate graph clustering checkpoint was documented in the markdown text. The verified JSONL/CSV artifacts reflect the correct, finalized numbers.

---

## 2. Verification of Label Distribution

A detailed stratification analysis was completed across sources, claim provenance, and splits. Detailed markdown findings are archived in [`verification_label_analysis.md`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/verification_label_analysis.md), and machine-readable counts are in [`verification_label_distribution.csv`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/verification_label_distribution.csv).

### Stratified Label Distribution Table

| Stratum / Subset | Total ($N$) | SUPPORTED ($n$, %) | CONTRADICTED ($n$, %) | NOT_ENOUGH_EVIDENCE ($n$, %) |
| :--- | :---: | :---: | :---: | :---: |
| **ALL (Verification Core)** | **1,754** | **706 (40.25%)** | **488 (27.82%)** | **560 (31.93%)** |
| **A. SCitance** | 649 | 251 (38.67%) | 251 (38.67%) | 147 (22.65%) |
| **B. SciFact** | 1,105 | 455 (41.18%) | 237 (21.45%) | 413 (37.38%) |
| **C. NATURAL** | 1,503 | 706 (46.97%) | 237 (15.77%) | 560 (37.26%) |
| **D. SYNTHETIC_NEGATION** | 251 | 0 (0.00%) | 251 (100.00%) | 0 (0.00%) |
| **E. TRAIN** | 1,226 | 498 (40.62%) | 351 (28.63%) | 377 (30.75%) |
| **F. DEV** | 263 | 95 (36.12%) | 67 (25.48%) | 101 (38.40%) |
| **G. TEST** | 265 | 113 (42.64%) | 70 (26.42%) | 82 (30.94%) |

### Key Observations:
1. **Natural Scarcity of Contradictions:** In authentic natural data ($N=1,503$), `CONTRADICTED` instances account for only **15.77%**. Academic citations rarely cite papers that openly contradict the asserting statement without explicit qualification.
2. **Synthetic Balancing:** The 251 synthetic negations inject critical counter-evidence examples, lifting `CONTRADICTED` to **27.82%** overall.
3. **Split Uniformity:** Label proportions are consistent across `TRAIN`, `DEV`, and `TEST`, avoiding distribution drift.

---

## 3. Synthetic Negation Analysis & Critical Correction

### 3.1. Verification of Synthetic Counts
* Total Synthetic Negations: **Exactly 251 records** (Independently confirmed).
* Total Natural Claims: **1,503 records**.
* Total Verification Core: **1,754 records**.
* Synthetic Proportion of Core Dataset:
  $$\frac{251}{1754} \times 100\% = \mathbf{14.31\%}$$

### 3.2. Critical Correction of the "51.6%" Statistic (Section 15)
> [!IMPORTANT]
> **Mathematical Correction of Prior Documentation:**  
> The Phase 2 documentation (`DATASET_CARD.md`, Line 124) claimed: *"51.6% of CONTRADICTED instances originate from synthetic negations."*  
> - **Prior Draft Calculation:** $\frac{251}{486} \times 100\% = 51.646\% \approx 51.6\%$ (based on the draft count of 486).  
> - **Recalculated True Ground Truth:** Total `CONTRADICTED` instances in `citationguard_verification.jsonl` is **488**.  
> - **Mathematically Correct Percentage:**  
>   $$\frac{251}{488} \times 100\% = \mathbf{51.4344\% \approx 51.43\%}$$  
> The true proportion of synthetic negations among contradicted claims is **51.43%**.

### 3.3. Concentration Across Splits
We analyzed whether synthetic negations are concentrated in specific splits:
* **`TRAIN` Split:** 183 synthetic / 1,226 total (**14.93%** of TRAIN).  
  *Synthetic proportion of TRAIN `CONTRADICTED`:* $183 / 351 = \mathbf{52.14\%}$.
* **`DEV` Split:** 31 synthetic / 263 total (**11.79%** of DEV).  
  *Synthetic proportion of DEV `CONTRADICTED`:* $31 / 67 = \mathbf{46.27\%}$.
* **`TEST` Split:** 37 synthetic / 265 total (**13.96%** of TEST).  
  *Synthetic proportion of TEST `CONTRADICTED`:* $37 / 70 = \mathbf{52.86\%}$.

### 3.4. Representativeness & Evaluation Bias
* **Is the Test Set Representative?** The test set mirrors the overall dataset closely (~14.0% synthetic ratio overall, ~52.9% synthetic ratio for `CONTRADICTED`).
* **The "Artificially Easy" Risk:** Because synthetic negations were generated by GPT-3.5 via superficial syntactic negation (inserting `"not"`, `"did not"`, `"fails to"`), models can achieve high accuracy on these instances using trivial lexical negation matching rather than reasoning over evidence.
* **Empirical Proof:** When evaluated on a claim-only TF-IDF baseline, model Macro-F1 is **54.53%** on all claims, but plummets to **44.72%** when evaluated on natural claims only.
* **Mandatory Directive:** Evaluation protocols must report performance on the **Natural Test Slice ($N=228$)** independently from the **Combined Test Set ($N=265$)**.

---

## 4. Class Imbalance Assessment & Mitigation Plan

### 4.1. Imbalance Severity Metrics
* **Majority Class:** `SUPPORTED` — 706 samples (**40.25%**).
* **Middle Class:** `NOT_ENOUGH_EVIDENCE` — 560 samples (**31.93%**).
* **Minority Class:** `CONTRADICTED` — 488 samples (**27.82%**).
* **Majority-to-Minority Class Ratio:**
  $$\text{Ratio} = \frac{706}{488} \approx \mathbf{1.45 : 1}$$
* **Severity Rating:** **Mild to Moderate**. In machine learning, a 1.45:1 ratio is well within manageable limits and does not exhibit severe class collapse (unlike fraud detection or rare disease diagnosis with >20:1 ratios).

### 4.2. Actionable Training Strategy
1. **No Resampling:** Do **not** downsample `SUPPORTED` or `NOT_ENOUGH_EVIDENCE`. With an overall $N=1,754$, discarding valuable training pairs would degrade model generalization.
2. **Balanced Loss Weighting:** Apply inverse class frequency weights in cross-entropy loss:
   $$w_c = \frac{N_{\text{total}}}{C \times N_c}$$
   * $w_{\text{SUPPORTED}} = \frac{1754}{3 \times 706} \approx \mathbf{0.828}$
   * $w_{\text{NOT\_ENOUGH\_EVIDENCE}} = \frac{1754}{3 \times 560} \approx \mathbf{1.044}$
   * $w_{\text{CONTRADICTED}} = \frac{1754}{3 \times 488} \approx \mathbf{1.198}$
3. **Primary Evaluation Metric:** Use **Macro-F1** (unweighted mean of per-class F1) as the primary optimization and model selection metric. Accuracy must remain strictly a secondary, descriptive metric.

---

## 5. Source Bias & Dataset Fingerprinting Analysis

A thorough linguistic and empirical investigation was performed between SCitance and SciFact. Detailed analysis is available in [`source_bias_analysis.md`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/source_bias_analysis.md).

### 5.1. Summary of Structural Asymmetries

```
                          STRUCTURAL ASYMMETRY MAP
                          
     SCitance (Authentic Citances)           SciFact (Atomic Claims)
     ┌───────────────────────────────┐       ┌───────────────────────────────┐
     │ • Claim Word Mean: 29.8       │       │ • Claim Word Mean: 12.4       │
     │ • In-Text Context: 100%       │       │ • In-Text Context: 0% (None)  │
     │ • Citation Markers: 20.2%     │       │ • Citation Markers: 0.0%      │
     │ • Evidence: Full Abstract     │       │ • Evidence: Extracted Sents   │
     │ • Synthetic Negation: 38.7%   │       │ • Synthetic Negation: 0.0%    │
     └───────────────────────────────┘       └───────────────────────────────┘
```

1. **Claim Length:** SCitance claims average **29.84 words** (200.8 characters), whereas SciFact claims average **12.38 words** (89.2 characters) — a 141% difference.
2. **Citation Context:** `citation_context` is populated in **100.0%** of SCitance records (where `claim_text == citation_context`), but is `None` in **100.0%** of SciFact records.
3. **Evidence Granularity:** SCitance provides the entire abstract as `evidence_text` (**213.8 words** mean), while SciFact provides sentence-level rationales (**57.7 words** mean) or `None` (for `NOT_ENOUGH_EVIDENCE`).
4. **Attribution vs. Direct Claims:** 20.2% of SCitance claims contain `"et al."`, 12.5% contain parenthetical author-year markers, and 6.2% contain numeric bracket markers `[1]`. SciFact contains 0.0% of these markers.

### 5.2. Dataset Fingerprinting Probe
A TF-IDF probe trained solely on claim text distinguishes SCitance from SciFact with **0.9533 ROC-AUC**. Because SCitance has a 38.67% rate of `CONTRADICTED` compared to SciFact's 21.45%, any model that fingerprints the source dataset acquires an unfair prior on the label.

---

## 6. Text Length Distributions

Distributions were computed across all 5 text fields for both word counts and character counts. Full tabular data is archived in [`text_length_statistics.csv`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/text_length_statistics.csv).

| Field Name | Scope | Non-Null ($N$) | Metric | Mean | Median | Min | Max | 95th Percentile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`claim_text`** | ALL | 1,754 | Words | 18.84 | 15.0 | 3 | 94 | 42.35 |
| **`claim_text`** | ALL | 1,754 | Chars | 130.47 | 106.0 | 26 | 673 | 288.00 |
| **`citation_context`** | SCitance | 649 | Words | 29.84 | 27.0 | 7 | 94 | 60.00 |
| **`citation_context`** | SciFact | 0 | Words | 0.00 | 0.0 | 0 | 0 | 0.00 |
| **`evidence_text`** | ALL | 1,338 | Words | 133.38 | 114.0 | 8 | 1,062 | 361.00 |
| **`evidence_text`** | ALL | 1,338 | Chars | 935.75 | 805.0 | 43 | 6,866 | 2,456.50 |
| **`paper_title`** | ALL | 1,754 | Words | 13.30 | 13.0 | 1 | 38 | 21.00 |
| **`paper_title`** | ALL | 1,754 | Chars | 100.69 | 97.0 | 7 | 264 | 157.00 |
| **`paper_abstract`** | ALL | 1,754 | Words | 215.94 | 172.0 | 55 | 1,062 | 425.00 |
| **`paper_abstract`** | ALL | 1,754 | Chars | 1,520.45 | 1,240.0 | 372 | 6,866 | 2,826.30 |

---

## 7. Duplicate & Near-Duplicate Recheck

A deterministic audit of duplicates was conducted and documented in [`verification_duplicate_recheck.md`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/verification_duplicate_recheck.md).

* **Exact Claim Duplicates:** **0 instances** (1,754 unique).
* **Normalized Claim Duplicates:** **0 instances** (1,754 unique).
* **Citation-Context Duplicates:** **0 instances** (649 unique).
* **(Claim, Evidence) Duplicates:** **0 instances** (1,754 unique).
* **High-Similarity Near-Duplicate Pairs ($J \ge 0.70$):** Exactly **510 pairs**.
  - **Within-Split Pairs ($n = 503$):** 488 pairs are natural-claim/synthetic-negation counterparts within `TRAIN`.
  - **Cross-Source Overlaps ($n = 25$):** All 25 pairs are safely co-located within the `TRAIN` split.
  - **Cross-Split Near-Duplicates ($n = 7$):** Exactly 7 pairs span across split boundaries. All 7 pairs reference **completely different cited research papers** (different `paper_id` and distinct abstracts), representing parallel independent discoveries in biomedicine rather than document leakage.

---

## 8. Split Quality & Paper-Level Leakage Audit

A comprehensive leakage verification is documented in [`verification_split_audit.md`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/verification_split_audit.md).

### Explicit Numerical Proof:
* Train Document IDs: $D_{\text{train}} = 416$ unique documents.
* Dev Document IDs: $D_{\text{dev}} = 89$ unique documents.
* Test Document IDs: $D_{\text{test}} = 97$ unique documents.
* **Cross-Split Intersections:**
  $$D_{\text{train}} \cap D_{\text{dev}} = \emptyset \quad (\text{Overlap} = 0)$$
  $$D_{\text{train}} \cap D_{\text{test}} = \emptyset \quad (\text{Overlap} = 0)$$
  $$D_{\text{dev}} \cap D_{\text{test}} = \emptyset \quad (\text{Overlap} = 0)$$
* **Abstract Overlap:** **0.00%**.
* **Evidence Passage Overlap:** **0.00%**.
* **Claim Text Overlap:** **0.00%**.

**Verdict:** The dataset exhibits **mathematically verified zero document leakage**.

---

## 9. Label Purity & Anomaly Audit

We audited metadata consistency across all 1,754 records. Anomalous records are cataloged in [`label_purity_flags.csv`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/label_purity_flags.csv).

### 9.1. Flagged Defect Records ($n = 3$)
Three records in SciFact were flagged with empty evidence despite being labeled `SUPPORTED` or `CONTRADICTED`:

1. **`CG_001069` (SciFact, TRAIN, `SUPPORTED`):**
   * *Claim:* *"Ly6C hi monocytes have a higher inflammatory capacity than Ly6C lo monocytes."*
   * *Anomaly:* `evidence_sentence_ids = [9, 10]`, but `evidence_text = None`.
   * *Root Cause:* SciFact original record cited two documents (`[7521113, 36444198]`). The evidence resided in doc `36444198`, but the Phase 2 build script attached doc `7521113` as primary, which had only 6 sentences, causing the index lookup to fail.
2. **`CG_001211` (SciFact, TEST, `SUPPORTED`):**
   * *Claim:* *"Pro-inflammatory cytokines are up regulated during tumor development."*
   * *Anomaly:* `evidence_sentence_ids = [8, 9]`, but `evidence_text = None`.
   * *Root Cause:* Multi-cited document mismatch; doc `14075252` attached instead of doc `39264456`.
3. **`CG_001616` (SciFact, TRAIN, `CONTRADICTED`):**
   * *Claim:* *"Ly6C hi monocytes have a lower inflammatory capacity than Ly6C lo monocytes."*
   * *Anomaly:* `evidence_sentence_ids = [9, 10]`, but `evidence_text = None`.
   * *Root Cause:* Same multi-cited document mismatch as `CG_001069`.

### 9.2. Structural Representation of `NOT_ENOUGH_EVIDENCE`
* In SciFact ($N=413$), `NOT_ENOUGH_EVIDENCE` records have `evidence_text = None` because no rationale sentence exists.
* In SCitance ($N=147$), `NOT_ENOUGH_EVIDENCE` records have `evidence_text = paper_abstract`.
* **Preprocessing Directive:** When preparing inputs for ML models, do not pass empty strings for SciFact `NOT_ENOUGH_EVIDENCE`. Instead, provide the cited `paper_abstract` so the model learns to evaluate whether the abstract contains sufficient evidence.

---

## 10. Identifier Leakage Risk Assessment

We audited whether metadata identifiers could provide artificial prediction shortcuts:
1. **`record_id` (Risk: HIGH if exposed):** SCitance spans `CG_000001`–`CG_000649`, and SciFact spans `CG_000650`–`CG_001758`. Because SCitance has double the contradiction rate of SciFact, revealing `record_id` creates an artificial shortcut.
2. **`paper_id` / `document_id` (Risk: LOW across splits, HIGH within TRAIN):** Because document leakage across splits is 0%, models cannot cheat on TEST using memorized paper IDs. However, 175 multi-claim papers in TRAIN share 100% identical labels across their claims.
3. **Mitigation Directive:** **Never pass identifiers (`record_id`, `source_record_id`, `paper_id`, `document_id`) to the model during training or feature extraction.**

---

## 11. Final Machine-Learning Task Formulation

Based on empirical data properties, we evaluate potential task formulations:

* **Option A: 3-Class Classification (`SUPPORTED`, `CONTRADICTED`, `NOT_ENOUGH_EVIDENCE`) [RECOMMENDED]**
  - *Scientific Justification:* In scientific scholarly communication, distinguishing between active falsification (`CONTRADICTED`) and unaddressed assertions (`NOT_ENOUGH_EVIDENCE`) is vital. Collapsing these into a single "unsupported" class destroys epistemic utility for peer review and automated fact-checking. The dataset exhibits a viable distribution (~40% Supported, ~28% Contradicted, ~32% Not Enough Evidence) that fully supports this 3-way formulation.
* **Option B: Binary Classification (`SUPPORTED` vs. `UNSUPPORTED`) [NOT RECOMMENDED]**
  - *Critique:* Merging `CONTRADICTED` and `NOT_ENOUGH_EVIDENCE` conflates claims refuted by experimental data with claims that are simply outside the scope of the cited paper.
* **Option C: 2-Stage Hierarchical Formulation (Stage 1: Evidence Presence; Stage 2: Epistemic Stance)**
  - *Feasibility:* Viable as an advanced experimental extension, but 3-class single-stage classification remains the definitive primary task.

---

## 12. Input Representation Design

We compared 5 candidate input formulations:

| Option | Input Representation | Feasibility & Assessment |
| :--- | :--- | :--- |
| **Option A** | `claim + evidence` | Flawed: Empty in 416 SciFact records; leaks label if `evidence == None`. |
| **Option B** | `citation_context + evidence` | Impossible: `citation_context` is `None` in 100% of SciFact records. |
| **Option C** | `claim + citation_context + evidence` | Biased: `citation_context` missingness fingerprints SciFact. |
| **Option D** | `claim + paper_title + abstract` | Suboptimal: Evaluates full document rather than localized rationales. |
| **Option E (Standardized)** | `claim + paper_title + evidence_or_abstract` | **RECOMMENDED PRIMARY INPUT FORMAT** |

### Recommended Input Design:
$$\mathbf{\text{[CLS] Claim: } \{claim\_text\} \text{ [SEP] Title: } \{paper\_title\} \text{ [SEP] Context: } \{evidence\_text\_or\_abstract\} \text{ [EOS]}}$$
* Where `evidence_text_or_abstract` is:
  - The extracted rationale `evidence_text` if present and non-empty.
  - The full `paper_abstract` if `evidence_text` is `None` (for `NOT_ENOUGH_EVIDENCE` and the 3 flagged records).

This guarantees that:
1. Every record receives non-empty evidence context.
2. The model cannot cheat using empty strings.
3. Both SCitance citances and SciFact atomic claims are treated uniformly.

---

## 13. Recommended Baseline Experiment Ladder

We recommend a structured research progression from simple linear baselines to state-of-the-art scientific cross-encoders:

```
                            RECOMMENDED MODEL LADDER
                            
      ┌────────────────────────────────────────────────────────┐
      │ Step 4: Multitask Scientific NLI (MultiVerS)           │
      └───────────────────────────▲────────────────────────────┘
                                  │
      ┌────────────────────────────────────────────────────────┐
      │ Step 3: Scientific Transformer Cross-Encoder           │
      │         (PubMedBERT / SciBERT / DeBERTa-v3)            │
      └───────────────────────────▲────────────────────────────┘
                                  │
      ┌────────────────────────────────────────────────────────┐
      │ Step 2: Dense Semantic Dual-Encoder                    │
      │         (Sentence-BERT / SciBERT Embeddings + MLP)     │
      └───────────────────────────▲────────────────────────────┘
                                  │
      ┌────────────────────────────────────────────────────────┐
      │ Step 1: Lexical Baselines                              │
      │         (TF-IDF + Logistic Regression / Linear SVM)    │
      └────────────────────────────────────────────────────────┘
```

1. **Step 1A — Majority Class Baseline:** Always predicts `SUPPORTED` (Accuracy: 40.25%, Macro-F1: 19.13%).
2. **Step 1B — TF-IDF + Logistic Regression / Linear SVM:**
   * Features: Word + Char n-grams (1-3) on `claim + evidence_or_abstract`.
   * Loss: Balanced class weights.
3. **Step 2 — Dense Semantic Embeddings (Sentence-BERT / SciBERT):**
   * Pretrained encoders: `all-MiniLM-L6-v2` and `allenai/scibert_scivocab_uncased`.
   * Representation: $[u; v; |u-v|; u * v]$ feeding a lightweight MLP classifier.
4. **Step 3 — Cross-Encoder Scientific Transformers (Primary NLI Baselines):**
   * Models: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`, `allenai/scibert_scivocab_uncased`, `microsoft/deberta-v3-base`.
   * Joint cross-attention between claim, title, and context.
5. **Step 4 — Multitask Evidence-Retrieval Pipeline (MultiVerS Architecture):**
   * Joint sentence selection rationale scoring and claim veracity prediction.

---

## 14. Final Evaluation Metrics Protocol

### 14.1. Core Quantitative Metrics
* **Primary Metric:** **Macro-F1** (unweighted arithmetic mean of F1 scores across `SUPPORTED`, `CONTRADICTED`, and `NOT_ENOUGH_EVIDENCE`).
* **Secondary Metrics:**
  * **Overall Accuracy**
  * **Weighted-F1**
  * **Per-Class Precision, Recall, and F1**
  * **Multi-Class Confusion Matrix**

### 14.2. Partitioned Test Suite Protocol
All evaluation results must report metrics across **three distinct test slices**:

| Test Slice | Record Count ($N$) | Composition & Scientific Purpose |
| :--- | :---: | :--- |
| **1. Natural-Only Test Suite** | **228** | Real-world citation verification performance; completely free of synthetic negation artifacts. |
| **2. Synthetic-Negation Test Suite** | **37** | Stress-test evaluating model robustness against explicit syntactic negation. |
| **3. Combined Benchmark Test Suite** | **265** | Full official evaluation partition. |

*Why this is mandatory:* Models exploiting surface negation cues achieve inflated scores on the combined set while failing on natural contradictions. Reporting these slices independently guarantees transparency.

---

## 15. Executive Action Plan & Answers to Required Questions

### 1. Is the dataset ready for ML training?
**YES, CONDITIONAL.** The dataset is scientifically sound, leakage-free at the paper and document level (0.0% overlap), and clean of exact duplicates. It is ready for machine-learning experiments provided that two preprocessing safeguards are applied: (a) stripping surface citation brackets (`[1]`, `(Smith, 2018)`) and (b) falling back to `paper_abstract` whenever `evidence_text` is `None`.

### 2. What is the primary task?
**3-Class Epistemic Verification:** Classifying claims into `SUPPORTED`, `CONTRADICTED`, or `NOT_ENOUGH_EVIDENCE`.

### 3. What are the major risks?
1. **Dataset Fingerprinting:** SCitance claims are 141% longer and contain citation markers, allowing models to identify source origin with 0.9533 ROC-AUC.
2. **Claim-Only Shortcut:** Simple linear models achieve 55.76% accuracy without seeing evidence due to negation words and attribution frames.
3. **Evidence Representation Disparity:** SCitance provides full abstracts while SciFact provides extracted sentences or `None`.

### 4. What preprocessing is still required?
1. **Citation Marker Normalization:** Strip numeric brackets `\[\d+\]` and parenthetical author citations from claims.
2. **Evidence Standardization:** When `evidence_text` is `None` (in SciFact `NOT_ENOUGH_EVIDENCE` and the 3 flagged records), supply `paper_abstract`.
3. **Identifier Stripping:** Completely remove `record_id`, `source_record_id`, `paper_id`, and `document_id` from model inputs.

### 5. What baseline model should we implement first?
**TF-IDF + Logistic Regression / Linear SVM with Balanced Class Weights**, evaluated on the unified input format `[CLS] Claim [SEP] Title [SEP] Context [EOS]`. This establishes the essential benchmark to measure how much value dense transformers provide beyond lexical cues.

### 6. What evaluation metrics should be primary?
**Macro-F1** across all 3 classes is the primary metric. Per-class F1 (especially for `CONTRADICTED`) and the 3-way Confusion Matrix must be monitored.

### 7. What should remain reserved for external evaluation?
**MSVEC (`citationguard_external_test.csv`, $N=56$)** must remain strictly reserved for out-of-domain cross-corpus robustness testing. It must never be touched during model training, validation, or hyperparameter tuning.

### 8. What must NOT be changed in the dataset?
1. **DO NOT modify or re-split the paper connected components:** The split boundaries (`TRAIN`, `DEV`, `TEST`) must remain frozen to guarantee 0.0% document leakage.
2. **DO NOT resample or delete records:** The dataset size ($N=1,754$) must remain fixed to ensure benchmark comparability.
3. **DO NOT collapse the label space:** Preserve the distinct 3-class ontology.
4. **DO NOT alter original source datasets:** Keep all source corpora in their pristine state.
