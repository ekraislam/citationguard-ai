# CitationGuard AI — Source-Wise Performance & Disparity Report
**Project:** CitationGuard AI — Evidence-Aware Citation Verification for Detecting Unsupported Academic Claims  
**Phase:** Phase 4 — Lexical Baselines & Evidence Ablation  
**Document:** `CitationGuard-Unified/experiments/reports/source_performance_report.md`  
**Associated Artifact:** [`metric_comparison.csv`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/experiments/reports/metric_comparison.csv)  
**Date:** October 8, 2026  

---

## 1. Executive Summary

To determine whether baseline models perform uniformly across different citation corpora, we evaluated model predictions post-hoc by source dataset on the frozen `TEST` split ($N=265$):
* **SCitance Sub-cohort ($N=93$, 35.09%):** Authentic citation sentences extracted from citing research papers.
* **SciFact Sub-cohort ($N=172$, 64.91%):** Decontextualized atomic claims formulated by expert annotators.

*Note: In accordance with Rule 5, `source_dataset` was strictly quarantined and never provided as an input feature during model training.*

Our diagnostic analysis reveals a **striking inversion of performance** between SCitance and SciFact depending on the input condition:
* In **Claim-Only mode (Condition A)**, SCitance appears artificially easier (+21.5% Accuracy) due to synthetic negation cues.
* In **Evidence-Aware mode (Condition C)**, SciFact becomes dramatically superior (+25.6% Accuracy, +30.7% Macro-F1) due to clean sentence-level rationale annotations.

---

## 2. Quantitative Performance Matrix (TEST Set)

### 2.1. Condition A (Claim Only) Source Comparison

| Model | Source Cohort | Test Samples ($N$) | Accuracy | Macro-F1 | SUPPORTED F1 | CONTRADICTED F1 | NOT_ENOUGH_EVIDENCE F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **SCitance** | 93 | **0.6452** | **0.6164** | 0.615 | **0.789** | 0.444 |
| **Logistic Regression** | **SciFact** | 172 | **0.4302** | **0.3957** | 0.543 | **0.290** | 0.354 |
| **Linear SVM** | **SCitance** | 93 | **0.6559** | **0.6123** | 0.627 | **0.810** | 0.400 |
| **Linear SVM** | **SciFact** | 172 | **0.4128** | **0.3887** | 0.519 | **0.333** | 0.314 |

*Finding:* When evaluating claims in isolation, SCitance outperforms SciFact by over **22% in Macro-F1**. This is an artifact of synthetic negations: SCitance's contradicted claims contain explicit GPT-3.5 negation tokens (`not`, `fails to`), allowing the model to achieve an F1 of **0.789 - 0.810** on SCitance contradictions. On SciFact, where contradictions are natural, contradicted F1 collapses to **0.290 - 0.333**.

---

### 2.2. Condition C (Claim + Strict Evidence) Source Comparison

| Model | Source Cohort | Test Samples ($N$) | Accuracy | Macro-F1 | SUPPORTED F1 | CONTRADICTED F1 | NOT_ENOUGH_EVIDENCE F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **SCitance** | 93 | **0.5054** | **0.4014** | 0.500 | 0.633 | **0.071** |
| **Logistic Regression** | **SciFact** | 172 | **0.7616** | **0.7086** | 0.730 | 0.412 | **0.984** |
| **Linear SVM** | **SCitance** | 93 | **0.4731** | **0.4520** | 0.407 | 0.615 | **0.333** |
| **Linear SVM** | **SciFact** | 172 | **0.7326** | **0.6772** | 0.712 | 0.388 | **0.931** |

*Finding:* In Condition C, the dynamic flips entirely. SciFact reaches **76.16% Accuracy and 0.7086 Macro-F1**, while SCitance plummets to **50.54% Accuracy and 0.4014 Macro-F1**.

---

### 2.3. Condition E (Full Standardized Input) Source Comparison

| Model | Source Cohort | Test Samples ($N$) | Accuracy | Macro-F1 | SUPPORTED F1 | CONTRADICTED F1 | NOT_ENOUGH_EVIDENCE F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **SCitance** | 93 | 0.3978 | 0.3800 | 0.222 | 0.523 | 0.395 |
| **Logistic Regression** | **SciFact** | 172 | 0.5814 | 0.5396 | 0.644 | 0.256 | 0.718 |
| **Linear SVM** | **SCitance** | 93 | 0.4409 | 0.4290 | 0.333 | 0.571 | 0.382 |
| **Linear SVM** | **SciFact** | 172 | 0.5872 | 0.5544 | 0.636 | 0.338 | 0.690 |

---

## 3. Causal Diagnosis of Source Disparity

The dramatic performance gap between SCitance and SciFact is driven by **three structural factors**:

```
                       SOURCE DISPARITY DYNAMICS
                       
    Condition A (Claim-Only):         Condition C (Strict Evidence):
    SCitance >> SciFact              SciFact >> SCitance
    ┌─────────────────────────┐      ┌─────────────────────────┐
    │ SCitance has 38.7%      │      │ SciFact has extracted   │
    │ synthetic negations     │      │ sentence rationales;    │
    │ -> High shortcut F1     │      │ SCitance has 215-word   │
    │ (CON F1 = 0.810)        │      │ noisy full abstracts    │
    └─────────────────────────┘      └─────────────────────────┘
```

1. **Evidence Granularity Asymmetry:**
   * In SciFact, positive and contradicted claims have concise sentence-level rationales (mean ~57 words), and unsupported claims have explicit absent evidence markers (`[NO_EVIDENCE_PROVIDED]`), yielding near-perfect detection of `NOT_ENOUGH_EVIDENCE` (F1 = 0.984).
   * In SCitance, 100% of claims are paired with the **entire cited abstract** (mean 214 words). Because the abstract is full of background terms related to the claim, bag-of-words classifiers fail to recognize when evidence is missing, causing SCitance `NOT_ENOUGH_EVIDENCE` F1 to collapse to **0.071**.
2. **Claim Phrasing & Decontextualization:**
   * SciFact claims are concise, active assertions (mean 12.4 words) with direct verbs (`increases`, `decreases`).
   * SCitance citances are discursive sentences (mean 29.8 words) filled with metadiscourse (`"previous studies have demonstrated that..."`), which distracts linear bag-of-words models.

---

## 4. Research Implications for Future Phases

1. **Sentence-Level Rationale Retrieval is Mandatory for SCitance:**
   * SCitance cannot be effectively verified at the abstract level using bag-of-words models. In future phases, an automated sentence-retrieval module (e.g., BM25 or dense bi-encoder) must extract the top-2 most relevant sentences from SCitance abstracts before feeding them to the verifier.
2. **Unified Evaluation Reporting:**
   * Research publications should report breakdown tables by source origin to prevent high SciFact performance from masking low abstract-level citance performance.
