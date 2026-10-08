# CitationGuard AI — Source Bias, Structural Asymmetry & Dataset Fingerprinting Analysis
**Project:** CitationGuard AI — Evidence-Aware Citation Verification for Detecting Unsupported Academic Claims  
**Phase:** Phase 3 — Verification Quality Audit & ML-Readiness Protocol  
**Document:** `CitationGuard-Unified/source_bias_analysis.md`  
**Associated Artifact:** [`text_length_statistics.csv`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/text_length_statistics.csv)  
**Date:** October 8, 2026  

---

## 1. Executive Summary

When harmonizing disparate scientific NLP benchmarks, a critical risk is that machine learning models learn **dataset-specific artifacts** ("shortcuts" or "dataset fingerprints") rather than genuine citation-evidence reasoning. If an empirical classifier can predict veracity simply from the stylistic formatting or source origin of a claim, test metrics become deceptive.

This audit evaluates the two constituent datasets of CitationGuard's core verification view:
* **SCitance** ($N=649$, 37.00%) — Derived from real in-text citation sentences (*citances*) in S2ORC.
* **SciFact** ($N=1,105$, 63.00%) — Derived from human-written atomic scientific claims.

Our empirical investigation reveals **severe structural, lexical, and formatting asymmetries** between SCitance and SciFact, culminating in a **claim-only shortcut** where models achieve 55.76% accuracy without ever reading the cited evidence.

---

## 2. Quantitative Structural & Text Length Comparison

| Feature / Dimension | SCitance ($N=649$) | SciFact ($N=1,105$) | Structural Divergence / Risk |
| :--- | :---: | :---: | :--- |
| **Claim Mean Word Length** | **29.84 words** (median: 27.0) | **12.38 words** (median: 12.0) | $\mathbf{+141\%}$ longer in SCitance |
| **Claim Mean Character Length** | **200.78 chars** (median: 181.0) | **89.17 chars** (median: 82.0) | $\mathbf{+125\%}$ longer in SCitance |
| **Claim Max Word Length** | 94 words | 39 words | Extreme tail length in SCitance |
| **Citation Context Availability** | **100.0% (649/649)** | **0.0% (0/1,105)** | SciFact lacks raw citation context (`None`) |
| **Claim Identity with Context** | `claim_text == citation_context` (100%) | N/A | SCitance did not decompose citances |
| **Evidence Mean Word Length** | **213.75 words** (median: 170.0) | **57.67 words** (median: 45.0) | Document-level vs. Sentence-level |
| **Evidence Representation** | `evidence_text == paper_abstract` (100%) | Sentence Rationales (689) / Empty (416) | Fundamental representation disparity |
| **Empty Evidence Count** | 0 records (0.0%) | 416 records (37.65%) | 413 NEE + 3 extraction defects |
| **Title Word Length** | 13.23 words (median: 13.0) | 13.34 words (median: 13.0) | Harmonized (100% repaired in Phase 2) |
| **Abstract Word Length** | 213.75 words (median: 170.0) | 217.22 words (median: 172.0) | Harmonized biomedical abstracts |

*Full percentile distributions are archived in [`text_length_statistics.csv`](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/text_length_statistics.csv).*

---

## 3. Lexical Markers & Punctuation Patterns

Because SCitance extracted verbatim sentences from scientific literature while SciFact employed annotators to rewrite claims into atomic propositions, the two sources exhibit distinct surface markers:

### 3.1. Citation Markers & Attribution Frames
* **Bracket Numeric Citations (e.g., `[1]`, `[7, 8]`):**
  * SCitance: **6.16%** (40 claims).
  * SciFact: **0.00%** (0 claims).
* **Author-Year Citations (e.g., `(Smith et al., 2018)`):**
  * SCitance: **12.48%** (81 claims).
  * SciFact: **0.00%** (0 claims).
* **"et al." Discourse Markers:**
  * SCitance: **20.18%** (131 claims).
  * SciFact: **0.00%** (0 claims).
* **4-Digit Year Mentions (e.g., `2014`):**
  * SCitance: **20.18%** (131 claims).
  * SciFact: **0.81%** (9 claims).

### 3.2. Attribution Verbs vs. Declarative Assertions
* **SCitance Claims:** Characterized by epistemic framing and meta-discourse:
  * Top distinctive terms: `"shown"`, `"studies"`, `"recently"`, `"recent"`, `"reported"`, `"demonstrated"`, `"suggests"`.
  * Typical structure: *"Using manipulations that enhance cSMAC formation, it was shown that cSMAC formation could enhance signaling by weak ligands (Cemerski et al. 2007) ."*
* **SciFact Claims:** Characterized by direct causal, biophysical propositions:
  * Top distinctive terms: `"increases"`, `"reduces"`, `"decreases"`, `"regulates"`, `"induces"`, `"causes"`.
  * Typical structure: *"DUSP4 increases apoptosis."*

---

## 4. Empirical Proof of Dataset Fingerprinting

To quantify the degree of dataset fingerprinting, we trained a series of diagnostic probes:

### 4.1. Source Prediction Probe (SCitance vs. SciFact)
We trained a simple TF-IDF Logistic Regression classifier with unigram and bigram features to predict dataset origin (`SCitance` vs. `SciFact`) using **only the claim text**:
* **5-Fold Cross-Validation ROC-AUC:** **0.9533 $\pm$ 0.0075**
* **Top Weights Predicting SCitance:** `["et al", "was", "shown", "study", "not", "that", "these", "alvarez"]`
* **Top Weights Predicting SciFact:** `["increases", "reduces", "decreases", "regulates", "induces", "causes"]`

This demonstrates that a model can distinguish SCitance from SciFact claims with over **95% confidence** without seeing any evidence.

### 4.2. Label Distribution by Source (The Shortcut Risk)
When a model detects dataset origin, it gains immediate prior information about the target label:
* If the sample is from **SCitance**, the probability of `CONTRADICTED` is **38.67%**.
* If the sample is from **SciFact**, the probability of `CONTRADICTED` drops to **21.45%**, while `NOT_ENOUGH_EVIDENCE` jumps to **37.38%**.

Furthermore, **100% of the synthetic negations** in the entire verification dataset originate from SCitance (251 / 251). SciFact contains zero synthetic negations.

### 4.3. Claim-Only Hypothesis Probe (Cheating Without Evidence)
We evaluated whether a classifier can predict the 3-way verification label without looking at the evidence text or the cited abstract:

| Model Probe | Accuracy | Macro-F1 | Notes |
| :--- | :---: | :---: | :--- |
| **Majority Baseline** | 40.25% | 19.13% | Predicts `SUPPORTED` unconditionally |
| **Claim-Only TF-IDF Probe (All Claims)** | **55.76% $\pm$ 1.40%** | **54.53% $\pm$ 1.39%** | $\mathbf{+15.5\%}$ above baseline without evidence! |
| **Claim-Only Probe (Natural Claims Only)** | 59.41% | 44.72% | Macro-F1 drops sharply without synthetic cues |

**Interpretation:** On the full dataset, the presence of negation tokens (`not`, `does not`, `did not`) coupled with SCitance attribution markers allows a simple linear model to achieve **54.53% Macro-F1** purely from linguistic priors. When restricted to natural claims, Macro-F1 plummets to 44.72% because natural contradictions lack simplistic negation patterns.

---

## 5. Architectural Mitigation Plan for ML Training

To ensure models learn genuine evidence verification rather than exploiting source fingerprints:

1. **Surface Marker Stripping (Preprocessing):**
   * Automatically strip explicit citation markers (`\[\d+\]`, `\([A-Za-z\s]+,?\s+\d{4}\)`, `et al.`) from `claim_text` prior to model tokenization.
2. **Standardized Input Format:**
   * Do not provide `citation_context` as a separate input field, because its absence immediately flags SciFact.
   * Provide a unified input sequence:  
     `[CLS] Claim: {cleaned_claim} [SEP] Title: {paper_title} [SEP] Evidence: {evidence_text_or_abstract} [EOS]`
3. **Evidence Representation Standardization:**
   * In SciFact `NOT_ENOUGH_EVIDENCE` instances where `evidence_text` is `None`, models must be fed the **full paper abstract** so they evaluate whether the abstract contains evidence, rather than exploiting `None` as a trivial flag for `NOT_ENOUGH_EVIDENCE`.
4. **Mandatory Natural-Only Evaluation Slice:**
   * All benchmark results must report performance on the natural-only test partition ($N=228$) alongside the combined test set ($N=265$) to expose models relying on synthetic negation shortcuts.
