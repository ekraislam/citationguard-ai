# CitationGuard AI — Multi-Level Deduplication Report
**Project:** CitationGuard AI — Evidence-Aware Citation Verification for Detecting Unsupported Academic Claims  
**Module:** Deduplication and Redundancy Audit  
**Document:** `CitationGuard-Unified/deduplication_report.md`  
**Date:** October 8, 2026  

---

## 1. Executive Summary

A critical failure mode in multi-dataset scientific NLP pipelines is silent duplicate inflation, where the same claims, citation contexts, or document rationales appear repeatedly across training and test splits. This report documents the **6-level deduplication protocol** implemented in CitationGuard AI Phase 2.

* **Total Records Audited:** 15,525 records across 5 source datasets.
* **Exact Internal Duplicates Identified:** 102 records marked `EXACT_DUPLICATE`.
* **Cross-Dataset Source Overlaps Identified:** 43 records marked `SOURCE_OVERLAP`.
* **Lexical Near-Duplicates Flagged (Jaccard $\ge 0.70$):** 25 cross-dataset pairs.
* **Master Dataset Policy:** All records are preserved in `citationguard_master.jsonl` with explicit `duplicate_group_id` and `duplicate_status` for full scientific provenance and auditability.
* **Verification View Policy:** Exactly duplicated rows are excluded from `citationguard_verification.jsonl` ($N=1,754$), ensuring an unpolluted machine-learning benchmark.

---

## 2. The 6-Level Deduplication Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       6-LEVEL DEDUPLICATION HIERARCHY                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Level 1: Exact Record Duplication (Identical IDs, identical text & metadata)│
│ Level 2: Exact Normalized Text Duplication (Case/punct/space normalized)    │
│ Level 3: Cross-Dataset Duplicate Claims (SciFact claim == SCitance claim)   │
│ Level 4: Duplicate Citation Contexts (Identical citing contexts in SciCite) │
│ Level 5: Duplicate Document Content (Shared abstracts between SciFact/MSVEC)│
│ Level 6: Near-Duplicate Claims (Token Jaccard ≥ 0.70 / Paraphrases / Minimal│
│          Pairs / Synthetic Negations)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Categorical Duplicate Statuses

| Duplicate Status | Definition | Handling in Master Dataset | Handling in Verification View |
| :--- | :--- | :--- | :--- |
| **`UNIQUE`** | No identical or near-duplicate counterpart identified. | Retained | Retained |
| **`EXACT_DUPLICATE`** | Byte-for-byte or normalized match of an existing record within the same source dataset. | Retained with provenance | **Removed** (102 instances) |
| **`SOURCE_OVERLAP`** | Record sharing semantic origin or high lexical overlap across different datasets. | Retained with cross-reference | Retained (both representations valid) |
| **`POSSIBLE_NEAR_DUPLICATE`** | Lexical overlap $\ge 70\%$, often minimal pairs or negation variants. | Retained with cross-reference | Retained |

---

## 3. Quantitative Deduplication Breakdown

### 3.1. Overview Statistics

| Metric | Master Dataset Count | Verification View Count |
| :--- | :---: | :---: |
| **Total Input Records** | 15,525 | 1,856 (candidate verification) |
| **Unique Records (`UNIQUE`)** | 15,380 | 1,711 |
| **Cross-Dataset Overlaps (`SOURCE_OVERLAP`)** | 43 | 43 |
| **Exact Internal Duplicates (`EXACT_DUPLICATE`)** | 102 | 102 (excluded) |
| **Final Exported Records** | **15,525** | **1,754** |

### 3.2. Deduplication Breakdown by Source Dataset

| Dataset | Total Records | Unique Records | Exact Duplicates | Cross-Dataset Overlaps | Primary Duplicate Sources |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SCitance** | 649 | 631 | 0 | 18 | Near-duplicate citances matching SciFact claims |
| **SciFact** | 1,409 | 1,302 | 82 | 25 | Internal train duplicates (e.g. ID 85 vs 86) & SciFact test dups |
| **SciCite** | 11,020 | 11,019 | 1 | 0 | Single duplicated citation string in `train.jsonl` |
| **SciClaim** | 2,391 | 2,372 | 19 | 0 | Repeated figure captions/headers in `val.tsv` |
| **MSVEC** | 56 | 56 | 0 | 0 | All claims unique |
| **Total** | **15,525** | **15,380** | **102** | **43** | |

---

## 4. Analysis of Concrete Duplicate Cases

### Case 1: Exact Duplicate in SciFact (Level 1 & 2)
In `data/data/claims_train.jsonl`:
* Record `CG_000734` (SciFact ID 85): `"Adult tissue-resident macrophages are seeded before birth."`
  * Evidence: `{'7521113': [{'sentences': [4], 'label': 'SUPPORT'}], '22406695': [{'sentences': [1], 'label': 'SUPPORT'}]}`
  * Status: `UNIQUE`
* Record `CG_000735` (SciFact ID 86): `"Adult tissue-resident macrophages are seeded before birth."`
  * Evidence: `{'7521113': [{'sentences': [4], 'label': 'SUPPORT'}], '22406695': [{'sentences': [1], 'label': 'SUPPORT'}]}`
  * Status: `EXACT_DUPLICATE`  
  * Action: Excluded from `citationguard_verification.jsonl` to prevent weight double-counting during model fine-tuning.

### Case 2: Cross-Dataset Source Overlap (Level 3 & 6)
SCitance and SciFact share identical underlying S2ORC source papers:
* **SCitance ID 31:** `"Recently a strong bias in the phage genome locations where the spacers were derived has been observed in many CRISPR subtypes that confer the immunity to phage ."`
* **SciFact ID 47:** `"A strong bias in the phage genome locations where the spacers were derived has been observed in many CRISPR subtypes that confer the immunity to phage."`
* **Jaccard Similarity:** 0.96.
* **Distinction:** SCitance preserves the authentic citation discourse opener (*"Recently"*) and punctuation formatting; SciFact presents the decontextualized proposition.
* **Resolution:** Both records are retained in the master dataset, linked under `duplicate_status = 'SOURCE_OVERLAP'`, and guaranteed to be placed into the **same guard split** to avoid cross-split evaluation leakage.

### Case 3: Minimal Pairs (Claim vs. Negation)
A high lexical Jaccard similarity ($\ge 0.70$) frequently occurs when a natural claim is paired with its synthetic negation:
* **Claim:** *"40mg/day dosage of folic acid and 2mg/day dosage of vitamin B12 does not affect chronic kidney disease progression."*
* **Negation:** *"40mg/day dosage of folic acid and 2mg/day dosage of vitamin B12 affects chronic kidney disease progression."*
* **Jaccard Similarity:** 0.88.
* **Resolution:** These are **NOT duplicates**. They represent contrastive factual pairs essential for stance classification. They are categorized as `UNIQUE` with complementary stance labels.
