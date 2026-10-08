# CitationGuard AI — Data Cleaning and Preprocessing Report
**Project:** CitationGuard AI — Evidence-Aware Citation Verification for Detecting Unsupported Academic Claims  
**Module:** Data Sanitization, Parsing, and Anomaly Remediation  
**Document:** `CitationGuard-Unified/data_cleaning_report.md`  
**Date:** October 8, 2026  

---

## 1. Executive Summary

This report documents the rigorous data cleaning, anomaly correction, and exclusion protocols executed during Phase 2 of the CitationGuard AI dataset pipeline. All preprocessing steps were deterministic, auditable, and executed strictly within `build_citationguard_dataset.py` without modifying any original source files.

---

## 2. Key Cleaning Interventions

### 2.1. SciClaim Test Set Anomaly Remediation (Rule 11)
* **The Problem:** The SciClaim repository contains two competing test files:
  1. `SciClaim_sentences_test.tsv` (83,070 bytes, 324 records)
  2. `SciClaim_sentences_test_withouT.tsv` (55,678 bytes, 324 records)
  Phase 1 audit revealed that **22 sentences** between these two files have directly conflicting labels (e.g. `1` vs `2`, `1` vs `0`, `2` vs `1`). The official published paper (Lin et al., *Data Intelligence*, 2025) and repository `README.md` report exact label counts: Claims (1): 77, Evidence (2): 85, None (0): 162. These counts match `SciClaim_sentences_test.tsv` exactly, while `SciClaim_sentences_test_withouT.tsv` is corrupted (`{'0': 164, '2': 83, '1': 77}`).
* **Action Taken:** `SciClaim_sentences_test_withouT.tsv` was **strictly and permanently excluded** from the CitationGuard pipeline. Only the verified `SciClaim_sentences_test.tsv` was ingested.

### 2.2. SciClaim Text Prefix Parsing and Title Extraction
* **The Problem:** Every sentence in `SciClaim_sentences_train.tsv`, `SciClaim_sentences_val.tsv`, and `SciClaim_sentences_test.tsv` had a paper title prepended as a string prefix:
  `"(title:Characterization of a newly Isolated Bacterium Pandoraea sp. B-6...) A newly isolated bacterium was screened out..."`
  Leaving this artifact in the text would distort tokenizers, positional embeddings, and n-gram analyses.
* **Action Taken:** A regular expression parser extracted the title substring into the schema field `paper_title`, while the sentence proposition was cleanly stripped and assigned to `claim_text`:
  ```python
  if raw_text.startswith("(title:"):
      parts = raw_text.split(")", 1)
      paper_title = parts[0].replace("(title:", "").strip()
      cleaned_text = parts[1].strip() if len(parts) > 1 else raw_text
  ```
  Result: 2,391 sentences cleanly parsed; 15 unique paper titles cataloged.

### 2.3. SCitance Corpus Title Restoration (Rule 5)
* **The Problem:** In `scitance-main/scitance-main/data/scitance/corpus.jsonl`, all 435 documents had their `title` field corrupted with the first sentence of the abstract (`abstract[0]`).
* **Action Taken:** An exact relational join on `doc_id` against SciFact's verified `corpus.jsonl` restored 100% of the true paper titles. Detailed in [scitance_metadata_repair_report.md](file:///c:/Users/hp/Desktop/CitationGuard-Datasets/CitationGuard-Unified/scitance_metadata_repair_report.md).

### 2.4. SCitance Synthetic Negation Provenance Tagging (Rule 10)
* **The Problem:** In SCitance, 38.67% of the claims are synthetically generated negations produced by prompting GPT-3.5 (`text-davinci-003`). Silently mixing synthetic negations with natural contradictions introduces artificial grammatical artifacts (e.g. repetitive "fails to", "does not").
* **Action Taken:** Every claim in SCitance was cross-referenced against `data/negations/negations.json`. Claims matching the synthetic negation dictionary were explicitly tagged with `synthetic_or_natural = 'SYNTHETIC_NEGATION'`. Natural in-text citation sentences were tagged as `NATURAL`.

### 2.5. SciFact Hidden Test Set Isolation
* **The Problem:** `data/data/claims_test.jsonl` contains 300 claims with no evidence annotations (`evidence: {}`) and no candidate document IDs (`cited_doc_ids: []`) because it was created for an external evaluation leaderboard.
* **Action Taken:** Rather than discarding these claims or fabricating missing evidence, these 300 records are preserved in the master dataset with `original_split = 'test'`, `guard_split = 'TEST_UNLABELED'`, `verification_label = null`, and `notes = 'SciFact hidden evaluation test set (unlabeled)'`. They are strictly excluded from the supervised verification view (`citationguard_verification.jsonl`).

---

## 3. Cleaning Summary Table

| Source Dataset | Input File(s) | Records Audited | Cleaning Action Taken | Records Retained in Master |
| :--- | :--- | :---: | :--- | :---: |
| **SciClaim** | `SciClaim_sentences_train.tsv`<br>`SciClaim_sentences_val.tsv`<br>`SciClaim_sentences_test.tsv` | 2,391 | Extracted `paper_title` prefix; normalized sentence text; excluded corrupted `test_withouT.tsv` | **2,391** |
| **SCitance** | `train.jsonl`<br>`dev.jsonl`<br>`test.jsonl`<br>`corpus.jsonl` | 649 claims<br>435 docs | Restored 435 true paper titles from SciFact; tagged 251 synthetic negations | **649** |
| **SciFact** | `claims_train.jsonl`<br>`claims_dev.jsonl`<br>`claims_test.jsonl` | 1,409 | Deduplicated internal exact duplicates; linked sentence rationales; marked test as `TEST_UNLABELED` | **1,409** |
| **MSVEC** | `msvec.json`<br>`corpus.jsonl` | 56 | Linked candidate abstracts; marked as `EXTERNAL_TEST` | **56** |
| **SciCite** | `train.jsonl`<br>`dev.jsonl`<br>`test.jsonl` | 11,020 | Cleaned citation intent strings; preserved rhetorical labels without mapping to verification | **11,020** |
| **Total** | | **15,525** | **Zero loss of valid research data** | **15,525** |
