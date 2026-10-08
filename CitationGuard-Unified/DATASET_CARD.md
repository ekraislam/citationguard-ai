---
annotations_creators:
- expert-annotated
- machine-generated (synthetic negations tracked)
language:
- en
license:
- cc-by-nc-4.0
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- SCitance
- SciFact
- SciCite
- SciClaim
- MSVEC
task_categories:
- text-classification
- natural-language-inference
- question-answering
task_ids:
- scientific-claim-verification
- citation-verification
- citation-intent-classification
- sentence-role-detection
pretty_name: CitationGuard AI Unified Research Dataset
---

# Dataset Card for CitationGuard AI

## 1. Dataset Name
**CitationGuard AI — Unified Research Dataset for Evidence-Aware Citation Verification**  
*Curated by:* CitationGuard AI Research Project  
*Release Date:* October 2026  
*Version:* 1.0.0  

## 2. Motivation
In scholarly publishing, unsupported citations and citation distortions erode scientific reproducibility. Automated verification of citations requires models that can determine whether an in-text citation sentence (*citance*) is factually supported, contradicted, or unaddressed by the cited scientific paper. Prior research datasets addressed fragments of this challenge in isolation—atomic claim verification (SciFact), raw citation intent (SciCite), sentence role labeling (SciClaim), or isolated citance verification (SCitance)—often suffering from metadata corruption, synthetic biases, or cross-dataset data leakage. CitationGuard AI provides the first harmonized, leakage-free, multi-task benchmark uniting these resources into a scientifically defensible master dataset.

## 3. Research Objective
The objective is to train and evaluate evidence-aware citation verification systems capable of:
1. Distinguishing between citations asserting empirical findings versus those providing methodological or background context.
2. Identifying atomic verifiable propositions within natural citation contexts.
3. Retrieving sentence-level rationales from cited literature abstracts.
4. Classifying citation veracity into `SUPPORTED`, `CONTRADICTED`, or `NOT_ENOUGH_EVIDENCE`.

## 4. Source Datasets
CitationGuard AI harmonizes 5 premier scientific NLP datasets:
1. **SCitance** (*Alvarez et al., ACL SDP 2024*): In-text citation verification using authentic citances.
2. **SciFact** (*Wadden et al., EMNLP 2020*): Gold-standard biomedical claim verification with sentence rationales.
3. **SciCite** (*Cohan et al., NAACL 2019*): Citation intent classification with structural scaffolds.
4. **SciClaim** (*Lin et al., Data Intelligence 2025*): Full-text sentence-level claim and evidence recognition.
5. **MSVEC** (*MultiVerS Benchmark*): Multi-domain fact-checked claims from Snopes and PolitiFact.

## 5. Data Composition
* **Master Dataset Size:** **15,525 records** (`citationguard_master.jsonl` / `citationguard_master.csv`).
* **Core Verification View:** **1,754 records** (`citationguard_verification.jsonl` / `citationguard_verification.csv`).
* **Intent Classification View:** **11,020 records** (`citationguard_intent.csv`).
* **Claim Detection View:** **2,391 records** (`citationguard_claim_detection.csv`).
* **External Robustness View:** **56 records** (`citationguard_external_test.csv`).

## 6. Unified Master Schema
Every sample is represented by the 23-field master schema:
* `record_id`: Unique identifier (e.g., `CG_000001`).
* `source_dataset`: One of `SCitance`, `SciFact`, `SciCite`, `SciClaim`, `MSVEC`.
* `source_file`: Relative path to source file.
* `source_record_id`: Original identifier from source dataset.
* `task_type`: Categorical task (`citation_verification`, `scientific_claim_verification`, `citation_intent`, `sentence_role_detection`, `external_robustness_verification`).
* `claim_text`: Formulated claim or sentence text.
* `citation_context`: In-text citation context string (if applicable).
* `evidence_text`: Ground-truth sentence rationale text from abstract.
* `paper_id`: Primary document/paper identifier.
* `document_id`: Integer corpus document ID.
* `paper_title`: Verified title of cited paper.
* `paper_abstract`: Full abstract text of cited paper.
* `original_label`: Exact unmodified label from source dataset.
* `normalized_label`: Standardized task label.
* `verification_label`: 3-class epistemic label (`SUPPORTED`, `CONTRADICTED`, `NOT_ENOUGH_EVIDENCE` or `null`).
* `original_split`: Split designated in source dataset (`train`, `dev`, `val`, `test`).
* `guard_split`: Leakage-safe split (`TRAIN`, `DEV`, `TEST`, `EXTERNAL_TEST`, `TEST_UNLABELED`).
* `evidence_sentence_ids`: Sentence index array for rationales.
* `synthetic_or_natural`: Provenance flag (`NATURAL`, `SYNTHETIC_NEGATION`, `UNKNOWN`).
* `duplicate_group_id`: Unique cluster ID for duplicate/near-duplicate grouping.
* `duplicate_status`: `UNIQUE`, `EXACT_DUPLICATE`, `VERIFIED_NEAR_DUPLICATE`, `SOURCE_OVERLAP`.
* `source_provenance`: Full academic citation and lineage.
* `notes`: Detailed operational metadata.

## 7. Label Semantics
The label space is divided into two mutually exclusive ontological planes:
1. **Epistemic Verification Plane:** `SUPPORTED`, `CONTRADICTED`, `NOT_ENOUGH_EVIDENCE`. Applied only to verification tasks.
2. **Rhetorical Intent Plane:** SciCite labels (`background`, `method`, `result`) and SciClaim labels (`claim`, `evidence`, `none`) are preserved in their respective tasks. They are **never** mapped into verification labels.

## 8. Cleaning Procedure
* SciClaim title prefixes `(title:<Title>) ` were parsed using regular expressions, populating `paper_title` and restoring clean sentence text.
* The corrupted test file `SciClaim_sentences_test_withouT.tsv` (exhibiting 22 conflicting labels) was permanently excluded. Only the verified `SciClaim_sentences_test.tsv` was ingested.
* SciFact's unlabeled test set claims were isolated as `TEST_UNLABELED`.

## 9. Deduplication Procedure
A 6-level deduplication protocol was executed. 102 exact internal duplicates were marked `EXACT_DUPLICATE` and removed from the clean verification view. 43 cross-dataset source overlaps and 25 lexical near-duplicates (Jaccard $\ge 0.70$) were linked under `duplicate_group_id` with `duplicate_status = 'SOURCE_OVERLAP'`, preserving representation diversity without evaluation leakage.

## 10. Leakage Prevention Methodology
To eliminate the 86.17% document-level overlap between SciFact and SCitance:
1. An undirected document connectivity graph was constructed across all cited documents.
2. 603 connected paper components were extracted via BFS.
3. Components were deterministically partitioned into `TRAIN` (417 components, 1,226 claims), `DEV` (89 components, 263 claims), and `TEST` (97 components, 265 claims).
4. Cross-split document leakage is mathematically verified to be **0.0%**.

## 11. Metadata Repair
In SCitance, 100% of the corpus documents (435/435) had their `title` field corrupted with `abstract[0]`. All 435 true titles were restored via relational key matching against the verified SciFact corpus metadata.

## 12. Synthetic Data Handling
251 claims in SCitance (38.67%) are synthetic LLM negations generated by GPT-3.5. These claims are explicitly flagged with `synthetic_or_natural = 'SYNTHETIC_NEGATION'`. In model evaluations, performance must be reported separately for natural vs. synthetic samples.

## 13. Train / Dev / Test Split Methodology
* **Verification View:** 70% Train (1,226), 15% Dev (263), 15% Test (265). Strictly grouped by Paper ID.
* **Citation Intent View:** Train (8,243), Dev (916), Test (1,861).
* **Claim Detection View:** Train (1,657), Dev (410), Test (324).
* **External Robustness View:** 100% External Test (56).

## 14. Known Limitations
1. Domain concentration: Core verification data is predominantly biomedical (derived from S2ORC / PubMed).
2. Synthetic contradictions: Genuine natural refutations in academic writing are rare; 51.6% of `CONTRADICTED` instances in the verification view rely on synthetic negations.
3. Abstract-level scope: Evidence is evaluated against abstracts rather than full-text PDF bodies.

## 15. Intended Use
* Research into citation verification, scientific claim checking, and hallucination detection in academic LLMs.
* Fine-tuning evidence-aware retrieval and natural language inference models.

## 16. Prohibited Misuse
* Never merge citation intent labels (`method`/`background`) into verification labels.
* Never evaluate models using standard random splits on SciFact/SCitance without paper grouping.
* Never use the external test set (MSVEC) for model training or prompt engineering.

## 17. Provenance Policy
All source records retain original identifiers, original labels, source file paths, and bibliographic citations in `source_provenance` and `notes`.

## 18. Reproducibility Instructions
To rebuild the entire unified dataset from scratch, run:
```bash
python build_citationguard_dataset.py
```
Execution is deterministic (Seed 42), requires only standard Python libraries with NumPy and SciPy, and generates all master artifacts and specialized views.
