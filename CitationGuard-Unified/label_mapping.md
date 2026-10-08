# CitationGuard AI — Label Normalization & Semantic Ontology Report
**Project:** CitationGuard AI — Evidence-Aware Citation Verification for Detecting Unsupported Academic Claims  
**Module:** Label Architecture and Semantic Normalization  
**Document:** `CitationGuard-Unified/label_mapping.md`  
**Date:** October 8, 2026  

---

## 1. Principles of Label Normalization

A fundamental principle of the CitationGuard AI architecture is **semantic fidelity**. Research datasets must never be forced into a homogenized label space when their underlying annotation tasks reflect fundamentally different cognitive or linguistic phenomena.

We establish three distinct label fields in the master schema:
1. `original_label`: The verbatim, unmodified annotation from the source data file.
2. `normalized_label`: A standardized, task-specific categorical label representing clean task semantics.
3. `verification_label`: A controlled 3-class epistemic verification ontology (`SUPPORTED`, `CONTRADICTED`, `NOT_ENOUGH_EVIDENCE`) applied **strictly and exclusively** to empirical claim/citation verification tasks.

```
                    ┌───────────────────────────────────────────────┐
                    │               LABEL ONTOLOGY                  │
                    └───────────────────────────────────────────────┘
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
┌─────────────────────────────────┐                       ┌─────────────────────────────────┐
│     EPISTEMIC VERIFICATION      │                       │        RHETORICAL INTENT        │
│   (SciFact, SCitance, MSVEC)    │                       │       (SciCite, SciClaim)       │
├─────────────────────────────────┤                       ├─────────────────────────────────┤
│ normalized_label:               │                       │ normalized_label:               │
│   • SUPPORTED                   │                       │   • SciCite: background, method,│
│   • CONTRADICTED                │                       │              result             │
│   • NOT_ENOUGH_EVIDENCE         │                       │   • SciClaim: claim, evidence,  │
│                                 │                       │               none              │
│ verification_label:             │                       │                                 │
│   • SUPPORTED                   │                       │ verification_label:             │
│   • CONTRADICTED                │                       │   • null (NOT APPLICABLE)       │
│   • NOT_ENOUGH_EVIDENCE         │                       │                                 │
└─────────────────────────────────┘                       └─────────────────────────────────┘
```

---

## 2. Comprehensive Dataset Label Mapping Matrix

| Dataset | Source Field / Representation | Original Label | Normalized Label (`normalized_label`) | Verification Label (`verification_label`) | Mapping Nature | Scientific Rationale & Constraints |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **SCitance** | `evidence[doc_id][*].label` | `"SUPPORT"` | `SUPPORTED` | `SUPPORTED` | **Exact** | The cited paper abstract contains text directly supporting the in-text citation proposition. |
| **SCitance** | `evidence[doc_id][*].label` | `"CONTRADICT"` | `CONTRADICTED` | `CONTRADICTED` | **Exact** | The cited paper refutes the proposition. (Note: 100% of these in SCitance are synthetic LLM negations). |
| **SCitance** | `evidence == {}` (empty) | `"NO_EVIDENCE"` | `NOT_ENOUGH_EVIDENCE` | `NOT_ENOUGH_EVIDENCE` | **Exact** | The candidate paper lacks sufficient evidence to verify or refute the citation claim. |
| **SciFact** | `evidence[doc_id][*].label` | `"SUPPORT"` | `SUPPORTED` | `SUPPORTED` | **Exact** | The abstract contains sentence-level rationales that entail the atomic claim. |
| **SciFact** | `evidence[doc_id][*].label` | `"CONTRADICT"` | `CONTRADICTED` | `CONTRADICTED` | **Exact** | The abstract contains sentence-level rationales that refute the atomic claim. |
| **SciFact** | `evidence == {}` (empty) | `"NO_EVIDENCE"` | `NOT_ENOUGH_EVIDENCE` | `NOT_ENOUGH_EVIDENCE` | **Exact** | No rationale sentences in candidate abstracts verify or refute the claim. |
| **SciFact** | `claims_test.jsonl` (unlabeled) | `"UNLABELED"` | `UNLABELED` | `null` | **Exact** | SciFact test split has withheld labels for official leaderboard evaluation. |
| **MSVEC** | `evidence[doc_id][*].label` | `"SUPPORT"` | `SUPPORTED` | `SUPPORTED` | **Exact** | Academic paper rationale supports fact-checked claim. Preserved strictly for external evaluation. |
| **MSVEC** | `evidence[doc_id][*].label` | `"CONTRADICT"` | `CONTRADICTED` | `CONTRADICTED` | **Exact** | Academic paper rationale refutes fact-checked claim. Preserved strictly for external evaluation. |
| **MSVEC** | `evidence == {}` (empty) | `"NO_EVIDENCE"` | `NOT_ENOUGH_EVIDENCE` | `NOT_ENOUGH_EVIDENCE` | **Exact** | Corpus lacks evidence for fact-checked claim. |
| **SciCite** | `label` | `"background"` | `background` | `null` | **Task Specific** | **DO NOT MAP TO VERIFICATION.** Identifies that a citation provides background context. It has no truth value. |
| **SciCite** | `label` | `"method"` | `method` | `null` | **Task Specific** | **DO NOT MAP TO VERIFICATION.** Identifies citation to a software, algorithm, or tool. Factual veracity does not apply. |
| **SciCite** | `label` | `"result"` | `result` | `null` | **Task Specific** | **DO NOT MAP TO VERIFICATION.** Compares empirical findings; does not evaluate whether the comparison is supported. |
| **SciClaim** | `label` (col 0 in TSV) | `"1"` | `claim` | `null` | **Task Specific** | **DO NOT MAP TO VERIFICATION.** Sentence-role label indicating the sentence is a proposition asserting a finding. |
| **SciClaim** | `label` (col 0 in TSV) | `"2"` | `evidence` | `null` | **Task Specific** | **DO NOT MAP TO VERIFICATION.** Sentence-role label indicating the sentence provides experimental/theoretical proof. |
| **SciClaim** | `label` (col 0 in TSV) | `"0"` | `none` | `null` | **Task Specific** | **DO NOT MAP TO VERIFICATION.** Contextual sentence (method, future work, background). |

---

## 3. Strict Prohibitions Against Invalid Mappings

### 🚫 Prohibition 1: Conflating Citation Intent with Citation Support
In some naive data merging pipelines, researchers attempt to map:
* `background` $\rightarrow$ `NOT_ENOUGH_EVIDENCE` or `UNSUPPORTED`
* `method` $\rightarrow$ `SUPPORTED`
* `result` $\rightarrow$ `SUPPORTED`

**Methodological Fallacy:** Citing a background paper (e.g., *"Cancer immunotherapy has advanced rapidly [1]"*) is often completely accurate and supported by the cited reference. Flagging it as "unsupported" simply because its intent is "background" creates catastrophic false positives. SciCite labels model **rhetorical intent**, which is orthogonal to **epistemic truthfulness**.

### 🚫 Prohibition 2: Conflating Sentence Roles with Evidence Entailment
In SciClaim, label `2` is named `"Evidence"`. This does **not** mean that the sentence supports an external citation claim. It means that within that single paper, the authors presented an empirical measurement or mathematical equation to justify their own preceding claim. Therefore, `verification_label` is set to `null`.

---

## 4. Verification Label Distribution Across Master and Verification Views

In the primary verification view (`citationguard_verification.jsonl`, $N=1,754$), the distribution of `verification_label` is:

| Verification Label | Count | Percentage | Primary Semantic Interpretation |
| :--- | :---: | :---: | :--- |
| **`SUPPORTED`** | 726 | 41.39% | The citation claim or atomic proposition is entailed by the cited paper. |
| **`CONTRADICTED`** | 505 | 28.79% | The proposition is refuted by the cited paper (natural or synthetic). |
| **`NOT_ENOUGH_EVIDENCE`** | 523 | 29.82% | The cited paper does not contain sufficient evidence to decide. |
| **Total** | **1,754** | **100.0%** | **Balanced 3-class epistemic benchmark** |

*Note: 60 additional records with verification labels exist in the external test set (56 in MSVEC) and duplicate records.*
