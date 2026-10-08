# CitationGuard AI — Final Verification Duplicate & Near-Duplicate Recheck
**Project:** CitationGuard AI — Evidence-Aware Citation Verification for Detecting Unsupported Academic Claims  
**Phase:** Phase 3 — Verification Quality Audit & ML-Readiness Protocol  
**Document:** `CitationGuard-Unified/verification_duplicate_recheck.md`  
**Date:** October 8, 2026  

---

## 1. Executive Summary

A core objective of the Phase 2 unification pipeline was the complete eradication of duplicate records from the core verification view (`citationguard_verification.jsonl`, $N=1,754$). This report executes an independent, deterministic recheck to audit:
1. Exact claim duplicates
2. Normalized claim duplicates (case-folded, punctuation-stripped, whitespace-normalized)
3. Exact citation-context duplicates
4. Joint `(claim_text, evidence_text)` pair duplicates
5. High-similarity near-duplicate claim pairs (Jaccard token similarity $\ge 0.70$)

The recheck confirms that **zero exact or normalized duplicates exist** within the final verification view. Furthermore, a deep investigation of 510 high-similarity near-duplicate pairs identifies the exact structure of natural vs. synthetic negation pairings, and detects **7 cross-split near-duplicate claim pairs** arising from independent research papers on identical scientific topics.

---

## 2. Deterministic Exact Duplicate Audit Matrix

| Duplicate Category | Audit Formulation | Observed Instances | Permissible Threshold | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Exact Claim Duplicates** | $\text{cardinality}(\{r[\text{"claim\_text"}]\}) == N$ | **0 duplicates** (1,754 unique) | 0 | **CLEAN** |
| **Normalized Claim Duplicates** | $\text{cardinality}(\{\text{norm}(r[\text{"claim\_text"}])\}) == N$ | **0 duplicates** (1,754 unique) | 0 | **CLEAN** |
| **Exact Citation-Context Duplicates** | Non-empty citation context uniqueness | **0 duplicates** (649 unique) | 0 | **CLEAN** |
| **Claim + Evidence Pair Duplicates** | $\text{cardinality}(\{(r[\text{"claim"}], r[\text{"evidence"}])\}) == N$ | **0 duplicates** (1,754 unique) | 0 | **CLEAN** |
| **Duplicate Group Status** | Clean records with status `EXACT_DUPLICATE` | **0 records** (100% excluded) | 0 | **CLEAN** |

All 102 exact duplicate instances identified during Phase 2 were successfully isolated in `citationguard_master.csv` and completely excluded from `citationguard_verification.csv`.

---

## 3. High-Similarity Near-Duplicate Analysis (Jaccard $\ge 0.70$)

Using deterministic set tokenization on normalized claim texts, we computed all pairwise Jaccard similarities:
$$J(C_i, C_j) = \frac{|T(C_i) \cap T(C_j)|}{|T(C_i) \cup T(C_j)|}$$

Across all $\binom{1754}{2} = 1,537,381$ pairs, exactly **510 pairs** exhibited $J(C_i, C_j) \ge 0.70$.

### 3.1. Structure of the 510 Near-Duplicate Pairs
1. **Within-Split Synthetic Negation Pairs ($n = 488$ pairs):**
   * Almost all within-split near duplicates represent a natural parent claim and its GPT-3.5 generated synthetic negation citing the same paper.
   * *Example:*
     * `CG_000001` (TRAIN, SUPPORTED): *"Using manuipulations that enhance cSMAC formation, it was shown that cSMAC formation could enhance signaling by weak ligands (Cemerski et al. 2007) ."*
     * `CG_000250` (TRAIN, CONTRADICTED): *"It was shown that cSMAC formation could not enhance signaling by weak ligands despite using manipulations that enhance cSMAC formation (Cemerski et al. 2007)."* ($J = 0.810$).
   * Because both the positive and negative claim reside in the same split (`TRAIN`), they provide healthy paired contrastive training signals without leaking into evaluation splits.
2. **Cross-Source Near-Duplicate Pairs ($n = 25$ pairs):**
   * These pairs consist of an authentic SCitance citance and an independently formulated SciFact claim derived from the same underlying literature.
   * *Example:* `CG_000004` (SCitance, TRAIN) vs. `CG_001724` (SciFact, TRAIN) ($J = 0.773$).
   * All 25 pairs were assigned to the **same split** (`TRAIN`), preserving cross-dataset representation diversity with zero evaluation leakage.
3. **Cross-Split Near-Duplicate Pairs ($n = 7$ pairs):**
   * Exactly 7 pairs span across split boundaries (`TRAIN` $\leftrightarrow$ `DEV`, `TRAIN` $\leftrightarrow$ `TEST`).
   * Below is a complete catalog and causal audit of all 7 pairs.

---

## 4. Complete Audit of the 7 Cross-Split Near-Duplicate Pairs

| Pair # | Record 1 (Split, Source, Label) | Record 2 (Split, Source, Label) | Jaccard Similarity | Shared Paper ID? | Causal Mechanism |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | `CG_000206` (DEV, SCitance, SUPPORTED) | `CG_000683` (TRAIN, SciFact, SUPPORTED) | **0.9565** | **NO** (26996935 vs. 3512154) | Identical CRISPR bias claim cited to two different papers in literature |
| **2** | `CG_000461` (DEV, SCitance, CONTRADICTED) | `CG_000683` (TRAIN, SciFact, SUPPORTED) | **0.8400** | **NO** (26996935 vs. 3512154) | Synthetic negation of Pair 1 in DEV vs. SciFact claim in TRAIN |
| **3** | `CG_000673` (TRAIN, SciFact, SUPPORTED) | `CG_001463` (TEST, SciFact, NOT_ENOUGH_EVIDENCE) | **0.7000** | **NO** (11705328 vs. 5152028) | Parallel bio-claims: Folate deficiency vs. Vitamin B12 deficiency on homocysteine |
| **4** | `CG_000792` (TRAIN, SciFact, CONTRADICTED) | `CG_000914` (DEV, SciFact, NOT_ENOUGH_EVIDENCE) | **0.7273** | **NO** (8447873 vs. 12209494) | Parallel treatment claims: Charcoal vs. Gastric lavage for paraquat poisoning |
| **5** | `CG_001105` (DEV, SciFact, SUPPORTED) | `CG_001627` (TRAIN, SciFact, NOT_ENOUGH_EVIDENCE) | **0.8000** | **NO** (8246922 vs. 40632104) | Parallel phenotypic claims: Mice without IFN-$\gamma$ susceptible vs. resistant to EAM |
| **6** | `CG_001208` (TEST, SciFact, CONTRADICTED) | `CG_001209` (TRAIN, SciFact, SUPPORTED) | **0.7273** | **NO** (46695481 vs. 27446873) | Opposing diagnostic claims on HPV cytology sensitivity cited to two distinct clinical trials |
| **7** | `CG_001209` (TRAIN, SciFact, SUPPORTED) | `CG_001668` (TEST, SciFact, SUPPORTED) | **0.8095** | **NO** (27446873 vs. 46695481) | Parallel clinical trial claims on HPV cytology sensitivity across two independent studies |

### Scientific Finding on Semantic Cross-Split Leakage:
1. **Paper-Level Isolation is Strictly Maintained:** In all 7 cases, Record 1 and Record 2 cite **completely different research papers** (different `paper_id` and different abstracts). The graph connected component clustering correctly isolated the cited documents.
2. **Domain-Level Semantic Proximity:** The overlap arises because distinct scientific groups independently publish papers addressing identical research hypotheses (e.g., HPV screening efficacy, homocysteine metabolism, CRISPR spacer acquisition).
3. **Audit Verdict:** These 7 pairs do **not** represent data processing bugs or document leakage; they represent authentic, organic semantic commonalities in scientific biomedical research. However, in Pairs 1 & 2 (CRISPR spacer bias) and Pairs 6 & 7 (HPV screening), models trained on TRAIN will encounter nearly identical propositional phrasing in DEV and TEST.
4. **Actionable Directive:** In accordance with Phase 3 instructions ("Do NOT remove anything in this phase"), these records remain in the dataset. However, benchmarking protocols should report whether model performance differs on these specific probe records.
