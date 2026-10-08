# SciClaim Dataset  
## 科学主张识别数据集（SciClaim）

[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-orange)](LICENSE)  
[![Paper DOI](https://img.shields.io/badge/DOI-10.3724/2096--7004.di.2025.0009-brightgreen)](https://doi.org/10.3724/2096-7004.di.2025.0009)

> **English** | **中文**

---

## 📌 Overview / 项目概述

This repository releases the **SciClaim dataset**, a manually annotated corpus for **scientific claim recognition** in full-text scientific articles. The dataset was developed as part of the study [*"Scientific Claim Recognition via Staged Fine-Tuning with LoRA"* (Data Intelligence, 2025)](https://doi.org/10.3724/2096-7004.di.2025.0009).

本仓库开源 **SciClaim 数据集**——一个面向**科学主张识别**（Scientific Claim Recognition）任务的人工标注语料库，覆盖全文科学论文。该数据集源自[论文《基于 LoRA 分阶段微调的科学主张识别》（Data Intelligence, 2025）](https://doi.org/10.3724/2096-7004.di.2025.0009)。

Scientific claims are propositions presented as established facts in research papers (e.g., “Our method outperforms baselines by 15%”). Unlike prior work limited to abstracts, SciClaim includes annotations across **introduction, methods, results, and discussion** sections, enabling more comprehensive claim detection.

科学主张指论文中被表述为既定事实的命题（例如：“我们的方法比基线提升15%”）。与以往仅关注摘要的研究不同，SciClaim 的标注覆盖论文的**引言、方法、结果与讨论**等全文章节，支持更全面的主张识别。

---

## 📁 Dataset Structure / 数据结构

```
sciclaim/
├── SciClaim_sentences_train.tsv
├── SciClaim_sentences_test.tsv
├── SciClaim_sentences_test_withouT.tsv
└── SciClaim_sentences_val.tsv
```

## 🛠️ Dataset Construction / 数据集构建

The SciClaim dataset was constructed to support fine-grained recognition of scientific claims in **full-text scientific articles**, with a focus on the bio-agriculture domain. Rigorous annotation protocols and multi-expert review ensure high data quality and reliability.

SciClaim 数据集面向**全文科学论文**中的科学主张识别任务构建，聚焦生物农业领域。通过严格的标注规范与多专家复核机制，保障数据的高质量与可靠性。

### Annotation Guidelines / 标注标准

Each sentence is labeled into one of three categories:

| Label | Category      | Description |
|-------|---------------|-------------|
| `1`   | **Claim**     | Abstract statements expressing core findings, relationships, or comparisons. Examples:<br>• *“(A, X, property)”* – Statements of properties<br>• *“(A, B, relationship)”* – Statements of relationships<br>• *“(A, B, more effective)”* – Comparative claims |
| `2`   | **Evidence**  | Concrete support for claims:<br>• **Experimental**: Quantitative results, measurements<br>• **Theoretical**: References to models, formulas, or established theories |
| `0`   | **None-type** | Contextual content not constituting a claim or evidence:<br>• Research background<br>• Objectives<br>• Method descriptions<br>• Future work suggestions |

> 💡 Note: In the released QA-format dataset, we map this 3-class labeling to a binary task (`"yes"` for Claim, `"no"` for Evidence + None-type), as the primary goal is **claim identification**.

### Annotation Process & Statistics / 标注流程与数据统计

- **Source**: 15 full-text scientific papers in bio-agriculture (2,391 sentences total)  
- **Annotators**: 3 trained experts per sentence  
- **Consensus**: Final label assigned if ≥2 annotators agree; otherwise, resolved via expert discussion  
- **Split**: By paper (not by sentence) to avoid data leakage:
  - **Train**: 11 papers (1,657 sentences)
  - **Validation**: 2 papers (410 sentences)
  - **Test**: 2 papers (324 sentences)

**Label Distribution**

| Split     | Papers | Claims (1) | Evidence (2) | None-type (0) | Total |
|-----------|--------|------------|--------------|----------------|-------|
| Train     | 11     | 482        | 375          | 800            | 1,657 |
| Val       | 2      | 111        | 132          | 167            | 410   |
| Test      | 2      | 77         | 85           | 162            | 324   |
| **Total** | **15** | **670**    | **592**      | **1,129**      | **2,391** |

**Inter-Annotator Agreement**  
Cohen’s Kappa = **0.953** (based on pairwise agreement over 2,391 sentences), indicating **nearly perfect agreement** and high annotation reliability.

### Annotation Examples / 标注示例

| Label | Sentence | Explanation |
|-------|--------|-------------|
| **1** (Claim) | *“The structural components of epidermis, vascular tissue, and sclerenchyma have low digestibilities, or can even be indigestible.”* | **Property statement** – Describes inherent attributes of biological structures |
| **1** (Claim) | *“These findings support the hypothesis that EFE can degrade cell wall structures, thereby allowing ruminal microbes earlier access…”* | **Relationship statement** – Links EFE activity to microbial access |
| **2** (Evidence) | *“S-CL activity… was increased by 22.44%, 5.83%, and 5.70% at 30th, 60th, and 90th days…”* | **Experimental evidence** – Quantitative results |
| **2** (Evidence) | *“The increase… is consistent with the Adsorption-Degradation Model.”* | **Theoretical evidence** – Cites an established model |
| **0** (None) | *“Further studies on retention of crop residues… are recommended…”* | **Future work** – Not a claim or evidence |
| **0** (None) | *“We conducted a half-year straw decomposition assay…”* | **Method description** – Procedural detail |

> These examples reflect how scientific reasoning is structured in real research articles and guided our annotation decisions.

---

## 📄 Citation / 引用

If you use the SciClaim dataset in your research, please cite our open-access paper:

若你在研究中使用 SciClaim 数据集，请引用我们的开放获取论文：

**English (BibTeX):**

```bibtex
@article{lin2025sciclaim,
  author    = {Xin Lin and Yajiao Wang and Zhixiong Zhang and others},
  title     = {Scientific Claim Recognition via Staged Fine-Tuning with LoRA},
  journal   = {Data Intelligence},
  year      = {2025},
  volume    = {7},
  number    = {2},
  pages     = {303--335},
  note      = {Open Access},
  url       = {https://www.sciengine.com/doi/10.3724/2096-7004.di.2025.0009},
  urldate   = {2025-10-11}
}
```

---

## 📝 License / 许可证

- **SciClaim Dataset**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)  
  → You are free to share and adapt the dataset for **non-commercial purposes**, as long as you give appropriate credit.

- **科学主张数据集**：[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)  
  → 允许**非商业用途**下的共享与改编，须注明原作者。

---

## ❓ Questions? / 问题反馈

For questions about the dataset, please open an issue on GitHub.

如有疑问，欢迎在 GitHub 提交 Issue。

---
