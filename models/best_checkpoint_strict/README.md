# CitationGuard AI — Model Checkpoint

This directory contains the model architecture configuration and tokenizer for **CitationGuard AI (Phase 5 Strict Evidence Protocol)**.

### Architecture Details
- **Base Model:** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
- **Classification Head:** 3-class linear classifier (`SUPPORTED`, `CONTRADICTED`, `NOT_ENOUGH_EVIDENCE`)
- **Context Window:** 512 tokens with strict segment-level delimiter formatting:
  ```
  Claim:\n{claim}\n\nTitle:\n{title}\n\nEvidence:\n{evidence or '[NO_EVIDENCE_PROVIDED]'}
  ```

### Weights File (`model.safetensors`)
Due to GitHub's file size limit (100MB), the 417.67 MB PyTorch safetensors binary (`model.safetensors`) is distributed via GitHub Releases / Hugging Face model repository.

Place `model.safetensors` in this folder:
```
models/best_checkpoint_strict/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```
