"""
CitationGuard AI — Inference Engine
===================================
Decoupled inference module that loads the fine-tuned PubMedBERT Strict Evidence
checkpoint from models/best_checkpoint_strict/ with local_files_only=True.

Supports CUDA GPU acceleration with automatic CPU fallback.
"""

import os
import re
import time
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Default relative path from web/ to models/best_checkpoint_strict/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models", "best_checkpoint_strict")

LABEL_LIST = ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_EVIDENCE"]

def clean_scientific_text(text: str, remove_citations: bool = True) -> str:
    """Safe Unicode and superficial citation marker cleaning."""
    if not text:
        return ""
    text = str(text)
    # Unicode punctuation normalization
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\xa0", " ")

    if remove_citations:
        # Bracket citations: [1], [12, 13], [1-3]
        text = re.sub(r'\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]', '', text)
        # Author-year citations: (Smith, 2020), (Smith et al., 2020)
        text = re.sub(r'\([A-Z][A-Za-z\s]+(?:et al\.?)?,\s*\d{4}[a-z]?\)', '', text)

    return re.sub(r'\s+', ' ', text).strip()


class CitationVerifier:
    """Production inference wrapper for CitationGuard PubMedBERT model."""

    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None):
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Checkpoint directory not found at: {self.model_dir}\n"
                "Please verify that models/best_checkpoint_strict/ exists."
            )
            
        # Determine device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"[InferenceEngine] Loading tokenizer and model from local path: {self.model_dir}")
        print(f"[InferenceEngine] Target device: {self.device}")

        # Strictly load local files without remote network dependencies
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()

        # Build clean int->str label mappings
        raw_id2label = getattr(self.model.config, "id2label", {0: "SUPPORTED", 1: "CONTRADICTED", 2: "NOT_ENOUGH_EVIDENCE"})
        self.id2label = {int(k): v for k, v in raw_id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Warmup forward pass for responsive subsequent queries
        self._warmup()
        print("[InferenceEngine] CitationVerifier initialized and ready.")

    def _warmup(self):
        """Runs a tiny dummy forward pass to warm up PyTorch graph and CUDA context."""
        try:
            dummy_text = "Claim:\nWarmup\n\nTitle:\nWarmup\n\nEvidence:\nWarmup"
            inputs = self.tokenizer(dummy_text, return_tensors="pt", max_length=64, truncation=True).to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)
        except Exception as e:
            print(f"[InferenceEngine] Warmup notice: {e}")

    def verify(self, claim: str, title: str = "", evidence: str = "") -> Dict[str, Any]:
        """
        Verifies whether evidence supports, contradicts, or is insufficient for a claim.

        Input format:
        Claim:
        {clean_claim}

        Title:
        {clean_title}

        Evidence:
        {clean_evidence or '[NO_EVIDENCE_PROVIDED]'}
        """
        clean_claim = clean_scientific_text(claim, remove_citations=True)
        clean_title = clean_scientific_text(title, remove_citations=False)
        
        ev_text = clean_scientific_text(evidence, remove_citations=True)
        if not ev_text:
            clean_evidence = "[NO_EVIDENCE_PROVIDED]"
        else:
            clean_evidence = ev_text

        # Strict Evidence input template matching Phase 5 training
        formatted_input = f"Claim:\n{clean_claim}\n\nTitle:\n{clean_title}\n\nEvidence:\n{clean_evidence}"

        t0 = time.perf_counter()
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
            pred_idx = int(torch.argmax(logits, dim=-1).item())

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # If batch size is 1 and probs is a scalar (safety check)
        if isinstance(probs, float):
            probs = [probs]

        predicted_label = self.id2label.get(pred_idx, "UNKNOWN")
        confidence = float(probs[pred_idx])

        # Return calibrated probabilities for all 3 classes
        probabilities_dict = {}
        for idx in range(len(LABEL_LIST)):
            lbl = self.id2label.get(idx, LABEL_LIST[idx])
            probabilities_dict[lbl] = round(float(probs[idx]), 4)

        return {
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": probabilities_dict,
            "device_used": str(self.device),
            "inference_time_ms": round(latency_ms, 2),
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "evidence_provided": clean_evidence != "[NO_EVIDENCE_PROVIDED]"
        }


# Global singleton instance
_verifier_instance: Optional[CitationVerifier] = None

def get_verifier() -> CitationVerifier:
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = CitationVerifier()
    return _verifier_instance
