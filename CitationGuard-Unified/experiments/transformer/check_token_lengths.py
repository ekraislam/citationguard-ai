import os
import sys
import json
import numpy as np
from transformers import AutoTokenizer

sys.path.append(r"c:\Users\hp\Desktop\CitationGuard-Datasets\CitationGuard-Unified\experiments\preprocessing")
from preprocess import clean_scientific_text, prepare_record_contexts

model_id = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
print(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("Tokenizer loaded successfully!")

data_path = r"c:\Users\hp\Desktop\CitationGuard-Datasets\CitationGuard-Unified\citationguard_verification.jsonl"
with open(data_path, "r", encoding="utf-8") as f:
    recs = [json.loads(line) for line in f if line.strip()]

strict_lengths = []
fallback_lengths = []
claim_lengths = []

for r in recs:
    claim = clean_scientific_text(r.get("claim_text", ""), remove_citations=True)
    title = clean_scientific_text(r.get("paper_title", ""), remove_citations=False)
    ctx_strict, ctx_fallback, _ = prepare_record_contexts(r)
    
    no_ev = "[NO_EVIDENCE_PROVIDED]"
    ev_str = ctx_strict if ctx_strict else no_ev
    ev_fb_str = ctx_fallback if ctx_fallback else no_ev
    
    text_strict = f"Claim:\n{claim}\n\nTitle:\n{title}\n\nEvidence:\n{ev_str}"
    text_fallback = f"Claim:\n{claim}\n\nTitle:\n{title}\n\nEvidence:\n{ev_fb_str}"
    
    tokens_s = tokenizer.encode(text_strict, truncation=False)
    tokens_fb = tokenizer.encode(text_fallback, truncation=False)
    tokens_c = tokenizer.encode(claim, truncation=False)
    
    strict_lengths.append(len(tokens_s))
    fallback_lengths.append(len(tokens_fb))
    claim_lengths.append(len(tokens_c))

results = {}
for name, arr in [
    ("strict_input", strict_lengths),
    ("fallback_input", fallback_lengths),
    ("claim_only", claim_lengths)
]:
    stats_dict = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "truncation": {
            "256": {"count": sum(1 for x in arr if x > 256), "pct": sum(1 for x in arr if x > 256) / len(arr) * 100},
            "384": {"count": sum(1 for x in arr if x > 384), "pct": sum(1 for x in arr if x > 384) / len(arr) * 100},
            "512": {"count": sum(1 for x in arr if x > 512), "pct": sum(1 for x in arr if x > 512) / len(arr) * 100},
        }
    }
    results[name] = stats_dict
    print(f"=== {name} ===")
    print(f"Mean: {stats_dict['mean']:.2f}, Median: {stats_dict['median']:.1f}, Max: {stats_dict['max']}, P95: {stats_dict['p95']:.1f}, P99: {stats_dict['p99']:.1f}")
    print(f"Truncated at 256: {stats_dict['truncation']['256']['pct']:.2f}%, at 384: {stats_dict['truncation']['384']['pct']:.2f}%, at 512: {stats_dict['truncation']['512']['pct']:.2f}%")
    print()

out_json = r"c:\Users\hp\Desktop\CitationGuard-Datasets\CitationGuard-Unified\experiments\transformer\token_length_stats.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved stats to", out_json)
