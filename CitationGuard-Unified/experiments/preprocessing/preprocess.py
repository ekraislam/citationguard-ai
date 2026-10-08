"""
CitationGuard AI — Reproducible Preprocessing Module
=====================================================
Deterministic text cleaning and experiment-time representation builder.

Requirements:
- Unicode normalization (NFKC)
- Whitespace normalization
- Safe punctuation normalization
- Citation marker removal (without altering chemical, genetic, or numeric statistical scientific notation)
- Preservation of original dataset attributes
- Generation of five standardized input conditions:
    Condition A: Claim Only
    Condition B: Claim + Title
    Condition C: Claim + Evidence (Strict)
    Condition D: Claim + Abstract Fallback
    Condition E: Full Standardized Input (Claim + Title + Abstract Fallback)
"""

import re
import unicodedata
from typing import Dict, Any, Tuple

# Surface citation patterns to safely remove:
# 1. Author et al. (Year) or Author (Year)
# e.g., "Alvarez et al. (2024)", "Smith (2018)"
AUTHOR_YEAR_PATTERN = re.compile(
    r'\b[A-Z][a-zA-Z\s\.,\-–]+et\s+al\.?\s*\((?:19|20)\d{2}[a-z]?\)',
    flags=re.UNICODE
)

# 2. Parenthetical citations with author and year or trailing numeric references:
# e.g., "(Smith et al., 2020)", "(Author, 2018)", "(Cemerski et al. 2007)", "(5)", "(35)"
# Must NOT match: "(95% CI, 1.07-3.08)", "(p < 0.05)", "(Figure S1G)", "(AML)", "(iPSCs)"
PAREN_CITE_PATTERN = re.compile(
    r'\(\s*[A-Z][a-zA-Z\s\.,\-–]+(?:et\s+al\.?,?\s*)?(?:19|20)\d{2}[a-z]?\s*\)',
    flags=re.UNICODE
)
PAREN_NUMERIC_PATTERN = re.compile(
    r'\(\s*(?:[1-9]\d{0,2}(?:\s*,\s*[1-9]\d{0,2})*)\s*\)',
    flags=re.UNICODE
)

# 3. Numeric bracket citations:
# e.g., "[1]", "[12]", "[7, 8]", "[1-3]", "[1–4]"
# Must NOT match: "[Ca2+]", "[3H]", "[18F]"
BRACKET_NUMERIC_PATTERN = re.compile(
    r'\[\s*(?:\d{1,3}(?:\s*[\-,–]\s*\d{1,3})?(?:\s*,\s*\d{1,3}(?:\s*[\-,–]\s*\d{1,3})?)*)\s*\]',
    flags=re.UNICODE
)

# 4. Trailing or orphaned "et al." discourse markers:
ET_AL_PATTERN = re.compile(r'\bet\s+al\b\.?', flags=re.IGNORECASE)

def normalize_unicode_and_whitespace(text: str) -> str:
    """Normalize unicode and collapse redundant whitespace."""
    if not text:
        return ""
    # Unicode NFKC normalization
    text = unicodedata.normalize('NFKC', str(text))
    # Replace non-breaking spaces, zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\uFEFF\u00A0]', ' ', text)
    # Normalize typographer quotes and dashes safely
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('–', '-').replace('—', '-')
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_citation_markers(text: str) -> str:
    """
    Remove surface citation markers without destroying scientific notation.
    Preserves:
    - Chemical formulas (e.g., [Ca2+], [3H]thymidine)
    - P-values and confidence intervals (e.g., (p < 0.05), (95% CI))
    - Figure and table references (e.g., (Figure 1), (Table S2))
    - Scientific abbreviations (e.g., (MEFs), (AML))
    """
    if not text:
        return ""
    
    cleaned = text
    # 1. Author et al. (Year)
    cleaned = AUTHOR_YEAR_PATTERN.sub('', cleaned)
    # 2. Parenthetical citations
    cleaned = PAREN_CITE_PATTERN.sub('', cleaned)
    cleaned = PAREN_NUMERIC_PATTERN.sub('', cleaned)
    # 3. Numeric bracket citations
    cleaned = BRACKET_NUMERIC_PATTERN.sub('', cleaned)
    # 4. Standalone "et al."
    cleaned = ET_AL_PATTERN.sub('', cleaned)
    
    # Clean up whitespace before punctuation (e.g., "word ." -> "word.")
    cleaned = re.sub(r'\s+([,.:;])', r'\1', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def clean_scientific_text(text: str, remove_citations: bool = True) -> str:
    """Complete cleaning pipeline for scientific text strings."""
    norm = normalize_unicode_and_whitespace(text)
    if remove_citations:
        norm = remove_citation_markers(norm)
    return norm

def prepare_record_contexts(record: Dict[str, Any]) -> Tuple[str, str, bool]:
    """
    Build context_strict and context_fallback fields according to Rule 3.
    Returns:
        (context_strict, context_fallback, used_fallback)
    """
    evidence = record.get("evidence_text")
    abstract = record.get("paper_abstract")
    
    has_valid_evidence = evidence is not None and len(str(evidence).strip()) > 0
    
    if has_valid_evidence:
        context_strict = clean_scientific_text(str(evidence), remove_citations=False)
        context_fallback = context_strict
        used_fallback = False
    else:
        context_strict = ""
        context_fallback = clean_scientific_text(str(abstract) if abstract else "", remove_citations=False)
        used_fallback = True
        
    return context_strict, context_fallback, used_fallback

def build_input_conditions(record: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate all 5 input conditions for a single record.
    Conditions:
      Condition A: Claim Only
      Condition B: Claim + Title
      Condition C: Claim + Evidence (Strict)
      Condition D: Claim + Abstract Fallback
      Condition E: Full Standardized Input (Claim + Title + Fallback Context)
    """
    claim = clean_scientific_text(record.get("claim_text", ""), remove_citations=True)
    title = clean_scientific_text(record.get("paper_title", ""), remove_citations=False)
    ctx_strict, ctx_fallback, _ = prepare_record_contexts(record)
    
    cond_a = claim
    cond_b = f"Claim: {claim}\nTitle: {title}" if title else f"Claim: {claim}"
    cond_c = f"Claim: {claim}\nEvidence: {ctx_strict}" if ctx_strict else f"Claim: {claim}\nEvidence: [NO_EVIDENCE_PROVIDED]"
    cond_d = f"Claim: {claim}\nEvidence: {ctx_fallback}" if ctx_fallback else f"Claim: {claim}\nEvidence: [NO_EVIDENCE_PROVIDED]"
    cond_e = f"Claim: {claim}\nTitle: {title}\nEvidence: {ctx_fallback}" if title else f"Claim: {claim}\nEvidence: {ctx_fallback}"
    
    return {
        "condition_a": cond_a,
        "condition_b": cond_b,
        "condition_c": cond_c,
        "condition_d": cond_d,
        "condition_e": cond_e,
    }
