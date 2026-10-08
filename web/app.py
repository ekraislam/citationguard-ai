"""
CitationGuard AI — Web Application Backend
==========================================
Flask API server providing scientific claim verification endpoints and serving
the CitationGuard web application interface.
"""

import os
import sys
from flask import Flask, request, jsonify, send_from_directory

# Ensure web directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from inference import get_verifier

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Sample benchmarks for instant interactive testing
SAMPLE_BENCHMARKS = [
    {
        "id": "sample-1",
        "name": "Soil Straw Decomposition (SUPPORTED)",
        "label": "SUPPORTED",
        "claim": "Deep straw burial accelerates straw decomposition and improves soil water repellency.",
        "title": "Deep Straw Burial Accelerates Straw Decomposition and Improves Soil Water Repellency",
        "evidence": "Field trials demonstrated that deep straw burial significantly accelerates straw decomposition rates by 24% while noticeably improving soil water repellency compared to surface mulching."
    },
    {
        "id": "sample-2",
        "name": "Lignin Biodegradation Inhibition (CONTRADICTED)",
        "label": "CONTRADICTED",
        "claim": "Pandoraea sp. B-6 is incapable of degrading kraft lignin under aerobic conditions.",
        "title": "Characterization of a newly Isolated Bacterium Pandoraea sp. B-6 capable of degrading kraft Lignin",
        "evidence": "The newly isolated strain Pandoraea sp. B-6 displayed remarkable capacity for breaking down kraft lignin under aerobic incubation, achieving over 41% total lignin degradation within 7 days."
    },
    {
        "id": "sample-3",
        "name": "Target Specificity Insufficiency (NOT_ENOUGH_EVIDENCE)",
        "label": "NOT_ENOUGH_EVIDENCE",
        "claim": "CRISPR-Cas9 off-target cleavage induces oncogenic chromosomal inversions in human hematopoietic stem cells.",
        "title": "High-throughput sequencing analysis of DNA repair pathways",
        "evidence": "DNA repair pathway activation was observed across standard endonuclease treatments, but off-target structural inversion frequencies were not assayed in hematopoietic lineages."
    }
]

# Lazy-load verifier singleton on first request or startup
verifier = None

def get_app_verifier():
    global verifier
    if verifier is None:
        verifier = get_verifier()
    return verifier


@app.route("/")
def index():
    """Serves the main CitationGuard web interface."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint confirming model status and runtime device."""
    try:
        v = get_app_verifier()
        return jsonify({
            "status": "healthy",
            "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            "checkpoint": "models/best_checkpoint_strict",
            "device": str(v.device),
            "classes": ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_EVIDENCE"]
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns curated scientific sample presets for 1-click user testing."""
    return jsonify({
        "success": True,
        "samples": SAMPLE_BENCHMARKS
    }), 200


@app.route("/api/verify", methods=["POST"])
def verify_claim():
    """
    Primary verification endpoint.
    Expects JSON: { "claim": str, "title": str, "evidence": str }
    Returns: { "success": true, "data": { ... } }
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request body must be valid JSON with 'Content-Type: application/json'."
        }), 400

    data = request.get_json() or {}
    claim = data.get("claim", "").strip()
    title = data.get("title", "").strip()
    evidence = data.get("evidence", "").strip()

    if not claim:
        return jsonify({
            "success": False,
            "error": "The 'claim' field is required and cannot be empty."
        }), 400

    try:
        v = get_app_verifier()
        result = v.verify(claim=claim, title=title, evidence=evidence)

        response_payload = {
            "success": True,
            "data": {
                "claim": claim,
                "title": title,
                "evidence": evidence,
                **result
            }
        }
        return jsonify(response_payload), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Inference execution failed: {str(e)}"
        }), 500


if __name__ == "__main__":
    # Initialize verifier before accepting traffic
    get_app_verifier()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[CitationGuard] Web server starting at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
