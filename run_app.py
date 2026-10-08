"""
CitationGuard AI — Application Launcher
=======================================
Root launcher script to start the CitationGuard web application server.
Loads the fine-tuned PubMedBERT Strict Evidence model and serves the interactive UI.

Usage:
    python run_app.py
    python run_app.py --port 8080
"""

import os
import sys
import argparse

# Add web directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
if WEB_DIR not in sys.path:
    sys.path.insert(0, WEB_DIR)

from inference import get_verifier
from app import app

def main():
    default_port = int(os.environ.get("PORT", 7860 if "SPACE_ID" in os.environ else 5000))
    default_host = os.environ.get("HOST", "0.0.0.0" if "SPACE_ID" in os.environ else "127.0.0.1")

    parser = argparse.ArgumentParser(description="CitationGuard AI Web Application Server")
    parser.add_argument("--host", type=str, default=default_host, help=f"Host interface (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to bind (default: {default_port})")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    print("=" * 68)
    print("  CITATIONGUARD AI — EVIDENCE-AWARE CITATION VERIFICATION")
    print("=" * 68)
    print(f"Project Directory:   {BASE_DIR}")
    print(f"Model Checkpoint:    {os.path.join(BASE_DIR, 'models', 'best_checkpoint_strict')}")
    
    # Pre-load verifier singleton
    print("\n[Startup] Initializing PubMedBERT inference engine...")
    verifier = get_verifier()
    print(f"[Startup] Inference engine initialized successfully on: {verifier.device}")
    
    print(f"\n[Startup] Web application is live and accessible at:")
    print(f"          --> http://{args.host}:{args.port}/")
    print("=" * 68)
    print("Press Ctrl+C to stop the server.\n")

    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
