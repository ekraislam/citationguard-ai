/**
 * CitationGuard AI — Handcrafted Client Controller
 * Dedicated to human scholarly clarity, tactile responsiveness, and rigorous peer review
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- 1. Theme Management (Light / Dark) ---
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const currentTheme = localStorage.getItem("citationguard_theme") || "light";
    document.documentElement.setAttribute("data-theme", currentTheme);

    themeToggleBtn.addEventListener("click", () => {
        const activeTheme = document.documentElement.getAttribute("data-theme");
        const nextTheme = activeTheme === "light" ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", nextTheme);
        localStorage.setItem("citationguard_theme", nextTheme);
    });

    // --- 2. DOM Elements ---
    const form = document.getElementById("verifyForm");
    const claimInput = document.getElementById("claimInput");
    const titleInput = document.getElementById("titleInput");
    const evidenceInput = document.getElementById("evidenceInput");
    const submitBtn = document.getElementById("submitBtn");
    const btnSpinner = document.getElementById("btnSpinner");
    const btnText = document.getElementById("btnText");
    const clearBtn = document.getElementById("clearBtn");

    const statusLabel = document.getElementById("statusLabel");

    const emptyState = document.getElementById("emptyState");
    const loadingState = document.getElementById("loadingState");
    const resultsContainer = document.getElementById("resultsContainer");

    const verdictCard = document.getElementById("verdictCard");
    const verdictLabel = document.getElementById("verdictLabel");
    const verdictDescription = document.getElementById("verdictDescription");
    const confidenceValue = document.getElementById("confidenceValue");

    const probSupVal = document.getElementById("probSupVal");
    const probSupBar = document.getElementById("probSupBar");
    const probConVal = document.getElementById("probConVal");
    const probConBar = document.getElementById("probConBar");
    const probNeeVal = document.getElementById("probNeeVal");
    const probNeeBar = document.getElementById("probNeeBar");

    const scholarlyNoteText = document.getElementById("scholarlyNoteText");
    const insightEvidence = document.getElementById("insightEvidence");
    const insightConfidenceGrade = document.getElementById("insightConfidenceGrade");

    const copyBriefBtn = document.getElementById("copyBriefBtn");
    const printReportBtn = document.getElementById("printReportBtn");
    const toastNotice = document.getElementById("toastNotice");

    const openSpecsBtn = document.getElementById("openSpecsBtn");
    const closeSpecsBtn = document.getElementById("closeSpecsBtn");
    const specsModal = document.getElementById("specsModal");
    const presetChips = document.querySelectorAll(".preset-chip");

    // Print Document Elements
    const printProposition = document.getElementById("printProposition");
    const printPublication = document.getElementById("printPublication");
    const printExcerpt = document.getElementById("printExcerpt");
    const printVerdictLabel = document.getElementById("printVerdictLabel");
    const printVerdictCertainty = document.getElementById("printVerdictCertainty");
    const printVerdictDesc = document.getElementById("printVerdictDesc");
    const printProbSup = document.getElementById("printProbSup");
    const printProbCon = document.getElementById("printProbCon");
    const printProbNee = document.getElementById("printProbNee");
    const printReasoning = document.getElementById("printReasoning");
    const printMetaDate = document.getElementById("printMetaDate");
    const printAuditId = document.getElementById("printAuditId");

    let presetSamples = [];
    let lastVerifiedPayload = null;

    // Toast helper
    function showToast(msg) {
        toastNotice.textContent = msg;
        toastNotice.style.display = "block";
        setTimeout(() => {
            toastNotice.style.display = "none";
        }, 2200);
    }

    // Literary & scholarly interpretations per predicted stance
    const VERDICT_DESCRIPTIONS = {
        SUPPORTED: "The cited excerpt provides direct empirical backing. The literature stands in clear harmony with the proposition.",
        CONTRADICTED: "The cited findings explicitly diverge from the assertion. There is direct tension between the claim and the cited findings.",
        NOT_ENOUGH_EVIDENCE: "The excerpt lacks sufficient empirical grounding to affirm or dispute this claim. The cited material is either silent or distant from the core assertion."
    };

    // Human scholarly reasoning notes
    function generateScholarlyReasoning(label, hasEvidence, confidence) {
        if (!hasEvidence) {
            return "No citation excerpt was supplied. In rigorous peer-review, an isolated proposition cannot be substantiated without reference literature, naturally yielding an inconclusive determination.";
        }
        if (label === "SUPPORTED") {
            return `The cited passage actively corroborates the proposition. Biomedical entities, directional findings, and reported outcomes in the excerpt align with substantial semantic agreement (${(confidence * 100).toFixed(1)}% certainty).`;
        }
        if (label === "CONTRADICTED") {
            return `The cited study reports outcomes that diverge from or negate the stated hypothesis. The empirical observations in the passage stand in explicit conflict with the assertion (${(confidence * 100).toFixed(1)}% certainty).`;
        }
        return `While reference text was provided, the excerpt lacks decisive clinical metrics or specific biomedical findings required to confirm or refute the asserted claim (${(confidence * 100).toFixed(1)}% uncertainty).`;
    }

    // --- 3. Specs Modal Handlers ---
    openSpecsBtn.addEventListener("click", () => {
        specsModal.style.display = "flex";
    });

    closeSpecsBtn.addEventListener("click", () => {
        specsModal.style.display = "none";
    });

    specsModal.addEventListener("click", (e) => {
        if (e.target === specsModal) {
            specsModal.style.display = "none";
        }
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && specsModal.style.display === "flex") {
            specsModal.style.display = "none";
        }
    });

    // --- 4. Humane System Health Check ---
    async function checkSystemHealth() {
        try {
            const resp = await fetch("/api/health");
            if (resp.ok) {
                statusLabel.textContent = "Reading Engine Ready";
            } else {
                statusLabel.textContent = "Engine Paused";
            }
        } catch (e) {
            statusLabel.textContent = "Workbench Offline";
        }
    }

    // --- 5. Fetch Benchmark Samples & Active State ---
    async function fetchSamplePresets() {
        try {
            const resp = await fetch("/api/samples");
            if (resp.ok) {
                const data = await resp.json();
                presetSamples = data.samples || [];
            }
        } catch (e) {
            // Quietly fallback
        }
    }

    function clearActiveChips() {
        presetChips.forEach(c => c.classList.remove("active"));
    }

    presetChips.forEach(chip => {
        chip.addEventListener("click", () => {
            const idx = parseInt(chip.getAttribute("data-sample"), 10);
            if (presetSamples[idx]) {
                const sample = presetSamples[idx];
                claimInput.value = sample.claim;
                titleInput.value = sample.title || "";
                evidenceInput.value = sample.evidence || "";
                
                // Highlight active chip
                clearActiveChips();
                chip.classList.add("active");

                claimInput.focus();
            }
        });
    });

    // Clear active chip when user edits manually
    [claimInput, titleInput, evidenceInput].forEach(inp => {
        inp.addEventListener("input", clearActiveChips);
    });

    // --- 6. Keyboard Shortcut (Ctrl+Enter / Cmd+Enter) ---
    document.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            if (claimInput.value.trim().length > 0) {
                e.preventDefault();
                form.dispatchEvent(new Event("submit"));
            }
        }
    });

    // --- 7. Form Submission ---
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const claim = claimInput.value.trim();
        const title = titleInput.value.trim();
        const evidence = evidenceInput.value.trim();

        if (!claim) {
            claimInput.focus();
            return;
        }

        // Enter Reading State
        submitBtn.disabled = true;
        btnSpinner.style.display = "inline-block";
        btnText.textContent = "Reading...";

        emptyState.style.display = "none";
        resultsContainer.style.display = "none";
        loadingState.style.display = "flex";

        try {
            const response = await fetch("/api/verify", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ claim, title, evidence })
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                throw new Error(result.error || "Failed to examine citation grounding.");
            }

            lastVerifiedPayload = result.data;
            renderResults(result.data);

        } catch (err) {
            alert(`Examination Notice: ${err.message}`);
            emptyState.style.display = "flex";
            resultsContainer.style.display = "none";
        } finally {
            submitBtn.disabled = false;
            btnSpinner.style.display = "none";
            btnText.textContent = "Examine Citation";
            loadingState.style.display = "none";
        }
    });

    // --- 8. Render Verification Outcome with Scholarly Care ---
    function renderResults(data) {
        const label = data.predicted_label;
        const confidencePct = (data.confidence * 100).toFixed(1);
        const probs = data.probabilities;

        // Populate Verdict Card
        verdictLabel.textContent = label.replace(/_/g, " ");
        verdictDescription.textContent = VERDICT_DESCRIPTIONS[label] || "Empirical grounding evaluated by PubMedBERT.";
        confidenceValue.textContent = `${confidencePct}%`;

        // Apply Colorway
        verdictCard.className = "verdict-card";
        if (label === "SUPPORTED") {
            verdictCard.classList.add("card-sup");
        } else if (label === "CONTRADICTED") {
            verdictCard.classList.add("card-con");
        } else {
            verdictCard.classList.add("card-nee");
        }

        // Probability Distribution
        const supPct = ((probs["SUPPORTED"] || 0) * 100).toFixed(1);
        const conPct = ((probs["CONTRADICTED"] || 0) * 100).toFixed(1);
        const neePct = ((probs["NOT_ENOUGH_EVIDENCE"] || 0) * 100).toFixed(1);

        probSupVal.textContent = `${supPct}%`;
        probConVal.textContent = `${conPct}%`;
        probNeeVal.textContent = `${neePct}%`;

        setTimeout(() => {
            probSupBar.style.width = `${supPct}%`;
            probConBar.style.width = `${conPct}%`;
            probNeeBar.style.width = `${neePct}%`;
        }, 60);

        // Populate Scholarly Observation (Human Reasoning)
        const reasoningText = generateScholarlyReasoning(label, data.evidence_provided, data.confidence);
        if (scholarlyNoteText) {
            scholarlyNoteText.textContent = reasoningText;
        }

        // Update Context Insight Strip
        if (insightEvidence) {
            insightEvidence.textContent = data.evidence_provided ? "Empirical Context Present" : "Isolated Proposition (No Excerpt)";
        }
        if (insightConfidenceGrade) {
            if (data.confidence >= 0.85) {
                insightConfidenceGrade.textContent = "Decisive Consensus (>85%)";
            } else if (data.confidence >= 0.60) {
                insightConfidenceGrade.textContent = "Substantial Certainty (60–85%)";
            } else {
                insightConfidenceGrade.textContent = "Cautious Boundary (<60%)";
            }
        }

        // Populate Print Document for PDF Generation
        if (printProposition) printProposition.textContent = `“${data.claim}”`;
        if (printPublication) printPublication.textContent = data.title || "None specified in submission";
        if (printExcerpt) printExcerpt.textContent = data.evidence_provided ? `“${data.evidence}”` : "[No literature excerpt cited — evaluated under baseline condition]";
        if (printVerdictLabel) printVerdictLabel.textContent = label.replace(/_/g, " ");
        if (printVerdictCertainty) printVerdictCertainty.textContent = `${confidencePct}%`;
        if (printVerdictDesc) printVerdictDesc.textContent = VERDICT_DESCRIPTIONS[label];
        if (printProbSup) printProbSup.textContent = `${supPct}%`;
        if (printProbCon) printProbCon.textContent = `${conPct}%`;
        if (printProbNee) printProbNee.textContent = `${neePct}%`;
        if (printReasoning) printReasoning.textContent = `“${reasoningText}”`;
        if (printMetaDate) printMetaDate.textContent = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
        if (printAuditId) printAuditId.textContent = Math.random().toString(36).substring(2, 8).toUpperCase();

        // Show Results
        resultsContainer.style.display = "flex";
    }

    // --- 9. Export Peer-Review Audit Brief (Markdown) ---
    copyBriefBtn.addEventListener("click", async () => {
        if (!lastVerifiedPayload) return;

        const data = lastVerifiedPayload;
        const label = data.predicted_label;
        const confidencePct = (data.confidence * 100).toFixed(1);
        const probs = data.probabilities;
        const reasoning = generateScholarlyReasoning(label, data.evidence_provided, data.confidence);
        const today = new Date().toISOString().split("T")[0];

        const memo = [
            `# CitationGuard AI — Scientific Evidence Audit Brief`,
            `*Generated for Academic Peer-Review & Evidence Grounding Verification*`,
            ``,
            `**Date:** ${today}`,
            `**Proposition:** "${data.claim}"`,
            `**Cited Reference:** ${data.title ? `"${data.title}"` : "None specified"}`,
            `**Cited Excerpt:** ${data.evidence_provided ? `"${data.evidence}"` : "[No literature excerpt cited — evaluated under baseline condition]"}`,
            ``,
            `---`,
            `### Evaluated Stance: **${label.replace(/_/g, " ")}** (${confidencePct}% Certainty)`,
            ``,
            `**Class Probability Spectrum:**`,
            `- Supported: ${((probs["SUPPORTED"] || 0) * 100).toFixed(1)}%`,
            `- Contradicted: ${((probs["CONTRADICTED"] || 0) * 100).toFixed(1)}%`,
            `- Inconclusive: ${((probs["NOT_ENOUGH_EVIDENCE"] || 0) * 100).toFixed(1)}%`,
            ``,
            `**Scholarly Assessment Note:**`,
            `> ${reasoning}`,
            ``,
            `---`,
            `**Evaluation Engine:** BiomedNLP-PubMedBERT-base-uncased-abstract (Phase 5 Strict Evidence Protocol)`,
            `**Inference Latency:** ${data.inference_time_ms} ms | **Sequence Length:** ${data.input_tokens || "--"} tokens`
        ].join("\n");

        try {
            await navigator.clipboard.writeText(memo);
            showToast("Peer-review audit brief copied to clipboard");
        } catch (e) {
            showToast("Failed to copy audit brief");
        }
    });

    // --- 10. Print / Save as PDF ---
    printReportBtn.addEventListener("click", () => {
        if (!lastVerifiedPayload) return;
        window.print();
    });

    // --- 11. Reset Workbench ---
    clearBtn.addEventListener("click", () => {
        claimInput.value = "";
        titleInput.value = "";
        evidenceInput.value = "";

        clearActiveChips();

        resultsContainer.style.display = "none";
        loadingState.style.display = "none";
        emptyState.style.display = "flex";

        probSupBar.style.width = "0%";
        probConBar.style.width = "0%";
        probNeeBar.style.width = "0%";

        claimInput.focus();
    });

    // Initialize
    checkSystemHealth();
    fetchSamplePresets();
});
