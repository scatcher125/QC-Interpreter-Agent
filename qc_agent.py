# QC Interpreter Agent for Alfred BAM Alignment Statistics
# Reva S
# 07-Jun-206

# DESCRIPTION
# QC Tool: Alfred is an efficient and versatile BAM alignment QC tool.
# Input:
# - Alfred BAM alignment QC `*.json.gz` output file from standard 30x Illumina whole exome 
#   sequencing on human data.
# - Agent uses OpenAI by default but can be configured to use other LLM providers supported by LangChain.
# Output: Report including PASS or FAIL status and plain English summary.
# Usage:
#   python qc_agent.py --input sample_qc.json
#   python qc_agent.py --input sample_qc.json --output report.txt
#   python qc_agent.py --input sample_qc.json --model gemini-2.5-flash
#   python qc_agent.py --input sample_qc.json --model claude-3-5-sonnet-20241022 --provider anthropic
#   python qc_agent.py --input sample_qc.json --model gpt-4o --provider openai
# ---------

# load requirements
import json
import argparse
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model

# NODE 1
# Define "State" object
class QCState(TypedDict):
    raw_input: dict          # Raw QC JSON loaded from file
    parsed_metrics: dict     # Cleaned structured QC metrics
    llm_summary: str         # Natural language LLM output
    pass_fail: str           # "PASS", "WARN", or "FAIL" status
    output_report: str       # Final report
    model_name: str          # LLM model name passed through state
    model_provider: str      # LLM provider passed through state
    assay_type: str          # "wes" or "wgs" — selects threshold set and prompt

## ---------------------------------------------------------------------------
# ASSAY CONFIGURATION
# Each assay defines its own threshold set, metric field order, INFO-only
# metrics, and LLM prompt glossary. Fraction metrics are 0-1, not percentages.
# ---------------------------------------------------------------------------

# Calibrated for 30x Illumina whole EXOME sequencing on human data.
WES_QC_THRESHOLDS = {
    "MappedFraction":    {"warn_low": 0.75, "fail_low": 0.50},   # alignment quality
    "DuplicateFraction": {"warn_high": 0.35, "fail_high": 0.50},  # library complexity
    "MedianMAPQ":        {"warn_low": 30,   "fail_low": 20},      # mapping confidence
    "FractionInBed":     {"warn_low": 0.70, "fail_low": 0.35},    # on-target rate
    "EnrichmentOverBed": {"warn_low": 2.0,  "fail_low": 0.9},     # capture efficiency
    "MedianCoverage":    {"warn_low": 15.0, "fail_low": 5.0},     # depth over targets
    "GCContent":         {"warn_low": 0.35, "fail_low": 0.20,
                          "warn_high": 0.60, "fail_high": 0.80},
}

# Calibrated for 30x Illumina whole GENOME sequencing on human data.
# Differences from WES, and why:
#   - FractionInBed / EnrichmentOverBed are ABSENT, not demoted. There is no
#     capture step in WGS, so these are undefined rather than merely
#     uninformative. Omitting them keeps the MISSING -> WARN rule from firing
#     on every WGS sample.
#   - DuplicateFraction is tighter: PCR-free WGS libraries typically run 1-10%.
#   - MedianCoverage is recentred on a ~30x genome-wide target.
#   - GCContent is recentred on the whole-genome value (~0.41) rather than the
#     GC-rich exome value.
#   - CoverageCV is PROMOTED to a thresholded metric. Coverage evenness is far
#     more diagnostic genome-wide than it is over small capture targets.
WGS_QC_THRESHOLDS = {
    "MappedFraction":    {"warn_low": 0.75, "fail_low": 0.50},
    "DuplicateFraction": {"warn_high": 0.20, "fail_high": 0.30},
    "MedianMAPQ":        {"warn_low": 30,   "fail_low": 20},
    "MedianCoverage":    {"warn_low": 25.0, "fail_low": 15.0},
    "CoverageCV":        {"warn_high": 0.35, "fail_high": 0.50},
    "GCContent":         {"warn_low": 0.38, "fail_low": 0.34,
                          "warn_high": 0.45, "fail_high": 0.50},
}

# Backwards-compatible alias for any external code importing the old name.
QC_THRESHOLDS = WES_QC_THRESHOLDS

# Metrics reported as context only; excluded from the PASS/WARN/FAIL rollup.
WES_INFO_ONLY = {"Mapped", "DuplicateMarked", "SDCoverage",
                 "MedianInsertSize", "SDInsertSize"}
WGS_INFO_ONLY = {"Mapped", "DuplicateMarked", "SDCoverage",
                 "MedianInsertSize", "SDInsertSize"}

# Field order controls the order metrics appear in the report.
WES_METRIC_FIELDS = [
    "Mapped", "MappedFraction", "DuplicateMarked", "DuplicateFraction",
    "MedianMAPQ", "FractionInBed", "MedianCoverage", "SDCoverage",
    "EnrichmentOverBed", "MedianInsertSize", "SDInsertSize", "GCContent",
]
WGS_METRIC_FIELDS = [
    "Mapped", "MappedFraction", "DuplicateMarked", "DuplicateFraction",
    "MedianMAPQ", "MedianCoverage", "SDCoverage", "CoverageCV",
    "MedianInsertSize", "SDInsertSize", "GCContent",
]

WES_GLOSSARY = """- MappedFraction: fraction of reads that aligned to the human reference genome
- DuplicateFraction: fraction of reads that are PCR duplicates (higher = worse library complexity)
- MedianMAPQ: median mapping quality score (higher = reads mapped more confidently)
- FractionInBed: fraction of reads landing on exome target regions (on-target rate)
- EnrichmentOverBed: fold enrichment of target (exon) regions over background genome
- MedianCoverage: median read depth across targeted exome regions
- SDCoverage: variability in coverage depth across targets (INFO only, not thresholded)
- MedianInsertSize: median DNA fragment size in base pairs
- GCContent: fraction of bases that are G or C (expected ~0.45-0.52 for human exome)
- Mapped: total number of mapped reads (INFO only)
- DuplicateMarked: total number of duplicate reads (INFO only)"""

WGS_GLOSSARY = """- MappedFraction: fraction of reads that aligned to the human reference genome
- DuplicateFraction: fraction of reads that are PCR duplicates (higher = worse library complexity; PCR-free genome libraries are normally very low)
- MedianMAPQ: median mapping quality score (higher = reads mapped more confidently)
- MedianCoverage: median read depth across the whole genome (not a captured subset)
- SDCoverage: absolute variability in depth genome-wide (INFO only, not thresholded)
- CoverageCV: coverage evenness, SDCoverage divided by MedianCoverage. Higher = patchier, less uniform depth across the genome
- MedianInsertSize: median DNA fragment size in base pairs
- GCContent: fraction of bases that are G or C (expected ~0.40-0.42 for the whole human genome)
- Mapped: total number of mapped reads (INFO only)
- DuplicateMarked: total number of duplicate reads (INFO only)
Note: there is no capture or enrichment step in whole genome sequencing, so on-target and enrichment metrics do not apply and are not reported."""

ASSAY_CONFIG = {
    "wes": {
        "label": "Whole Exome Sequencing (WES)",
        "description": ("This is whole exome sequencing (WES) data from a human sample "
                        "sequenced on Illumina with 30x target coverage."),
        "thresholds": WES_QC_THRESHOLDS,
        "info_only": WES_INFO_ONLY,
        "fields": WES_METRIC_FIELDS,
        "glossary": WES_GLOSSARY,
    },
    "wgs": {
        "label": "Whole Genome Sequencing (WGS)",
        "description": ("This is whole genome sequencing (WGS) data from a human sample "
                        "sequenced on Illumina at approximately 30x genome-wide coverage. "
                        "There is no capture or target enrichment step."),
        "thresholds": WGS_QC_THRESHOLDS,
        "info_only": WGS_INFO_ONLY,
        "fields": WGS_METRIC_FIELDS,
        "glossary": WGS_GLOSSARY,
    },
}

# Accepted --assay spellings, normalised to canonical keys.
ASSAY_ALIASES = {
    "wes": "wes", "exome": "wes", "wxs": "wes", "panel": "wes",
    "wgs": "wgs", "genome": "wgs",
}

# function to evaluate metrics against thresholds
def evaluate_metrics(parsed_metrics: dict,
                     thresholds: dict = None,
                     info_only: set = None) -> dict:
    """
    Evaluates each metric against the supplied threshold set.
    Returns {metric_name: {"value": ..., "status": "PASS"|"WARN"|"FAIL"|"INFO"|"MISSING"}}.
    INFO-only metrics are passed through without evaluation.
    Metrics present in the threshold set but null in the JSON are marked
    MISSING and treated as WARN in rollup. Metrics absent from the threshold
    set are skipped entirely (this is how WGS drops capture metrics).
    Defaults to the WES configuration for backwards compatibility.
    """
    thresholds_map = WES_QC_THRESHOLDS if thresholds is None else thresholds
    info_set = WES_INFO_ONLY if info_only is None else info_only

    results = {}
    for metric, value in parsed_metrics.items():
        if metric in ("sample_id", "assay_type"):
            continue
        if metric in info_set:
            results[metric] = {"value": value, "status": "INFO"}
            continue
        if metric not in thresholds_map:
            continue
        if value is None:
            results[metric] = {"value": None, "status": "MISSING"}
            continue

        t = thresholds_map[metric]
        status = "PASS"
        if "fail_low" in t and value < t["fail_low"]:
            status = "FAIL"
        elif "warn_low" in t and value < t["warn_low"]:
            status = "WARN"
        elif "fail_high" in t and value > t["fail_high"]:
            status = "FAIL"
        elif "warn_high" in t and value > t["warn_high"]:
            status = "WARN"

        results[metric] = {"value": value, "status": status}
    return results

# determine overall pass-fail status for sample
def overall_pass_fail(evaluated: dict) -> str:
    """
    Rolls up per-metric statuses into a single sample-level verdict.
    Any FAIL → FAIL. Any WARN or MISSING (with no FAIL) → WARN. Otherwise → PASS.
    INFO metrics are excluded from rollup.
    """
    statuses = {v["status"] for v in evaluated.values()}

    if "FAIL" in statuses:
        return "FAIL"
    elif "WARN" in statuses or "MISSING" in statuses:
        return "WARN"
    else:
        return "PASS"


# NODE 0 — ASSAY ROUTER
def route_assay(state: QCState) -> QCState:
    """
    Normalises the requested assay type and writes it back to state.
    Runs first so that downstream nodes can rely on state["assay_type"]
    being a canonical key of ASSAY_CONFIG.
    """
    requested = (state.get("assay_type") or "wes").strip().lower()
    assay = ASSAY_ALIASES.get(requested)
    if assay is None:
        valid = ", ".join(sorted(set(ASSAY_ALIASES)))
        raise ValueError(f"Unknown assay type '{requested}'. Valid values: {valid}")
    return {**state, "assay_type": assay}


def assay_branch(state: QCState) -> str:
    """Conditional-edge selector: returns the canonical assay key."""
    return state["assay_type"]


def _compute_coverage_cv(median_cov, sd_cov):
    """
    Coverage evenness = SDCoverage / MedianCoverage. Returns None if either
    input is missing or the median is non-positive, which surfaces as MISSING
    rather than raising.
    """
    if median_cov is None or sd_cov is None:
        return None
    try:
        if float(median_cov) <= 0:
            return None
        return round(float(sd_cov) / float(median_cov), 3)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _parse_qc(state: QCState, assay: str) -> QCState:
    """Shared parse/evaluate logic, parameterised by assay type."""
    raw = state["raw_input"]
    cfg = ASSAY_CONFIG[assay]

    metrics = {"sample_id": raw.get("sample_id", "Unknown Sample")}
    for field in cfg["fields"]:
        if field == "CoverageCV":
            metrics[field] = _compute_coverage_cv(
                raw.get("MedianCoverage"), raw.get("SDCoverage")
            )
        else:
            metrics[field] = raw.get(field)

    evaluated = evaluate_metrics(metrics, cfg["thresholds"], cfg["info_only"])
    verdict = overall_pass_fail(evaluated)
    metrics["evaluated"] = evaluated
    metrics["overall_status"] = verdict
    metrics["assay_type"] = assay
    metrics["assay_label"] = cfg["label"]
    return {**state, "parsed_metrics": metrics, "pass_fail": verdict}


# NODE 1a — WES PARSE
def parse_qc_wes(state: QCState) -> QCState:
    """Extract and evaluate exome QC metrics (capture-aware)."""
    return _parse_qc(state, "wes")


# NODE 1b — WGS PARSE  <-- the new node
def parse_qc_wgs(state: QCState) -> QCState:
    """
    Extract and evaluate whole-genome QC metrics.
    Skips capture metrics entirely and derives CoverageCV for evenness.
    """
    return _parse_qc(state, "wgs")


# Backwards-compatible alias: existing imports of parse_qc_metrics still work.
parse_qc_metrics = parse_qc_wes


# NODE 2
# function: generate QC summary in natural language
def generate_llm_summary(state: QCState) -> QCState:
    """Use an LLM to generate a plain-English QC summary."""
    # init_chat_model selects the correct LangChain integration based on model/provider.
    # The appropriate API key must be set as an environment variable:
    #   OpenAI:    OPENAI_API_KEY
    #   Anthropic: ANTHROPIC_API_KEY
    #   Google:    GOOGLE_API_KEY
    llm = init_chat_model(
        model=state["model_name"],
        model_provider=state["model_provider"] if state["model_provider"] else None,
        temperature=0.3,
        max_tokens=1500,
    )

    metrics = state["parsed_metrics"]
    evaluated = metrics["evaluated"]
    cfg = ASSAY_CONFIG[state.get("assay_type", "wes")]

    # Build readable metric block for prompt
    metric_lines = []
    for metric, result in evaluated.items():
        value = result["value"]
        status = result["status"]
        if value is not None:
            metric_lines.append(f"  - {metric}: {value} [{status}]")
        else:
            metric_lines.append(f"  - {metric}: Not available [{status}]")

    prompt = f"""You are a senior bioinformatician reviewing sequencing QC metrics for a research collaborator 
who is an expert biologist but has no computational background. Your job is to explain whether 
this sample passed quality control and what the results mean for their downstream analysis.
Be concise, clear, and avoid jargon. If there are issues, explain what they mean biologically 
and whether the analysis can still proceed. Do not mention specific software tools or pipeline steps.
{cfg['description']}
Sample ID: {metrics['sample_id']}
Assay: {cfg['label']}
Overall Status: {metrics['overall_status']}
QC Metrics (status: PASS/WARN/FAIL/INFO/MISSING):
{chr(10).join(metric_lines)}
Metric notes:
{cfg['glossary']}
Please write a short plain prose response (no numbered lists, no bullet points) covering:
- A 2-3 sentence summary of the overall sample quality
- Whether downstream analysis (e.g. variant calling) can proceed
- Any recommended actions if there are issues
Keep the total response under 120 words.
"""

    response = llm.invoke(prompt)
    summary = response.content.strip()
    return {**state, "llm_summary": summary}


# NODE 3
# function: combine QC metrics & LLM summary into clean human-readable report
def format_report(state: QCState) -> QCState:
    """Assemble the final human-readable QC report."""
    metrics = state["parsed_metrics"]
    evaluated = metrics["evaluated"]
    status_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(state["pass_fail"], "")

    # Build metrics table with per-metric status indicators
    metric_lines = []
    for metric, result in evaluated.items():
        value = result["value"]
        status = result["status"]
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️", "MISSING": "❓"}.get(status, "")
        if value is not None:
            metric_lines.append(f"  {icon} {metric:<22} {value}")
        else:
            metric_lines.append(f"  {icon} {metric:<22} N/A")

    assay_label = metrics.get("assay_label", "Whole Exome Sequencing (WES)")
    report = f"""
================================================================================
  QC REPORT — {metrics['sample_id']}
  Assay: {assay_label}
  Overall Status: {status_emoji} {state['pass_fail']}
================================================================================

METRICS SUMMARY
---------------
{chr(10).join(metric_lines)}

  Legend: ✅ PASS  ⚠️ WARN  ❌ FAIL  ℹ️ INFO (not thresholded)  ❓ MISSING

BIOLOGICAL INTERPRETATION
-----------------------------------
{state['llm_summary']}

================================================================================
"""
    return {**state, "output_report": report}


# function: Build LangGraph
def build_graph() -> StateGraph:
    """
    route_assay ──┬─→ parse_qc_wes ─┐
                  └─→ parse_qc_wgs ─┴─→ llm_summary → format_report → END
    """
    graph = StateGraph(QCState)
    graph.add_node("route_assay", route_assay)
    graph.add_node("parse_qc_wes", parse_qc_wes)
    graph.add_node("parse_qc_wgs", parse_qc_wgs)
    graph.add_node("llm_summary", generate_llm_summary)
    graph.add_node("format_report", format_report)

    graph.set_entry_point("route_assay")
    graph.add_conditional_edges(
        "route_assay",
        assay_branch,
        {"wes": "parse_qc_wes", "wgs": "parse_qc_wgs"},
    )
    graph.add_edge("parse_qc_wes", "llm_summary")
    graph.add_edge("parse_qc_wgs", "llm_summary")
    graph.add_edge("llm_summary", "format_report")
    graph.add_edge("format_report", END)
    return graph.compile()

# Main function
def main():
    parser = argparse.ArgumentParser(description="LangGraph QC Interpreter Agent")
    parser.add_argument("--input", required=True, help="Path to QC JSON file")
    parser.add_argument("--output", default=None, help="Optional path to save report")
    parser.add_argument("--model", default="gemini-2.5-flash", help="LLM model name (default: gemini-2.5-flash)")
    parser.add_argument("--provider", default="google_genai",
                        help="LLM provider: openai, anthropic, google_genai, etc. "
                             "Optional — init_chat_model infers from model name if not set.")
    parser.add_argument("--assay", default="wes", choices=["wes", "wgs", "exome", "genome", "wxs", "panel"],
                        help="Assay type, selects threshold set and interpretation context "
                             "(default: wes)")
    args = parser.parse_args()

    with open(args.input) as f:
        qc_data = json.load(f)

    graph = build_graph()
    result = graph.invoke({
        "raw_input": qc_data,
        "model_name": args.model,
        "model_provider": args.provider,
        "assay_type": args.assay,
    })

    print(result["output_report"])

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result["output_report"])
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
