# QC Interpreter Agent for Alfred BAM Alignment Statistics
# Reva S
# 07-Jun-2026
#
# DESCRIPTION
# QC Tool: Alfred is an efficient and versatile BAM alignment QC tool.
# Input:
# - Alfred BAM alignment QC `*.json.gz` or `*.json` output file from 30x Illumina whole exome (WES) 
#   or whole genome (WGS) sequencing on human data.
# - Assay type is selected with --assay and determines the threshold set, which metrics are reported, 
#  and the interpretation context given to the LLM.
# - Agent uses free Gemini LLM by default but can be configured to use other LLM providers supported
#   by LangChain (OpenAI, Anthropic etc.).
# Output: Report including PASS/WARN/FAIL status and plain English summary.
# Usage:
#   python qc_agent.py --input sample_qc.json
#   python qc_agent.py --input sample_qc.json.gz
#   python qc_agent.py --input sample_qc.json.gz --output report.md
#   python qc_agent.py --input sample_qc.json.gz --assay wgs
#   python qc_agent.py --input sample_qc.json.gz --model gemini-2.5-flash
#   python qc_agent.py --input sample_qc.json.gz --model claude-3-5-sonnet-20241022 --provider anthropic
#   python qc_agent.py --input sample_qc.json --model gpt-4o --provider openai
#   python qc_agent.py --diagram > docs/workflow.mmd
# ---------

# load requirements
import json
import gzip
import argparse
from functools import lru_cache
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model

# STATE
class QCState(TypedDict):
    raw_input: dict          # Raw QC JSON loaded from file
    parsed_metrics: dict     # Structured QC metrics
    llm_summary: str         # Natural language LLM output
    pass_fail: str           # "PASS", "WARN", or "FAIL" status
    output_report: str       # Final report
    model_name: str          # LLM model name
    model_provider: str      # LLM provider
    assay_type: str          # "wes" or "wgs"

# ---------------------------------------------------------------------------
# ASSAY CONFIGURATION
# Each assay defines its own threshold set and LLM prompt glossary. Fraction metrics are 0-1.
# ---------------------------------------------------------------------------

# QC Metric thresholds calibrated for 30x Illumina WES on human data.
WES_QC_THRESHOLDS = {
    "MappedFraction":    {"warn_low": 0.75, "fail_low": 0.50},    # alignment quality
    "DuplicateFraction": {"warn_high": 0.35, "fail_high": 0.50},  # library complexity
    "MedianMAPQ":        {"warn_low": 30,   "fail_low": 20},      # mapping confidence
    "FractionInBed":     {"warn_low": 0.70, "fail_low": 0.35},    # on-target rate
    "EnrichmentOverBed": {"warn_low": 2.0,  "fail_low": 0.9},     # capture efficiency
    "MedianCoverage":    {"warn_low": 15.0, "fail_low": 5.0},     # depth over targets
    "GCContent":         {"warn_low": 0.35, "fail_low": 0.20,
                          "warn_high": 0.60, "fail_high": 0.80},
}

# QC Metric thresholds calibrated for 30x Illumina WGS on human data.
WGS_QC_THRESHOLDS = {
    "MappedFraction":    {"warn_low": 0.75, "fail_low": 0.50},
    "DuplicateFraction": {"warn_high": 0.20, "fail_high": 0.30},
    "MedianMAPQ":        {"warn_low": 30,   "fail_low": 20},
    "MedianCoverage":    {"warn_low": 25.0, "fail_low": 15.0},
    "SDCoverage":        {"warn_high": 10.0, "fail_high": 15.0},
    "GCContent":         {"warn_low": 0.38, "fail_low": 0.34,
                          "warn_high": 0.45, "fail_high": 0.50},
}

# Metrics reported as context; excluded from PASS/WARN/FAIL calculations.
WES_INFO_ONLY = {"Mapped", "DuplicateMarked", "SDCoverage",
                 "MedianInsertSize", "SDInsertSize"}
WGS_INFO_ONLY = {"Mapped", "DuplicateMarked", "MedianInsertSize", 
                 "SDInsertSize"}

# Order of metrics in report.
WES_METRIC_FIELDS = [
    "Mapped", "MappedFraction", "DuplicateMarked", "DuplicateFraction",
    "MedianMAPQ", "FractionInBed", "MedianCoverage", "SDCoverage",
    "EnrichmentOverBed", "MedianInsertSize", "SDInsertSize", "GCContent"
]
WGS_METRIC_FIELDS = [
    "Mapped", "MappedFraction", "DuplicateMarked", "DuplicateFraction",
    "MedianMAPQ", "MedianCoverage", "SDCoverage", "MedianInsertSize", 
    "SDInsertSize", "GCContent"
]

# Metrics glossary.
WES_GLOSSARY = """- MappedFraction: fraction of reads that aligned to the human reference genome.
- DuplicateFraction: fraction of reads that are PCR duplicates (higher = worse library complexity).
- MedianMAPQ: median mapping quality score (higher = reads mapped more confidently).
- FractionInBed: fraction of reads landing on exome target regions (on-target rate).
- EnrichmentOverBed: fold enrichment of target (exon) regions over background genome.
- MedianCoverage: median read depth across targeted exome regions.
- SDCoverage: variability in coverage depth across targets (INFO only, not thresholded).
- MedianInsertSize: median DNA fragment size in base pairs
- GCContent: fraction of bases that are G or C (typically ~0.45-0.52 for human exome; values between 0.35 and 0.60 are treated as acceptable)
- Mapped: total number of mapped reads (INFO only)
- DuplicateMarked: total number of duplicate reads (INFO only)"""
WGS_GLOSSARY = """- MappedFraction: fraction of reads that aligned to the human reference genome
- DuplicateFraction: fraction of reads that are PCR duplicates (higher = worse library complexity; PCR-free genome libraries are normally very low)
- MedianMAPQ: median mapping quality score (higher = reads mapped more confidently)
- MedianCoverage: median read depth across the whole genome (not a captured subset)
- SDCoverage: variability in read depth genome-wide. Higher = patchier, less uniform coverage, with more regions sequenced too shallowly to call variants reliably
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
        "glossary": WES_GLOSSARY
    },
    "wgs": {
        "label": "Whole Genome Sequencing (WGS)",
        "description": ("This is whole genome sequencing (WGS) data from a human sample "
                        "sequenced on Illumina at approximately 30x genome-wide coverage. "
                        "There is no capture or target enrichment step."),
        "thresholds": WGS_QC_THRESHOLDS,
        "info_only": WGS_INFO_ONLY,
        "fields": WGS_METRIC_FIELDS,
        "glossary": WGS_GLOSSARY
    },
}

# function to load QC JSON; can handle gzip-compressed input
def load_qc_json(path: str) -> dict:
    """
    Alfred outputs `*.json.gz` by default, and this script also accepts gunzipped `*.json`.
    Detects gzip by magic bytes rather than by file extension, so mislabelled 
    or renamed file is handled appropriately.
    """
    with open(path, "rb") as fh:
        is_gzip = fh.read(2) == b"\x1f\x8b"
    opener = gzip.open if is_gzip else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)

# function to evaluate metrics against thresholds
def evaluate_metrics(parsed_metrics: dict,
                     thresholds: dict = None,
                     info_only: set = None) -> dict:
    """
    Evaluates each metric against the supplied threshold set.
    Returns {metric_name: {"value": ..., "status": "PASS"|"WARN"|"FAIL"|"INFO"|"MISSING"}}.
    INFO-only metrics are passed through without evaluation.
    Metrics present in the threshold set but null in the JSON are marked MISSING and treated 
    as WARN. Metrics absent from the threshold set are not considered.
    Defaults to WES configuration.
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

# NODE 1 — ASSAY ROUTER
def route_assay(state: QCState) -> QCState:
    """
    Validates the requested assay type and writes it back to state.
    Runs first so that downstream nodes can rely on state["assay_type"]
    being a key of ASSAY_CONFIG.
    """
    assay = (state.get("assay_type") or "wes").strip().lower()
    if assay not in ASSAY_CONFIG:
        valid = ", ".join(sorted(ASSAY_CONFIG))
        raise ValueError(f"Unknown assay type '{assay}'. Valid values: {valid}")
    return {**state, "assay_type": assay}

def assay_branch(state: QCState) -> str:
    """Conditional-edge selector: returns the assay key."""
    return state["assay_type"]

def _parse_qc(state: QCState, assay: str) -> QCState:
    """Shared parse/evaluate logic, parameterised by assay type."""
    raw = state["raw_input"]
    cfg = ASSAY_CONFIG[assay]
    metrics = {"sample_id": raw.get("sample_id", "Unknown Sample")}
    for field in cfg["fields"]:
        metrics[field] = raw.get(field)
    evaluated = evaluate_metrics(metrics, cfg["thresholds"], cfg["info_only"])
    verdict = overall_pass_fail(evaluated)
    metrics["evaluated"] = evaluated
    metrics["overall_status"] = verdict
    metrics["assay_type"] = assay
    metrics["assay_label"] = cfg["label"]
    return {**state, "parsed_metrics": metrics, "pass_fail": verdict}

# NODE 2a — WES PARSE
def parse_qc_wes(state: QCState) -> QCState:
    """Extract and evaluate exome QC metrics."""
    return _parse_qc(state, "wes")

# NODE 2b — WGS PARSE
def parse_qc_wgs(state: QCState) -> QCState:
    """Extract and evaluate whole-genome QC metrics."""
    return _parse_qc(state, "wgs")

# NODE 3
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
        model_provider=state.get("model_provider") or None,
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

# NODE 4
# function: combine QC metrics & LLM summary into a clean human-readable report
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


# function: render the compiled graph as a Mermaid diagram for Markdown reports
def workflow_mermaid(assay: str | None = None) -> str:
    """
    Render the compiled LangGraph as a fenced Mermaid block for embedding in
    Markdown reports. Generated from the compiled graph.
    """
    try:
        body = build_graph().get_graph().draw_mermaid().strip()

        # format for improved visualization
        if body.startswith("---"):
            end = body.find("---", 3)
            if end != -1:
                body = body[end + 3:].lstrip()
    except Exception as exc:
        return f"\n## Workflow\n\n_Diagram unavailable: {type(exc).__name__}: {exc}_\n"

    if assay in ASSAY_CONFIG:
        body += (
            "\n    classDef taken fill:#d5f5d5,stroke:#2e7d32,stroke-width:2px;"
            f"\n    class parse_qc_{assay} taken;"
        )
    return f"\n## Workflow\n\n```mermaid\n{body}\n```\n"

# function: assemble the report as Markdown for rendered contexts
def format_report_markdown(state: QCState) -> str:
    """
    Markdown variant of the report. Uses a table so alignment survives proportional fonts,
    and embeds the workflow diagram.
    """
    metrics = state["parsed_metrics"]
    evaluated = metrics["evaluated"]
    icons = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️", "MISSING": "❓"}
    rows = []
    for metric, result in evaluated.items():
        value = result["value"]
        shown = value if value is not None else "N/A"
        rows.append(f"| {icons.get(result['status'], '')} | {metric} | {shown} | {result['status']} |")
    return f"""# QC Report — {metrics['sample_id']}

**Assay:** {metrics.get('assay_label', '')}  
**Overall Status:** {icons.get(state['pass_fail'], '')} {state['pass_fail']}

## Metrics Summary

|  | Metric | Value | Status |
|--|--------|-------|--------|
{chr(10).join(rows)}

Legend: ✅ PASS ⚠️ WARN ❌ FAIL ℹ️ INFO (not thresholded) ❓ MISSING

## Biological Interpretation

{state['llm_summary']}
{workflow_mermaid(metrics.get('assay_type'))}
"""

# function: Build LangGraph
@lru_cache(maxsize=1)
def build_graph():
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
    parser.add_argument("--input", help="Path to QC JSON file (.json or .json.gz)")
    parser.add_argument("--output", default=None, help="Optional path to save report")
    parser.add_argument("--model", default="gemini-2.5-flash", help="LLM model name (default: gemini-2.5-flash)")
    parser.add_argument("--provider", default="google_genai",
                        help="LLM provider: openai, anthropic, google_genai, etc. "
                             "Optional — init_chat_model infers from model name if not set.")
    parser.add_argument("--assay", default="wes", choices=["wes", "wgs"], type=str.lower,
                        help="Assay type, selects threshold set and interpretation context "
                             "(default: wes)")
    parser.add_argument("--diagram", action="store_true",
                        help="Print the workflow graph as a Mermaid diagram and exit")
    args = parser.parse_args()
    # Mermaid diagram is generated from the compiled graph, so it cannot drift
    # out of sync with the node and edge definitions above.
    if args.diagram:
        print(build_graph().get_graph().draw_mermaid())
        return
    if not args.input:
        parser.error("--input is required unless --diagram is given")
    qc_data = load_qc_json(args.input)
    graph = build_graph()
    result = graph.invoke({
        "raw_input": qc_data,
        "model_name": args.model,
        "model_provider": args.provider,
        "assay_type": args.assay,
    })
    print(result["output_report"])
    if args.output:
        if args.output.lower().endswith((".md", ".markdown")):
            text = format_report_markdown(result)
        else:
            text = result["output_report"]
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Report saved to {args.output}")

if __name__ == "__main__":
    main()