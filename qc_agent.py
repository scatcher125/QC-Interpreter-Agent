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

# Define QC thresholds
# - Thresholds are calibrated for 30x Illumina whole exome sequencing on human data.
# - Fraction metrics are evaluated as 0-1, not percentages.
QC_THRESHOLDS = {
    "MappedFraction": {"warn_low":0.75,"fail_low":0.50},            # Alignment quality
    "DuplicateFraction": {"warn_high":0.35,"fail_high":0.50 },      # duplicates/lib complexity
    "MedianMAPQ": {"warn_low":30,"fail_low":20},                    # mapping quality
    "FractionInBed": {"warn_low":0.70,"fail_low":0.35},             # on-target rate in WES
    "EnrichmentOverBed": {"warn_low":2.0,"fail_low":0.9},           # fold enrichment of exons over whole genome
    "MedianCoverage": {"warn_low":15.0,"fail_low":5.0},             # read depth in exons
    "GCContent": {"warn_low":0.35,"fail_low":0.20,
                  "warn_high":0.60,"fail_high":0.80}                # GC content (contamination/amplification bias)
    }

# Metrics passed to report as context and do not determine PASS/WARN/FAIL.
INFO_ONLY = {"Mapped", "DuplicateMarked", "SDCoverage", "MedianInsertSize", "SDInsertSize"}

# function to evaluate metrics against thresholds
def evaluate_metrics(parsed_metrics: dict) -> dict:
    """
    Evaluates each metric against QC_THRESHOLDS.
    Returns a dict of {metric_name: {"value": ..., "status": "PASS"|"WARN"|"FAIL"|"INFO"|"MISSING"}}.
    INFO_ONLY metrics are passed through without evaluation.
    Metrics absent from the JSON are marked MISSING and treated as WARN in rollup.
    """
    results = {}

    for metric, value in parsed_metrics.items():
        if metric == "sample_id":
            continue

        # Pass through informational metrics without thresholding
        if metric in INFO_ONLY:
            results[metric] = {"value": value, "status": "INFO"}
            continue

        # Skip metrics not in thresholds
        if metric not in QC_THRESHOLDS:
            continue

        # Flag missing values — may indicate BED file was not supplied to alfred
        if value is None:
            results[metric] = {"value": None, "status": "MISSING"}
            continue

        thresholds = QC_THRESHOLDS[metric]
        status = "PASS"

        # Check low-end thresholds (metrics where lower is worse)
        if "fail_low" in thresholds and value < thresholds["fail_low"]:
            status = "FAIL"
        elif "warn_low" in thresholds and value < thresholds["warn_low"]:
            status = "WARN"

        # Check high-end thresholds (metrics where higher is worse)
        elif "fail_high" in thresholds and value > thresholds["fail_high"]:
            status = "FAIL"
        elif "warn_high" in thresholds and value > thresholds["warn_high"]:
            status = "WARN"

        # Re-check high-end for metrics with bounds in both directions
        # (e.g. GCContent)
        if status == "PASS":
            if "fail_high" in thresholds and value > thresholds["fail_high"]:
                status = "FAIL"
            elif "warn_high" in thresholds and value > thresholds["warn_high"]:
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


# function: parse QC metrics from Alfred JSON output file; apply thresholds.
def parse_qc_metrics(state: QCState) -> QCState:
    """Extract and evaluate QC metrics against thresholds."""
    raw = state["raw_input"]
    sample_id = raw.get("sample_id", "Unknown Sample")

    metrics = {
        "sample_id": sample_id,
        "Mapped": raw.get("Mapped"),
        "MappedFraction": raw.get("MappedFraction"),
        "DuplicateMarked": raw.get("DuplicateMarked"),
        "DuplicateFraction": raw.get("DuplicateFraction"),
        "MedianMAPQ": raw.get("MedianMAPQ"),
        "FractionInBed": raw.get("FractionInBed"),
        "MedianCoverage": raw.get("MedianCoverage"),
        "SDCoverage": raw.get("SDCoverage"),
        "EnrichmentOverBed": raw.get("EnrichmentOverBed"),
        "MedianInsertSize": raw.get("MedianInsertSize"),
        "SDInsertSize": raw.get("SDInsertSize"),
        "GCContent": raw.get("GCContent"),
    }

    evaluated = evaluate_metrics(metrics)
    verdict = overall_pass_fail(evaluated)

    metrics["evaluated"] = evaluated
    metrics["overall_status"] = verdict

    return {**state, "parsed_metrics": metrics, "pass_fail": verdict}


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

This is whole exome sequencing (WES) data from a human sample sequenced on Illumina with 30x target coverage.

Sample ID: {metrics['sample_id']}
Overall Status: {metrics['overall_status']}

QC Metrics (status: PASS/WARN/FAIL/INFO/MISSING):
{chr(10).join(metric_lines)}

Metric notes:
- MappedFraction: fraction of reads that aligned to the human reference genome
- DuplicateFraction: fraction of reads that are PCR duplicates (higher = worse library complexity)
- MedianMAPQ: median mapping quality score (higher = reads mapped more confidently)
- FractionInBed: fraction of reads landing on exome target regions (on-target rate)
- EnrichmentOverBed: fold enrichment of target (exon) regions over background genome
- MedianCoverage: median read depth across targeted exome regions
- SDCoverage: variability in coverage depth across targets (INFO only, not thresholded)
- MedianInsertSize: median DNA fragment size in base pairs
- GCContent: fraction of bases that are G or C (expected ~0.45-0.52 for human exome)
- Mapped: total number of mapped reads (INFO only)
- DuplicateMarked: total number of duplicate reads (INFO only)

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

    report = f"""
================================================================================
  QC REPORT — {metrics['sample_id']}
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
    graph = StateGraph(QCState)

    graph.add_node("parse_qc", parse_qc_metrics)
    graph.add_node("llm_summary", generate_llm_summary)
    graph.add_node("format_report", format_report)

    graph.set_entry_point("parse_qc")
    graph.add_edge("parse_qc", "llm_summary")
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
    args = parser.parse_args()

    with open(args.input) as f:
        qc_data = json.load(f)

    graph = build_graph()
    result = graph.invoke({
        "raw_input": qc_data,
        "model_name": args.model,
        "model_provider": args.provider,
    })

    print(result["output_report"])

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result["output_report"])
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
