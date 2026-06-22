# QC Interpreter Agent

Whole exome sequencing captures the protein-coding regions of the human genome and is  widely used in disease research and clinical genomics — but raw sequencing data must pass rigorous quality checks before any biological conclusions can be drawn.

This agent is a LangGraph-based agentic workflow that parses sequencing Quality Control (QC) metrics and uses an LLM to generate plain-English summaries for non-computational research collaborators. It is built to support Illumina 30x coverage Whole Exome Sequencing (WES) data alignment assessed using Alfred — an efficient and versatile BAM (Binary Alignment Map) alignment QC tool.

## Purpose

In genomics research, QC reports are full of technical metrics (mapping rates, duplication rates, coverage depth) that are essential for bioinformaticians but opaque to bench scientists and clinicians. This agent bridges that gap by automatically interpreting QC outputs and producing clear, actionable summaries — keeping scientific discussions focused on biology rather than the pipeline.

## Architecture

Three LangGraph nodes in sequence:

```
[parse_qc_metrics] → [generate_llm_summary] → [format_report]
```

1. **parse_qc_metrics** — Loads QC metrics from JSON, evaluates against established thresholds, and flags any issues
2. **generate_llm_summary** — Sends structured metrics to an LLM with a bioinformatics-aware prompt to generate a plain-English interpretation
3. **format_report** — Assembles a clean, human-readable report combining raw metrics and the LLM summary

## Setup

Clone this repository using instructions found here [GitHub Docs: Cloning a Repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository). Navigate to the main directory in your user interface of choice.

Load required Python packages.

```bash
pip install -r requirements.txt
```

The agent uses Google Gemini by default (free tier). Set the appropriate API key as an environment variable for your chosen provider:

```bash
# Google Gemini (default, free tier)
export GOOGLE_API_KEY="your-key-here"        # get from aistudio.google.com

# OpenAI
export OPENAI_API_KEY="your-key-here"        # get from platform.openai.com

# Anthropic
export ANTHROPIC_API_KEY="your-key-here"     # get from console.anthropic.com
```

On Windows (PowerShell):
```powershell
$env:GOOGLE_API_KEY = "your-key-here"
```

Note: Google Gemini is the default LLM provider because it offers a free API tier through Google AI Studio with no billing or credit card required, making this tool accessible without any upfront cost. OpenAI and Anthropic are supported as alternatives but require paid API access.

## Usage

```bash
# Basic usage (uses gemini-2.5-flash by default)
python qc_agent.py --input sample_qc.json

# Save report to file
python qc_agent.py --input sample_qc.json --output report.txt

# Use a different model
python qc_agent.py --input sample_qc.json --model gemini-2.5-flash
python qc_agent.py --input sample_qc.json --model gpt-4o --provider openai
python qc_agent.py --input sample_qc.json --model claude-3-5-sonnet-20241022 --provider anthropic
```

## Input Format

Generate the input file by running Alfred with a BED file of exome targets:

```bash
alfred qc -r ref.fa -b targets.bed -j qc.json.gz sample.bam
```

The agent expects an unzipped `.json` file. Alfred outputs `.json.gz` by default — remember to unzip before running:

```bash
gunzip qc.json.gz
```

The following fields are used (all optional except `sample_id`):

```json
{
    "sample_id": "SAMPLE_003_low_ontarget",
    "Mapped": 82000000,
    "MappedFraction": 0.955,
    "DuplicateMarked": 9840000,
    "DuplicateFraction": 0.120,
    "MedianMAPQ": 56,
    "FractionInBed": 0.621,
    "MedianCoverage": 44.7,
    "SDCoverage": 41.2,
    "EnrichmentOverBed": 1.4,
    "MedianInsertSize": 178,
    "SDInsertSize": 55.1,
    "GCContent": 0.409
}
```

`FractionInBed` and `EnrichmentOverBed` require the `-b` flag. If absent, they will be flagged as MISSING and contribute a WARN to the overall verdict.

## QC Thresholds

Thresholds are calibrated for 30x Illumina WES on human data. Fraction metrics are 0–1 (not percentages).

| Metric | WARN | FAIL |
|--------|------|------|
| MappedFraction | <0.75 | <0.50 |
| DuplicateFraction | >0.35 | >0.50 |
| MedianMAPQ | <30 | <20 |
| FractionInBed | <0.70 | <0.35 |
| EnrichmentOverBed | <2.0 | <0.9 |
| MedianCoverage | <15x | <5x |
| GCContent | <0.35 or >0.60 | <0.20 or >0.80 |

The following metrics are reported as context only and do not affect the verdict: `Mapped`, `DuplicateMarked`, `SDCoverage`, `MedianInsertSize`, `SDInsertSize`.

## Example Output

```
================================================================================
  QC REPORT — SAMPLE_003_low_ontarget
  Overall Status: ⚠️ WARN
================================================================================

METRICS SUMMARY
---------------
  ✅ MappedFraction         0.955
  ✅ DuplicateFraction      0.12
  ✅ MedianMAPQ             56
  ⚠️ FractionInBed          0.621
  ⚠️ EnrichmentOverBed      1.4
  ✅ MedianCoverage         44.7
  ✅ GCContent              0.409
  ℹ️ Mapped                 82000000
  ℹ️ DuplicateMarked        9840000
  ℹ️ SDCoverage             41.2
  ℹ️ MedianInsertSize       178
  ℹ️ SDInsertSize           55.1

  Legend: ✅ PASS  ⚠️ WARN  ❌ FAIL  ℹ️ INFO (not thresholded)  ❓ MISSING

BIOLOGICAL INTERPRETATION
-----------------------------------
This sample shows good overall alignment quality with 95.5% of reads mapping
to the human genome and low duplication. However, only 62.1% of reads landed
on the targeted exome regions, with modest enrichment over background (1.4x vs
the expected >2x). This suggests the capture step was partially inefficient.
Variant calling can proceed but sensitivity for detecting variants in
lower-coverage exons may be reduced. Consider reviewing the capture protocol
for future runs.

================================================================================
```

## Extensions (future work)

- Support Whole Genome Sequencing (WGS)
- Batch processing across multiple samples
- Gzip input support (`*.json.gz`) directly
- Mermaid workflow diagram: Once the program supports more analysis types, documenting the exact nodes for a workflow will become more critical.
- Email notification integration
- Interactive HTML report output to improve visual appeal
- RAG over internal QC history to contextualize current sample against cohort
