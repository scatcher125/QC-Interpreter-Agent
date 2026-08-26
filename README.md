# QC Interpreter Agent

Whole exome sequencing (WES) captures protein-coding regions of the human genome, and whole genome sequencing (WGS) captures the entire genome including non-coding regions. Both are widely used in disease research and clinical genomics, but raw sequencing data must pass rigorous quality checks before any biological conclusions can be drawn.

This agent is a LangGraph-based agentic workflow that parses sequencing Quality Control (QC) metrics and uses a LLM to generate plain-English summaries for non-computational research collaborators. Currently, it is designed to support Illumina 30x coverage Whole Exome Sequencing (WES) and Whole Genome Sequencing (WGS) data alignment assessed using Alfred — an efficient and versatile BAM (Binary Alignment Map) alignment QC tool, with expansion plans in the near future.

## Purpose

In genomics research, QC reports summarize technical metrics (mapping rates, duplication rates, coverage depth) that are essential for bioinformaticians but opaque to bench scientists and clinicians. This agent bridges that gap by interpreting QC outputs and producing clear, actionable summaries in order to keep scientific discussions focused on biology rather than the pipeline.
WES and WGS have different quality expectations, and the `--assay` parameter routes the sample down an assay-specific branch with its own thresholds, metric list, and LLM glossary.

## Architecture

Five LangGraph nodes, branching on assay type:
```
                ┌→ [parse_qc_wes] ┐
[route_assay] ──┤                 ├→ [llm_summary] → [format_report]
                └→ [parse_qc_wgs] ┘
```

1. **route_assay** — Parses JSON file, validates inputs against specified assay configuration, and writes it to the shared "state" for downstream processing.
2. **parse_qc_wes** — Loads exome QC metrics from JSON and evaluates against capture-aware WES thresholds to flag any issues.
3. **parse_qc_wgs** — Loads genome QC metrics from JSON and evaluates against WGS thresholds to flag any issues.
4. **llm_summary** — Sends structured metrics and assay-specific glossary to the LLM with a bioinformatics-aware prompt to generate a plain-English interpretation.
5. **format_report** — Assembles a clean, human-readable report combining raw metrics and the LLM summary.

## Setup
Clone this repository using instructions found here [GitHub Docs: Cloning a Repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository). 

Within your user interface of choice, load required Python packages.

```bash
pip install -r requirements.txt
```

The agent uses Google Gemini by default since it offers a free tier through Google AI Studio, but parameters can be adjusted to specify other LLMs supported by LangChain (OpenAI, Anthropic etc.). Set the selected API key as an environment variable:

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

## Usage

```bash
# Basic usage (uses gemini-2.5-flash and WES thresholds by default)
python qc_agent.py --input sample_qc.json.gz
python qc_agent.py --input sample_qc.json
# Whole Genome Sequencing
python qc_agent.py --input sample_wgs_qc.json.gz --assay wgs
# Save report to file
python qc_agent.py --input sample_qc.json.gz --output report.txt
# Use a different model
python qc_agent.py --input sample_qc.json.gz --model gemini-2.5-flash
python qc_agent.py --input sample_qc.json.gz --model gpt-4o --provider openai
python qc_agent.py --input sample_qc.json.gz --model claude-3-5-sonnet-20241022 --provider anthropic
```

`--assay` Accepts `wes` (default) for the WES thresholds, or `wgs` for WGS, case insensitive.

## Input Format

Generate the input file by running Alfred. For WES, pass a BED file of exome targets:

```bash
alfred qc -r ref.fa -b targets.bed -j qc.json.gz sample.bam
```

For WGS, omit `-b` since there are no capture targets:

```bash
alfred qc -r ref.fa -j qc.json.gz sample.bam
```
The agent can take in the default Alfred output `.json.gz` or an unzipped `.json` file.

All fields are optional. `sample_id` falls back to `Unknown Sample` if absent, and any missing thresholded metric is reported as MISSING and contributes a WARN. `FractionInBed` and `EnrichmentOverBed` require the `-b` flag. In WES mode, if absent they will be flagged as MISSING and contribute a WARN to the overall verdict.


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

## QC Thresholds

Thresholds are calibrated for 30x Illumina human data and selected by `--assay`. Fraction metrics are 0–1.

**WES**
| Metric | WARN | FAIL |
|--------|------|------|
| MappedFraction | <0.75 | <0.50 |
| DuplicateFraction | >0.35 | >0.50 |
| MedianMAPQ | <30 | <20 |
| FractionInBed | <0.70 | <0.35 |
| EnrichmentOverBed | <2.0 | <0.9 |
| MedianCoverage | <15x | <5x |
| GCContent | <0.35 or >0.60 | <0.20 or >0.80 |

**WGS**
| Metric | WARN | FAIL |
|--------|------|------|
| MappedFraction | <0.75 | <0.50 |
| DuplicateFraction | >0.20 | >0.30 |
| MedianMAPQ | <30 | <20 |
| MedianCoverage | <25x | <15x |
| SDCoverage | >10x | >15x |
| GCContent | <0.38 or >0.45 | <0.34 or >0.50 |

The following metrics are reported as context only and do not affect the verdict: `Mapped`, `DuplicateMarked`, `SDCoverage`, `MedianInsertSize`, `SDInsertSize`. `SDCoverage` is context-only in WES mode but thresholded in WGS mode.

## Example Output

```
================================================================================
  QC REPORT — SAMPLE_003_low_ontarget
  Assay: Whole Exome Sequencing (WES)
  Overall Status: ⚠️ WARN
================================================================================
METRICS SUMMARY
---------------
  ℹ️ Mapped                 82000000
  ✅ MappedFraction         0.955
  ℹ️ DuplicateMarked        9840000
  ✅ DuplicateFraction      0.12
  ✅ MedianMAPQ             56
  ⚠️ FractionInBed          0.621
  ✅ MedianCoverage         44.7
  ℹ️ SDCoverage             41.2
  ⚠️ EnrichmentOverBed      1.4
  ℹ️ MedianInsertSize       178
  ℹ️ SDInsertSize           55.1
  ✅ GCContent              0.409
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

- ~~Support `.json.gz` input files.~~
- ~~Process WGS samples~~
- ~~Markdown report output to improve visual appeal~~
- Batch processing across multiple samples.
- ~~Mermaid workflow diagram: Once the program supports more analysis types, documenting the exact nodes for a workflow will become more critical.~~
- Email notification integration.
- RAG over past Alfred QC outputs to contextualize current sample against cohort.
- Node to support Picard QC metrics from WES
- Node to support Picard QC metrics from WES
