# QC Interpreter Agent

Whole exome sequencing captures the protein-coding regions of the human genome, and whole genome sequencing captures the genome in its entirety — both are widely used in disease research and clinical genomics, but raw sequencing data must pass rigorous quality checks before any biological conclusions can be drawn.
This agent is a LangGraph-based agentic workflow that parses sequencing Quality Control (QC) metrics and uses an LLM to generate plain-English summaries for non-computational research collaborators. It is built to support Illumina 30x coverage Whole Exome Sequencing (WES) and Whole Genome Sequencing (WGS) data alignment assessed using Alfred — an efficient and versatile BAM (Binary Alignment Map) alignment QC tool.


## Purpose

In genomics research, QC reports are full of technical metrics (mapping rates, duplication rates, coverage depth) that are essential for bioinformaticians but opaque to bench scientists and clinicians. This agent bridges that gap by automatically interpreting QC outputs and producing clear, actionable summaries — keeping scientific discussions focused on biology rather than the pipeline.
WES and WGS have different quality expectations, so `--assay` routes the sample down an assay-specific branch with its own thresholds, metric list, and LLM glossary.


## Architecture

Five LangGraph nodes, branching on assay type:
```
                ┌→ [parse_qc_wes] ┐
[route_assay] ──┤                 ├→ [llm_summary] → [format_report]
                └→ [parse_qc_wgs] ┘
```

1. **route_assay** — Normalizes the requested assay against a table of accepted aliases and writes the canonical key (`wes` or `wgs`) to state, so downstream nodes can assume a valid value
2. **parse_qc_wes** — Loads exome metrics from JSON, evaluates against the capture-aware WES thresholds, and flags any issues
3. **parse_qc_wgs** — Loads genome metrics, skips capture metrics entirely, derives `CoverageCV`, and evaluates against WGS thresholds
4. **llm_summary** — Sends structured metrics and the assay-specific glossary to an LLM with a bioinformatics-aware prompt to generate a plain-English interpretation
5. **format_report** — Assembles a clean, human-readable report combining raw metrics and the LLM summary
Both parse nodes share one `_parse_qc` implementation parameterized by assay, so the two paths can't drift apart in their evaluation or rollup logic.


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
# Basic usage (uses gemini-2.5-flash and WES thresholds by default)
python qc_agent.py --input sample_qc.json
# Whole Genome Sequencing
python qc_agent.py --input sample_wgs_qc.json --assay wgs
# Save report to file
python qc_agent.py --input sample_qc.json --output report.txt
# Use a different model
python qc_agent.py --input sample_qc.json --model gemini-2.5-flash
python qc_agent.py --input sample_qc.json --model gpt-4o --provider openai
python qc_agent.py --input sample_qc.json --model claude-3-5-sonnet-20241022 --provider anthropic
```

`--assay` accepts `wes` (default), `exome`, `wxs`, and `panel` for the WES thresholds, or `wgs` and `genome` for WGS. `panel` is a convenience alias only — panels are usually sequenced much deeper than 30x, so the WES coverage thresholds will be lenient.


## Input Format

Generate the input file by running Alfred. For WES, pass a BED file of exome targets:

```bash
alfred qc -r ref.fa -b targets.bed -j qc.json.gz sample.bam
```

For WGS, omit `-b` — there are no capture targets:

```bash
alfred qc -r ref.fa -j qc.json.gz sample.bam
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

`FractionInBed` and `EnrichmentOverBed` require the `-b` flag. In WES mode, if absent they will be flagged as MISSING and contribute a WARN to the overall verdict. In WGS mode they are never read, so their absence carries no penalty.
`CoverageCV` is derived rather than supplied — it is computed as `SDCoverage / MedianCoverage`. If either input is missing or the median is non-positive it resolves to MISSING and contributes a WARN, rather than raising.


## QC Thresholds

Thresholds are calibrated for 30x Illumina human data and selected by `--assay`. Fraction metrics are 0–1 (not percentages).

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
| CoverageCV | >0.35 | >0.50 |
| GCContent | <0.38 or >0.45 | <0.34 or >0.50 |

The WGS set differs deliberately: capture metrics are absent rather than demoted, since there's no capture step to assess; duplication is tighter because PCR-free genome libraries typically run 1–10%; coverage and GC are recentered on genome-wide values; and `CoverageCV` is promoted to a thresholded metric, because coverage evenness is far more diagnostic genome-wide than over small capture targets.
The following metrics are reported as context only and do not affect the verdict: `Mapped`, `DuplicateMarked`, `SDCoverage`, `MedianInsertSize`, `SDInsertSize`. `SDCoverage` stays INFO-only even in WGS mode — its normalized form, `CoverageCV`, carries the graded signal instead.


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

- Batch processing across multiple samples
- Gzip input support (`*.json.gz`) directly
- Mermaid workflow diagram: Once the program supports more analysis types, documenting the exact nodes for a workflow will become more critical.
- Email notification integration
- Interactive HTML report output to improve visual appeal
- RAG over internal QC history to contextualize current sample against cohort