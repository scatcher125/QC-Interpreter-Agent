# QC Report — SAMPLE_001_clean

**Assay:** Whole Exome Sequencing (WES)  
**Overall Status:** ✅ PASS

## Metrics Summary

|  | Metric | Value | Status |
|--|--------|-------|--------|
| ℹ️ | Mapped | 87500000 | INFO |
| ✅ | MappedFraction | 0.971 | PASS |
| ℹ️ | DuplicateMarked | 8750000 | INFO |
| ✅ | DuplicateFraction | 0.1 | PASS |
| ✅ | MedianMAPQ | 58 | PASS |
| ✅ | FractionInBed | 0.821 | PASS |
| ✅ | MedianCoverage | 87.3 | PASS |
| ℹ️ | SDCoverage | 32.1 | INFO |
| ✅ | EnrichmentOverBed | 18.4 | PASS |
| ℹ️ | MedianInsertSize | 185 | INFO |
| ℹ️ | SDInsertSize | 52.3 | INFO |
| ✅ | GCContent | 0.412 | PASS |

Legend: ✅ PASS ⚠️ WARN ❌ FAIL ℹ️ INFO (not thresholded) ❓ MISSING

## Biological Interpretation

Good news! Your sample, SAMPLE_001_clean, passed all quality control checks with excellent results. We see a high proportion of your sequencing data mapped correctly to the human genome, and the sequencing effort was highly focused on the exome regions you're interested in. Crucially, the median coverage across your target regions is 87x, which is well above our 30x target, ensuring robust and reliable detection of genetic variants. You can confidently proceed with your downstream analysis, such as variant calling, without any concerns about data quality.

## Workflow

```mermaid
graph TD;
	__start__([<p>__start__</p>]):::first
	route_assay(route_assay)
	parse_qc_wes(parse_qc_wes)
	parse_qc_wgs(parse_qc_wgs)
	llm_summary(llm_summary)
	format_report(format_report)
	__end__([<p>__end__</p>]):::last
	__start__ --> route_assay;
	llm_summary --> format_report;
	parse_qc_wes --> llm_summary;
	parse_qc_wgs --> llm_summary;
	route_assay -. &nbsp;wes&nbsp; .-> parse_qc_wes;
	route_assay -. &nbsp;wgs&nbsp; .-> parse_qc_wgs;
	format_report --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
    classDef taken fill:#d5f5d5,stroke:#2e7d32,stroke-width:2px;
    class parse_qc_wes taken;
```

