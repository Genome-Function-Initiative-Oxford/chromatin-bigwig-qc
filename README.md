# chromatin-bigwig-qc

**Machine learning–based quality control for chromatin accessibility bigWig tracks, grounded in biologically stable reference features.**

---

## Overview

**chromatin-bigwig-qc** is a machine learning–based framework for quality control (QC) of chromatin accessibility bigWig signal tracks (e.g. ATAC-seq, DNase-seq, CUT&RUN, CUT&Tag).
The method operates **directly on processed bigWig coverage files**, avoiding dependence on raw sequencing data, and produces a **continuous, interpretable quality score** for each track.

Rather than relying on ad hoc summary statistics alone, chromatin-bigwig-qc evaluates signal quality by measuring how well a track recovers **biologically stable genomic features**, such as:

- constitutively active promoters,
- CTCF binding sites,
- and high-confidence peak structure and enrichment patterns.

This makes the QC score robust across experiments, batches, and datasets, and particularly well suited for large-scale data integration and downstream machine learning workflows.

---

## Key ideas

1. **Operate on final signal representations**  
   Quality assessment is performed on bigWig coverage tracks, which are the standard input for many downstream analyses and public datasets.

2. **Use biologically grounded reference features**  
   Signal quality is assessed based on recovery of stable regulatory and architectural elements of the genome, rather than arbitrary thresholds.

3. **Leverage interpretable machine learning**  
   A gradient-boosted tree model (XGBoost) integrates multiple complementary signal-level metrics into a single, continuous QC score.

---

## Method summary

1. **Peak calling**  
   Peaks are identified using an ML-based peak caller (e.g. *LanceOtron*), which provides rich peak-level annotations beyond simple enrichment.

2. **Signal summarisation**  
   For each bigWig track, the tool computes summary statistics including peak counts, enrichment, shape, and coverage metrics.

3. **Feature integration with XGBoost**  
   These features are passed to a pre-trained XGBoost regression model, which outputs a continuous quality score.

All processing is performed **in memory**, and no intermediate files are written during prediction.

---

### Dependencies

- pyBigWig
- numpy
- pandas
- xgboost

---

## Basic usage

```python
from chromatin_bigwig_qc.pipeline import ChromatinBigwigQC

qc = ChromatinBigwigQC()
qc.load_model("model.json")

results = qc.predict_from_pairs([
    ("sample.bw", "sample_peaks.tsv")
])

print(results[["collection_name", "experiment_name", "prediction"]])
```

---

## Input requirements

### bigWig files
- Indexed bigWig format
- Chromosome naming must match peak calls (e.g. `chr1` vs `1`)

### Peak-call files

Peak-call files must be generated using **LanceOtron**. The framework assumes LanceOtron-style peak annotations and relies on several of its model-derived features to ensure consistent and biologically meaningful quality assessment.

LanceOtron produces a tab-delimited, BED-like file containing the following columns:


#### Column description (brief)
- `chrom`, `start`, `end`: Genomic coordinates of the peak region  
- `overall_peak_score`: LanceOtron confidence score for the peak  
- `shape_score`: Measure of conformity to an expected accessibility peak shape  
- `enrichment_score`: Signal enrichment relative to local background  
- `max_coverage_position`: Genomic position of maximal signal within the peak  
- `region_id`: Unique identifier for the peak region  
- `pvalue_*`: P-values computed at increasing genomic scales, capturing local and broader background significance  

Not all LanceOtron-derived columns are explicitly used in the current implementation of **chromatin-bigwig-qc**. However, users are expected to provide the complete, unfiltered LanceOtron output to ensure compatibility, reproducibility, and future extensibility of the framework.

---

## Design choices

- Prediction-only (no training)
- No disk writes during inference
- Feature order can be enforced for reproducibility
- Interpretable XGBoost model

---

## Use cases

- Large-scale QC of public chromatin accessibility datasets
- Filtering samples prior to integration
- Quality-aware downstream machine learning pipelines

---

## License

GPL-3.0

---

## Citation

If you use **chromatin-bigwig-qc** in your work, please cite the following preprint:

> Riva, S. G., Venkatesh, S. S., Gür, E. R., Starnieri, N., Sergeant, M., Baxter, M., & Hughes, J. R.  
> *CREST-GV: Cell types Ranking and Enrichment Score for selective Genetic Variants.*  
> medRxiv, 2025.07.29.  
> https://www.medrxiv.org/