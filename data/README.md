# External data dependencies

Nothing under `data/` is tracked in git (`data/raw/` and `data/interim/` are
gitignored) - these are the external inputs the pipeline reads, and the
regenerated intermediate outputs it writes. Every path below is defined once
in `src/bigwig_qc/config.py` and overridable with the listed environment
variable, so the pipeline can point at a different mount without code changes.

| Data | Default path | Override env var | Size |
|---|---|---|---|
| bigWigs + unfiltered LanceOtron peak calls, per collection | `/project/Wellcome_Discovery/datashare/CREST-GV_collection/` | `BIGWIG_QC_COLLECTION_ROOT` | large, per-collection |
| Chromosome name remap (RefSeq accession -> `chr*`) | `/project/Wellcome_Discovery/shared/00_publicly_available/06_dbSNP/bcftools_rename_file.txt` | `BIGWIG_QC_CHROM_RENAME_FILE` | ~23 KB |
| RefSeq hg38 gene annotation (GTF) | `/project/Wellcome_Discovery/shared/00_publicly_available/14_RefSeq/ncbi_dataset/data/GCF_000001405.40/genomic.gtf` | `BIGWIG_QC_REFSEQ_GTF` | ~1.9 GB |
| Housekeeping gene list (GTEx sheet), PLOS Comp Biol 10.1371/journal.pcbi.1010295 supplementary table S014 | `data/external/pcbi.1010295.s014.xlsx` (repo-relative) | `BIGWIG_QC_HOUSEKEEPING_XLSX` | ~740 KB |
| Constant CTCF sites (recurrence across samples) | `/project/Wellcome_Discovery/sriva/Git/CTCF_collection/metadata/intersect.bed` | `BIGWIG_QC_CTCF_BED` | ~25 MB |

The first three rows are shared cluster infrastructure (Wellcome Discovery
project storage) and only resolve on the Oxford CBRG/Wellcome cluster. The
housekeeping gene list is small and public - vendor it into this repo under
`data/external/` (not tracked in git, so it isn't copied in automatically and
the repository stays small by default) or point
`BIGWIG_QC_HOUSEKEEPING_XLSX` at a copy elsewhere.

`data/interim/` is where `scripts/run_features.py` / `run_targets.py` /
`build_dataset.py` write their outputs (see the top-level README's pipeline
quickstart) before `scripts/train_model.py` consumes them.
