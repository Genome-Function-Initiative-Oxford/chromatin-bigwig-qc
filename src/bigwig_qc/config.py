"""Central configuration: external data locations and fixed constants.

All external paths resolve to the Oxford CBRG/Wellcome cluster by default and
are overridable via environment variables, so the pipeline can run against a
different mount (or a local copy of the reference files) without code changes.
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var_name: str, default: str) -> Path:
    return Path(os.environ.get(var_name, default))


# Per-experiment bigWigs and LanceOtron peak calls (see data/README.md).
BIGWIG_COLLECTION_ROOT = _env_path(
    "BIGWIG_QC_COLLECTION_ROOT",
    "/project/Wellcome_Discovery/datashare/CREST-GV_collection",
)
PEAKS_UNFILTERED_ROOT = Path(
    os.environ.get("BIGWIG_QC_PEAKS_ROOT", str(BIGWIG_COLLECTION_ROOT / "peaks_unfiltered"))
)

# Chromosome name remap used to align the RefSeq GTF with the bigWig/peak chrom names.
CHROM_RENAME_FILE = _env_path(
    "BIGWIG_QC_CHROM_RENAME_FILE",
    "/project/Wellcome_Discovery/shared/00_publicly_available/06_dbSNP/bcftools_rename_file.txt",
)

# RefSeq gene annotation (hg38), used to derive constant-promoter TSS windows.
REFSEQ_GTF = _env_path(
    "BIGWIG_QC_REFSEQ_GTF",
    "/project/Wellcome_Discovery/shared/00_publicly_available/14_RefSeq/ncbi_dataset/data/GCF_000001405.40/genomic.gtf",
)

# Housekeeping gene list (GTEx sheet), PLOS Comp Biol 10.1371/journal.pcbi.1010295
# supplementary table S014. Small and public; vendor it into data/external/ (see
# data/README.md) or point BIGWIG_QC_HOUSEKEEPING_XLSX at a copy elsewhere.
HOUSEKEEPING_GENES_XLSX = _env_path(
    "BIGWIG_QC_HOUSEKEEPING_XLSX",
    str(_REPO_ROOT / "data" / "external" / "pcbi.1010295.s014.xlsx"),
)

# Constant CTCF sites (recurrence across samples), from the CTCF_collection repo.
CONSTANT_CTCF_BED = _env_path(
    "BIGWIG_QC_CTCF_BED",
    "/project/Wellcome_Discovery/sriva/Git/CTCF_collection/metadata/intersect.bed",
)

# "pancreatic_pbmc" excluded: known experimental problems with this
# collection, corroborated by E2's leave-one-collection-out result (held out,
# it scored R2=-3.23, Spearman rho=0.14 - near-chance rank correlation). Its
# pbmc_HemaCare tracks have mean_peak_score averaging 0.09, far below the
# 0.52 dataset-wide average (only 65/490 other tracks score that low), which
# is consistent with genuinely noisy peak calls rather than a modelling
# artefact. See results/e2/loco_metrics_per_collection.tsv.
COLLECTION_NAMES = [
    "H1_hESCs",
    "catlas_fetal",
    "CAD",
    "immune_cell",
    "MPAL",
    "catlas_adult",
    "ludwig2019",
    "super_PBMC",
    "calderon",
    "Days7_10_13_17",
]

AUTOSOMES_AND_SEX_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

PROMOTER_WINDOW_BP = 100
CONSTANT_CTCF_MIN_RECURRENCE = 0.99
PEAK_SCORE_THRESHOLDS = (0.5, 0.95)
PEAK_CALL_SUFFIX = "_L-tron.bed"

# E1 degradation benchmark (bigwig_qc.degradation): fixed-width bin used to
# rasterise a bigWig's signal before thinning/noise-injection and to write the
# degraded bigWig back out. Coarser than base-pair resolution but on the same
# order as LanceOtron's own default 400 bp candidate-peak window.
DEGRADATION_BIN_SIZE_BP = 100
DOWN_SAMPLE_FRACTIONS = (0.50, 0.25, 0.10, 0.05, 0.01)
NOISE_INJECTION_FRACTIONS = (0.10, 0.30, 0.50, 0.80)
DEGRADATION_SEEDS = (0, 1, 2)
N_SEED_TRACKS = 30
