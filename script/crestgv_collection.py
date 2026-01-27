from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
import glob, sys
import pandas as pd

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

from chromatin_bigwig_qc.pipeline import ChromatinBigwigQC

# Base paths
peaks_base = "/project/Wellcome_Discovery/datashare/CREST-GV_collection/peaks_unfiltered"
bigwigs_base = "/project/Wellcome_Discovery/datashare/CREST-GV_collection"

# ---- 1) Collect peaks: all files inside each collection folder ----
# Adjust this glob if your peak files are in subfolders (e.g. "**/*")
peak_files = glob.glob(f"{peaks_base}/*/*")  # files directly under peaks_unfiltered/<collection>/
# If peaks are nested, use:
# peak_files = glob.glob(f"{peaks_base}/**/*", recursive=True)

peaks_by_collection = defaultdict(list)
for p in peak_files:
    pth = Path(p)
    collection = pth.parent.name  # peaks_unfiltered/<collection>/<file>
    if pth.is_file():
        peaks_by_collection[collection].append(str(pth))

# ---- 2) Collect bigwigs: all files under each collection's bigwigs folder ----
bigwig_files = glob.glob(f"{bigwigs_base}/*/bigwigs/*")

bigwigs_by_collection = defaultdict(list)
for bw in bigwig_files:
    bw_path = Path(bw)
    collection = bw_path.parents[1].name  # CREST-GV_collection/<collection>/bigwigs/<file>
    if bw_path.is_file():
        bigwigs_by_collection[collection].append(str(bw_path))

# ---- 3) Merge into one nested dict (sorted by filename for easy checking) ----
all_collections = sorted(set(peaks_by_collection) | set(bigwigs_by_collection))

collection_dict = {
    c: {
        "peaks": sorted(
            peaks_by_collection.get(c, []),
            key=lambda p: Path(p).name
        ),
        "bigwigs": sorted(
            bigwigs_by_collection.get(c, []),
            key=lambda p: Path(p).name
        ),
    }
    for c in all_collections
}

for collection, d in collection_dict.items():
    peaks = d["peaks"]
    bigwigs = d["bigwigs"]

    if not peaks and not bigwigs:
        continue

    print(f"\n=== {collection} ===")

    for i, (peak, bw) in enumerate(zip_longest(peaks, bigwigs, fillvalue=None), start=1):
        bw_name = bw if bw else "MISSING"
        peak_name = peak if peak else "MISSING"
        
        print(f"  [{i}]")
        print(f"    peaks : {peak_name}")
        print(f"    bw    : {bw_name}")

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "../Model/model.json"  # adjust
qc = ChromatinBigwigQC()
qc.load_model(MODEL_PATH)

# -----------------------------
# Helper: infer cell type name
# -----------------------------
def infer_cell_type(bw_path: str, peak_path: str) -> str:
    """
    Prefer bigwig filename stem as the cell type name (without extension).
    Fallback to peak filename stem if needed.
    """
    if bw_path and bw_path != "MISSING":
        return Path(bw_path).stem
    if peak_path and peak_path != "MISSING":
        return Path(peak_path).stem
    return "UNKNOWN"

# -----------------------------
# Run predictions for all pairs
# -----------------------------
all_rows = []

for collection, d in collection_dict.items():
    peaks = d["peaks"]
    bigwigs = d["bigwigs"]

    if not peaks and not bigwigs:
        continue

    for peak, bw in zip_longest(peaks, bigwigs, fillvalue=None):
        # Skip incomplete pairs
        if peak is None or bw is None:
            continue

        cell_type = infer_cell_type(bw, peak)

        # Predict for this single pair
        out = qc.predict_from_pairs([(bw, peak)])  # returns df with prediction + qc_label
        # print(out)

        # Add metadata columns
        out.insert(0, "cell_type", cell_type)
        out.insert(0, "collection", collection)
        # out["bigwig_path"] = bw
        # out["peak_path"] = peak

        all_rows.append(out)

# Concatenate into one DataFrame
results_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

results_df.to_csv("../metadata/CREST-GV_collection_QC.csv", sep='\t', index=False)