import argparse
import pandas as pd
from intervaltree import IntervalTree
from glob import glob
import os


# Load gene defintions
def load_genes():
    """
    Load protein coding gene definitions from RefSeq
    """
    chrom_names_file = "/project/Wellcome_Discovery/shared/00_publicly_available/06_dbSNP/bcftools_rename_file.txt"
    chrom_relabel = {}
    with open(chrom_names_file) as f:
        for line in f.readlines():
            chrom_relabel[line.strip().split("\t")[0]] = line.strip().split("\t")[1]

    genes = pd.read_csv(
        "/project/Wellcome_Discovery/shared/00_publicly_available/14_RefSeq/ncbi_dataset/data/GCF_000001405.40/genomic.gtf",
        skiprows=5, header=None, sep="\t"
    )
    genes = genes[genes[2] == "gene"]
    genes = genes[genes[8].str.contains("protein_coding|miRNA|lnc_RNA")]
    genes = genes.replace({0: chrom_relabel})
    genes = genes[genes[0].isin(["chr{}".format(i) for i in range(1, 23)] + ["chrX", "chrY"])]
    genes[3] = genes[3].astype(int)
    genes[4] = genes[4].astype(int)
    genes["gene_id"] = genes[8].str.split(";", expand=True)[0].str.split(" ", expand=True)[1].str.strip("\"")
    return genes


def get_tree_constant_promoters(genes):
    # https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010295#sec013
    # Housekeeping genes to define a better protein set
    housekeeping_genes = pd.read_excel("../metadata/10_1371_journal_pcbi_1010295/pcbi.1010295.s014.xlsx", sheet_name="GTEx")
    
    padding_window = 100
    genes_tree = {}
    set_chroms = set(genes[0])
    for chrom in set_chroms:
        genes_tree[chrom] = IntervalTree()
        sub_genes = genes[(genes[0] == chrom) & genes.gene_id.isin(housekeeping_genes.Genes)]
        for _, gene in sub_genes.iterrows():
            if gene[6] == "+":
                genes_tree[chrom][gene[3] - padding_window:gene[3] + 1] = "TSS"
            else:
                genes_tree[chrom][gene[4]-1:gene[4] + padding_window] = "TSS"

    return genes_tree


def get_ctcf_tree():
    """
    Use constant CTCF regions from recent publication
    """
    constant_ctcf_peaks = pd.read_csv(
        "/project/Wellcome_Discovery/sriva/Git/CTCF_collection/metadata/intersect.bed",
        sep="\t",
        header=None
    )
    constant_ctcf_peaks = constant_ctcf_peaks[constant_ctcf_peaks[4] >= 0.99]
    print(constant_ctcf_peaks.shape[0])

    ctcf_tree = {}
    set_chroms = set(constant_ctcf_peaks[0])
    for chrom in set_chroms:
        ctcf_tree[chrom] = IntervalTree()
        for _, ctcf_region in constant_ctcf_peaks[constant_ctcf_peaks[0] == chrom].iterrows():
            ctcf_tree[chrom][ctcf_region[1]:ctcf_region[2]] = ctcf_region[4]
    
    return ctcf_tree


def summarise_data(directory_path, output_filepath):
    """
    Summarize a collection of peak calls
    """
    genes = load_genes()
    promoter_tree = get_tree_constant_promoters(genes)
    ctcf_tree = get_ctcf_tree()
    
    print(f"Directory path {directory_path}")
    file_pattern = os.path.join(directory_path, "*_L-tron.bed")
    files = glob(file_pattern)

    summary_data = []
    print(f"N files for {file_pattern}", len(files))
    for file in files:
        collection_name = os.path.dirname(file).split("/")[-1]
        experiment_name = os.path.basename(file).replace("_L-tron.bed", "")
        
        peak_calls = pd.read_csv(file, sep="\t")
        peak_calls["promoter"] = peak_calls.apply(
            lambda row: len(promoter_tree.get(row.chrom, IntervalTree())[row.start:row.end]) > 0, axis=1
        )
        peak_calls["ctcf"] = peak_calls.apply(
            lambda row: len(ctcf_tree.get(row.chrom, IntervalTree())[row.start:row.end]) > 0, axis=1
        )
        
        summary_data.append({
            "experiment_name": experiment_name,
            "collection_name": collection_name,

            "n_promoters": peak_calls[peak_calls.promoter].shape[0],
            "n_promoters_0_5": peak_calls[peak_calls.promoter & (peak_calls.overall_peak_score >= 0.5)].shape[0],
            "n_promoters_0_95": peak_calls[peak_calls.promoter & (peak_calls.overall_peak_score >= 0.95)].shape[0],

            "n_ctcf": peak_calls[peak_calls.ctcf].shape[0],
            "n_ctcf_0_5": peak_calls[peak_calls.ctcf & (peak_calls.overall_peak_score >= 0.5)].shape[0],
            "n_ctcf_0_95": peak_calls[peak_calls.ctcf & (peak_calls.overall_peak_score >= 0.95)].shape[0],
        })
    
    print("Amount of summary data: ", len(summary_data))
    summary_data = pd.DataFrame(summary_data)
    summary_data.to_csv(output_filepath, sep="\t", index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take collection peak calls and summarise constant CTCF and Promoter counts"
    )
    parser.add_argument("--input", required=True, help="Path to directory of peak calls")
    parser.add_argument("--output", required=True, help="tsv with summary counts")
    args = parser.parse_args()

    summarise_data(args.input, args.output)
