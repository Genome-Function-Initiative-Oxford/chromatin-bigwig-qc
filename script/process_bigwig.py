import argparse
import pyBigWig
import json
import numpy as np
import pandas as pd


def summarise_data(bw_path, peak_call_path, output_json):
    """
    Summarize a bigWig file and save the results as JSON.
    """
    
    bw = pyBigWig.open(bw_path)
    peak_calls = pd.read_csv(peak_call_path, sep="\t")

    total_bigwig_sum = sum([bw.stats(chrom, type="sum", exact=True)[0] for chrom in set(peak_calls.chrom)])
    row_sums = []
    for _, row in peak_calls.iterrows():
        row_sums.append(bw.stats(row.chrom, row.start, row.end, type="sum", exact=True)[0])
    peak_calls["bigwig_sum"] = row_sums
    peak_calls["length"] = peak_calls.end - peak_calls.start
    
    summary = {
        "mean_peak_score": peak_calls.overall_peak_score.mean(),
        "median_peak_length": peak_calls.overall_peak_score.median(),
        
        "mean_shape_score": peak_calls.shape_score.mean(),
        "mean_peak_length": peak_calls.length.mean(),
        
        "mean_enrichment_score": peak_calls.enrichment_score.mean(),
        "median_enrichment_score": peak_calls.enrichment_score.median(),
        "lower_quartile_enrichment_score": peak_calls.enrichment_score.quantile(0.25),
        "upper_quartile_enrichment_score": peak_calls.enrichment_score.quantile(0.75),

        "percentage_coverage": peak_calls.bigwig_sum.sum() / total_bigwig_sum,
        "percentage_coverage_0_5": peak_calls[peak_calls.overall_peak_score >= 0.5].bigwig_sum.sum() / total_bigwig_sum,
        "percentage_coverage_0_95": peak_calls[peak_calls.overall_peak_score >= 0.95].bigwig_sum.sum() / total_bigwig_sum,

        "n_peaks": peak_calls.shape[0],
        "n_peaks_0_5": peak_calls[peak_calls.overall_peak_score >= 0.5].shape[0],
        "n_peaks_0_95": peak_calls[peak_calls.overall_peak_score >= 0.95].shape[0],

        "total_bigwig_sum": round(total_bigwig_sum / 1000),
    }


    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Summary saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize a bigWig file and output as JSON. Measures quality"
    )
    parser.add_argument("--input-bigwig", required=True, help="Path to the input bigWig file.")
    parser.add_argument("--input-peak-calls", required=True, help="Path to the input peak calls file.")
    parser.add_argument("--output", required=True, help="Path to the output JSON file.")
    args = parser.parse_args()

    summarise_data(args.input_bigwig, args.input_peak_calls, args.output)
