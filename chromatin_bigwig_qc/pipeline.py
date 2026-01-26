from __future__ import annotations

# Standard library imports
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import pyBigWig
import xgboost as xgb


class ChromatinBigwigQC:
    """
    Prediction-only pipeline for chromatin bigWig quality control.
    """

    # Full set of required columns in the peak (BED-like) file
    REQUIRED_PEAK_COLS = [
        "chrom",
        "start",
        "end",
        "overall_peak_score",
        "shape_score",
        "enrichment_score",
        "max_coverage_position",
        "region_id",
        "pvalue_chrom",
        "pvalue_10kb",
        "pvalue_20kb",
        "pvalue_30kb",
        "pvalue_40kb",
        "pvalue_50kb",
        "pvalue_60kb",
        "pvalue_70kb",
        "pvalue_80kb",
        "pvalue_90kb",
        "pvalue_100kb",
    ]

    def __init__(
        self,
        feature_columns: Optional[Sequence[str]] = None,
        random_state: int = 42,
        predict_dtype: np.dtype = np.float32,
    ):
        """
        Initialize the QC pipeline.

        predict_dtype:
            dtype used when converting features to numpy before xgboost prediction.
            Using float32/float64 avoids xgboost's pandas dtype inspection (and warnings).
        """
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
        self.random_state = random_state
        self.predict_dtype = predict_dtype
        self.model: Optional[xgb.XGBRegressor] = None

    @staticmethod
    def _safe_stats_sum(
        bw: pyBigWig.pyBigWig,
        chrom: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> float:
        """Safely compute a sum statistic from a bigWig file; return 0.0 on None/errors."""
        try:
            if start is None and end is None:
                result = bw.stats(chrom, type="sum", exact=True)
            else:
                result = bw.stats(chrom, int(start), int(end), type="sum", exact=True)

            if result is None or len(result) == 0 or result[0] is None:
                return 0.0
            return float(result[0])
        except Exception:
            return 0.0

    def summarise_bigwig_peak(self, bw_path: str, peak_call_path: str) -> Dict[str, Union[int, float]]:
        """Summarise bigWig + peak calls in memory and return a dict of QC features."""
        # Open bigWig
        bw = pyBigWig.open(bw_path)
        if bw is None:
            raise FileNotFoundError(f"Could not open bigWig file: {bw_path}")

        # Load peak calls
        peak_calls = pd.read_csv(peak_call_path, sep="\t", header=0, low_memory=False)

        # Strictly validate required columns for the peak file
        missing = set(self.REQUIRED_PEAK_COLS) - set(peak_calls.columns)
        if missing:
            raise ValueError(
                "Peak calls file is missing required columns.\n\n"
                f"Missing columns:\n  {sorted(missing)}\n\n"
                "This QC model is valid ONLY for peak files generated with LanceOtron.\n"
                "Please ensure peaks were called using LanceOtron and that the full\n"
                "set of LanceOtron output columns is present.\n\n"
                "LanceOtron repository:\n"
                "  https://github.com/LHentges/LanceOtron\n"
            )


        # Compute total signal in bigWig over relevant chromosomes only
        total_bigwig_sum = 0.0
        for chrom in peak_calls["chrom"].dropna().unique():
            total_bigwig_sum += self._safe_stats_sum(bw, chrom)

        # Compute signal per peak
        peak_calls = peak_calls.copy()
        peak_calls["bigwig_sum"] = [
            self._safe_stats_sum(bw, row.chrom, row.start, row.end)
            for _, row in peak_calls.iterrows()
        ]
        peak_calls["length"] = peak_calls["end"].astype(int) - peak_calls["start"].astype(int)

        total_safe = total_bigwig_sum if total_bigwig_sum != 0 else 1.0

        summary = {
            "mean_peak_score": peak_calls["overall_peak_score"].mean(),
            "median_peak_length": peak_calls["overall_peak_score"].median(),
            "mean_shape_score": peak_calls["shape_score"].mean(),
            "mean_peak_length": peak_calls["length"].mean(),
            "mean_enrichment_score": peak_calls["enrichment_score"].mean(),
            "median_enrichment_score": peak_calls["enrichment_score"].median(),
            "lower_quartile_enrichment_score": peak_calls["enrichment_score"].quantile(0.25),
            "upper_quartile_enrichment_score": peak_calls["enrichment_score"].quantile(0.75),
            "percentage_coverage": peak_calls["bigwig_sum"].sum() / total_safe,
            "percentage_coverage_0_5": peak_calls.loc[
                peak_calls["overall_peak_score"] >= 0.5, "bigwig_sum"
            ].sum() / total_safe,
            "percentage_coverage_0_95": peak_calls.loc[
                peak_calls["overall_peak_score"] >= 0.95, "bigwig_sum"
            ].sum() / total_safe,
            "n_peaks": int(len(peak_calls)),
            "n_peaks_0_5": int((peak_calls["overall_peak_score"] >= 0.5).sum()),
            "n_peaks_0_95": int((peak_calls["overall_peak_score"] >= 0.95).sum()),
            "total_bigwig_sum": total_bigwig_sum,
        }

        bw.close()
        return summary

    def load_model(self, model_path: str) -> xgb.XGBRegressor:
        """Load a pre-trained XGBoost model from disk."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        return self.model

    def _prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert summary DataFrame into numeric feature matrix X.
        Ensures purely numeric values and fills missing values.
        """
        df = df.copy()

        if self.feature_columns is not None:
            missing = [c for c in self.feature_columns if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required feature columns: {missing}")
            X = df[self.feature_columns]
        else:
            # infer numeric columns, excluding identifiers
            exclude = {"collection_name", "experiment_name"}
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.feature_columns = [c for c in numeric_cols if c not in exclude]
            X = df[self.feature_columns]

        # Convert to numeric, coerce bad values to NaN, fill with 0
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Force dense float dtype early
        X = X.astype(self.predict_dtype, copy=False)

        return X

    def predict_from_pairs(self, pairs: Sequence[Tuple[str, str]]) -> pd.DataFrame:
        """Run prediction directly from (bigWig, peak-call) file pairs."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call `load_model()` first.")

        summaries = []
        for idx, (bw_path, peak_path) in enumerate(pairs):
            summary = self.summarise_bigwig_peak(bw_path, peak_path)
            summary["collection_name"] = f"collection_{idx}"
            summary["experiment_name"] = f"experiment_{idx}"
            summaries.append(summary)

        df = pd.DataFrame(summaries)

        X = self._prepare_X(df)

        # Avoid pandas dtype inspection warnings inside xgboost:
        # pass numpy array (dense) instead of DataFrame.
        X_np = X.to_numpy(dtype=self.predict_dtype, copy=False)

        # If the model was trained with early stopping, best_iteration will exist.
        # If not, you may want to drop iteration_range and just call predict(X_np).
        predictions = self.model.predict(X_np, iteration_range=(0, self.model.best_iteration + 1))

        df["prediction"] = predictions

        # Map continuous prediction to QC labels
        conditions = [
            df["prediction"] > 6471.38,
            (df["prediction"] <= 6471.38) & (df["prediction"] >= 4754.02),
            df["prediction"] < 4754.02,
        ]
        labels = ["Good", "Decent", "Bad"]
        df["qc_label"] = np.select(conditions, labels)

        # Return only the relevant columns
        return df[["prediction", "qc_label"]]
