"""Expand a seed-track list into the full E1 (track, mode, level, seed) job manifest.

One row per combination - the atomic unit of work processed by
experiments/e1_degradation_benchmark/run_one_degradation.py, one row per
SLURM array task in scripts/slurm/run_e1_degradation.sh.

down_sample level=1.0 is included as the undegraded reference point: thinning
to fraction=1.0 is the identity (bigwig_qc.degradation.thin_signal), so this
re-runs LanceOtron on the un-degraded signal using the same pipeline version
as every degraded point, giving a fair "0% degradation" baseline.
"""
import pandas as pd

from . import config


def build_manifest(seed_tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, track in seed_tracks.iterrows():
        for level in (1.0,) + config.DOWN_SAMPLE_FRACTIONS:
            for seed in config.DEGRADATION_SEEDS:
                rows.append(
                    {
                        "collection_name": track.collection_name,
                        "experiment_name": track.experiment_name,
                        "degradation_mode": "down_sample",
                        "level": level,
                        "seed": seed,
                    }
                )
        for level in config.NOISE_INJECTION_FRACTIONS:
            for seed in config.DEGRADATION_SEEDS:
                rows.append(
                    {
                        "collection_name": track.collection_name,
                        "experiment_name": track.experiment_name,
                        "degradation_mode": "inject_noise",
                        "level": level,
                        "seed": seed,
                    }
                )
    return pd.DataFrame(rows)
