import pandas as pd

from bigwig_qc import config
from bigwig_qc.manifest import build_manifest


def _seed_tracks():
    return pd.DataFrame(
        [
            {"collection_name": "A", "experiment_name": "a1"},
            {"collection_name": "B", "experiment_name": "b1"},
        ]
    )


def test_build_manifest_row_count_matches_expected_combinations():
    manifest = build_manifest(_seed_tracks())

    n_down_sample_levels = 1 + len(config.DOWN_SAMPLE_FRACTIONS)  # +1 for the level=1.0 baseline
    n_noise_levels = len(config.NOISE_INJECTION_FRACTIONS)
    expected_rows_per_track = (n_down_sample_levels + n_noise_levels) * len(config.DEGRADATION_SEEDS)

    assert manifest.shape[0] == expected_rows_per_track * 2


def test_build_manifest_includes_down_sample_identity_baseline():
    manifest = build_manifest(_seed_tracks())
    baseline = manifest[
        (manifest.experiment_name == "a1")
        & (manifest.degradation_mode == "down_sample")
        & (manifest.level == 1.0)
    ]
    assert baseline.shape[0] == len(config.DEGRADATION_SEEDS)


def test_build_manifest_covers_both_modes_for_every_track():
    manifest = build_manifest(_seed_tracks())
    for experiment_name in ("a1", "b1"):
        modes = set(manifest[manifest.experiment_name == experiment_name]["degradation_mode"])
        assert modes == {"down_sample", "inject_noise"}
