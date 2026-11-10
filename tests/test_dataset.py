import json

import pandas as pd

from bigwig_qc import config
from bigwig_qc.dataset import build_model_dataset


def _write_feature_json(path, collection_name, experiment_name, n_peaks):
    (path / f"{collection_name}.{experiment_name}.json").write_text(json.dumps({"n_peaks": n_peaks}))


def _write_targets_tsv(path, collection_name, rows):
    pd.DataFrame(rows).to_csv(path / f"{collection_name}.tsv", sep="\t", index=False)


def test_build_model_dataset_excludes_collections_not_in_config(tmp_path, monkeypatch):
    summary_dir = tmp_path / "features"
    targets_dir = tmp_path / "targets"
    summary_dir.mkdir()
    targets_dir.mkdir()

    monkeypatch.setattr(config, "COLLECTION_NAMES", ["kept_collection"])

    for collection_name in ("kept_collection", "excluded_collection"):
        _write_feature_json(summary_dir, collection_name, "e1", n_peaks=100)
        _write_targets_tsv(
            targets_dir,
            collection_name,
            [{"experiment_name": "e1", "collection_name": collection_name, "n_promoters": 1, "n_ctcf": 1}],
        )

    data = build_model_dataset(summary_dir, targets_dir)

    assert set(data.collection_name) == {"kept_collection"}
    assert data.shape[0] == 1
