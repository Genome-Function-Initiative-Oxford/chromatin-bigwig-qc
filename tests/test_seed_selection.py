import pandas as pd

from bigwig_qc.seed_selection import select_seed_tracks


def _data():
    rows = []
    for collection in ("A", "B"):
        for i in range(5):
            rows.append(
                {
                    "collection_name": collection,
                    "experiment_name": f"{collection}{i}",
                    "n_promoters": i * 10,
                    "n_ctcf": i * 5,
                }
            )
    return pd.DataFrame(rows)


def test_select_seed_tracks_spans_all_collections():
    selected = select_seed_tracks(_data(), n_tracks=4)
    assert set(selected.collection_name) == {"A", "B"}


def test_select_seed_tracks_picks_highest_recovery_first():
    selected = select_seed_tracks(_data(), n_tracks=4)
    assert {"A4", "B4"} <= set(selected.experiment_name)


def test_select_seed_tracks_respects_n_tracks_cap():
    selected = select_seed_tracks(_data(), n_tracks=4)
    assert selected.shape[0] == 4
