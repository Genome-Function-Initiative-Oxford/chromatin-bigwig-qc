"""Select representative seed tracks for the E1 controlled-degradation benchmark.

Ranks tracks by their already-measured quality (constant promoter + CTCF
recovery - the same quantity bigwig_qc.model is trained to predict) and takes
the top few per collection, so the seed set spans every collection rather than
being dominated by whichever one scores highest overall.
"""
import math

import pandas as pd

from .model import regression_target


def rank_tracks(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["recovery_score"] = regression_target(data)
    return data.sort_values("recovery_score", ascending=False)


def select_seed_tracks(data: pd.DataFrame, n_tracks: int) -> pd.DataFrame:
    ranked = rank_tracks(data)
    n_collections = ranked.collection_name.nunique()
    per_collection_cap = math.ceil(n_tracks / n_collections)

    top_per_collection = ranked.groupby("collection_name", group_keys=False).head(per_collection_cap)
    return rank_tracks(top_per_collection).head(n_tracks)
