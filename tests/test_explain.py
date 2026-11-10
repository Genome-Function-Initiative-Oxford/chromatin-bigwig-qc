import numpy as np

from bigwig_qc.explain import top_interacting_pair


def test_top_interacting_pair_finds_largest_off_diagonal_magnitude():
    # 3 features, 2 samples. Interaction (0, 2) has the largest magnitude
    # (5.0), diagonal (main-effect) entries are deliberately huge to confirm
    # they're excluded from the off-diagonal search.
    interaction_values = np.array(
        [
            [[100.0, 1.0, 5.0], [1.0, 100.0, 0.5], [5.0, 0.5, 100.0]],
            [[90.0, -1.0, -5.0], [-1.0, 90.0, 0.2], [-5.0, 0.2, 90.0]],
        ]
    )
    feature_columns = ["feature_a", "feature_b", "feature_c"]
    pair = top_interacting_pair(interaction_values, feature_columns)
    assert set(pair) == {"feature_a", "feature_c"}


def test_top_interacting_pair_ignores_diagonal_even_when_it_dominates():
    interaction_values = np.array([[[1000.0, 0.1], [0.1, 1000.0]]])
    feature_columns = ["feature_a", "feature_b"]
    pair = top_interacting_pair(interaction_values, feature_columns)
    assert set(pair) == {"feature_a", "feature_b"}
