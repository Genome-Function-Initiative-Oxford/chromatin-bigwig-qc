"""Variance-decomposition helpers for E7's mixed-effects analysis."""


def intraclass_correlation(group_variance: float, residual_variance: float) -> float:
    """Fraction of total variance attributable to the grouping factor (e.g.
    collection identity) rather than within-group (residual) variation.
    """
    return group_variance / (group_variance + residual_variance)
