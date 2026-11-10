"""SHAP-based explanation of the trained quality-score model."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def compute_shap_values(model: xgb.XGBRegressor, X: pd.DataFrame, y: pd.Series = None):
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X, y)


def save_summary_plot(shap_values, X: pd.DataFrame, output_path_stub) -> None:
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(f"{output_path_stub}.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{output_path_stub}.png", format="png", bbox_inches="tight", dpi=300)
    plt.close()


def compute_shap_interaction_values(model: xgb.XGBRegressor, X: pd.DataFrame):
    """Pairwise SHAP interaction values (Lundberg et al. 2018) - an (n, p, p)
    array; interaction_values[:, i, j] is feature i and j's joint contribution
    beyond their individual (main) effects, for every sample.
    """
    explainer = shap.TreeExplainer(model)
    return explainer.shap_interaction_values(X)


def save_dependence_plot(
    shap_values, X: pd.DataFrame, feature: str, output_path_stub, interaction_index="auto"
) -> None:
    """SHAP dependence plot for one feature, coloured by its strongest
    interacting feature (interaction_index="auto", SHAP's default) unless
    overridden.
    """
    shap.dependence_plot(feature, shap_values, X, interaction_index=interaction_index, show=False)
    plt.savefig(f"{output_path_stub}.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{output_path_stub}.png", format="png", bbox_inches="tight", dpi=300)
    plt.close()


def top_interacting_pair(shap_interaction_values, feature_columns) -> tuple:
    """Return the (feature_a, feature_b) pair with the largest mean absolute
    off-diagonal interaction magnitude, across all samples.
    """
    mean_abs = np.abs(shap_interaction_values).mean(axis=0)
    off_diagonal = mean_abs.copy()
    np.fill_diagonal(off_diagonal, 0)
    i, j = np.unravel_index(off_diagonal.argmax(), off_diagonal.shape)
    return feature_columns[i], feature_columns[j]
