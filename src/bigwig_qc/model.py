"""Train and tune the XGBoost quality-score regressor.

Target: n_ctcf + n_promoters (constant-site recovery count).

Reproduces the three-stage grid search from the CIBCB pipeline:
  1) n_estimators / learning_rate
  2) max_depth / min_child_weight / gamma / subsample / colsample_bytree
  3) reg_alpha / reg_lambda
"""
from dataclasses import dataclass, field

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

RANDOM_STATE = 42

STAGE_1_GRID = {"n_estimators": [10, 25, 50, 100, 150, 200], "learning_rate": [0.05, 0.1, 0.2]}
STAGE_2_GRID = {
    "max_depth": [2, 3, 4],
    "min_child_weight": [2, 3, 4],
    "gamma": [0, 0.05, 0.1],
    "subsample": [0.6, 0.8, 0.9],
    "colsample_bytree": [0.8, 0.9, 1.0],
}
STAGE_3_GRID = {"reg_alpha": [0, 0.05, 0.1], "reg_lambda": [0.05, 0.1, 0.5, 1]}


@dataclass
class TrainResult:
    model: xgb.XGBRegressor
    best_params: dict = field(default_factory=dict)
    test_r2: float = 0.0
    test_mse: float = 0.0


def regression_target(data: pd.DataFrame) -> pd.Series:
    return data.n_ctcf + data.n_promoters


def _grid_search(fixed_params: dict, grid: dict, X_train, y_train) -> dict:
    search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, **fixed_params),
        param_grid=grid,
        cv=5,
        scoring="neg_mean_squared_error",
    )
    search.fit(X_train, y_train)
    return search.best_params_


def train(
    data: pd.DataFrame,
    feature_columns: list,
    test_size: float = 0.2,
    grids: tuple = (STAGE_1_GRID, STAGE_2_GRID, STAGE_3_GRID),
) -> TrainResult:
    X = data[feature_columns]
    y = regression_target(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

    params = {}
    for grid in grids:
        params.update(_grid_search(params, grid, X_train, y_train))

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, **params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return TrainResult(
        model=model,
        best_params=params,
        test_r2=r2_score(y_test, y_pred),
        test_mse=mean_squared_error(y_test, y_pred),
    )


def save_model(model: xgb.XGBRegressor, path) -> None:
    model.save_model(str(path))


def load_model(path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


# Keys that reflect the ORIGINAL training data (e.g. a per-column
# feature_types list sized to that training set's column count), not true
# hyperparameters. A loaded model's get_params() includes these, and passing
# them straight into a new XGBRegressor(**params) breaks the moment that new
# model is fit on a different number of columns (e.g. a feature-ablation
# experiment) with a column-count mismatch error from XGBoost's C++ core.
DATA_DEPENDENT_PARAM_KEYS = ("feature_types",)


def clean_hyperparams(params: dict) -> dict:
    """Strip DATA_DEPENDENT_PARAM_KEYS from a (loaded) model's get_params() output.

    Always apply this before reusing a loaded model's hyperparameters to
    construct a *new* XGBRegressor for fitting on a different feature set.
    """
    return {k: v for k, v in params.items() if k not in DATA_DEPENDENT_PARAM_KEYS}
