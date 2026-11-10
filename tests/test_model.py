import numpy as np
import pandas as pd
import xgboost as xgb

from bigwig_qc.model import clean_hyperparams, load_model, save_model


def _synthetic_data(n=60, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = X.sum(axis=1) + rng.normal(scale=0.1, size=n)
    return X, y


def test_loaded_model_get_params_includes_stale_feature_types(tmp_path):
    X, y = _synthetic_data(n_features=5)
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2)
    model.fit(X, y)

    path = tmp_path / "model.json"
    save_model(model, path)
    loaded = load_model(path)

    params = loaded.get_params()
    assert params.get("feature_types") is not None
    assert len(params["feature_types"]) == 5


def test_clean_hyperparams_strips_feature_types(tmp_path):
    X, y = _synthetic_data(n_features=5)
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2)
    model.fit(X, y)
    path = tmp_path / "model.json"
    save_model(model, path)

    params = clean_hyperparams(load_model(path).get_params())
    assert "feature_types" not in params
    # the real hyperparameters survive the cleaning
    assert params["n_estimators"] == 5
    assert params["max_depth"] == 2


def test_cleaned_hyperparams_allow_refitting_on_a_different_column_count(tmp_path):
    # Reproduces the exact failure this fixes: a model trained/saved on 5
    # columns, then its hyperparameters reused to fit a *new* XGBRegressor on
    # a 3-column dataset (e.g. a feature-ablation experiment) - this raised
    # "Length of feature_type must be equal to number of columns" before
    # clean_hyperparams existed.
    X, y = _synthetic_data(n_features=5)
    original = xgb.XGBRegressor(n_estimators=5, max_depth=2)
    original.fit(X, y)
    path = tmp_path / "model.json"
    save_model(original, path)

    params = clean_hyperparams(load_model(path).get_params())

    X_fewer_columns, y_fewer_columns = _synthetic_data(n_features=3, seed=1)
    refit = xgb.XGBRegressor(**params)
    refit.fit(X_fewer_columns, y_fewer_columns)  # must not raise
    refit.predict(X_fewer_columns)
