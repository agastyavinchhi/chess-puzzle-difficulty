"""Fit baseline, Ridge, and LightGBM models on data/features.parquet.

Outputs to artifacts/: metrics.json, predictions.parquet, importance.csv, model.txt
"""

import itertools
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prepare import CATEGORICAL, FEATURES, FEATURES_PATH

ARTIFACTS = Path(__file__).parent / "artifacts"
SEED = 42

LGB_GRID = {
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.03, 0.1],
    "min_child_samples": [20, 100],
}


def scores(y, pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
    }


def fit_ridge(X, y):
    numeric = [f for f in FEATURES if f not in CATEGORICAL]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 3, 12))).fit(X, y)


def fit_lgb(X, y):
    """Small grid search on a validation slice, then refit on all training data."""
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=SEED)
    best = None
    for values in itertools.product(*LGB_GRID.values()):
        params = dict(zip(LGB_GRID, values))
        model = lgb.LGBMRegressor(n_estimators=3000, random_state=SEED, verbose=-1, **params)
        model.fit(
            X_tr,
            y_tr,
            eval_X=X_val,
            eval_y=y_val,
            categorical_feature=CATEGORICAL,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        rmse = scores(y_val, model.predict(X_val))["RMSE"]
        if best is None or rmse < best[0]:
            best = (rmse, params, model.best_iteration_)
    rmse, params, n_trees = best
    print(f"LightGBM best on validation: RMSE {rmse:.1f}, {params}, {n_trees} trees")
    model = lgb.LGBMRegressor(n_estimators=n_trees, random_state=SEED, verbose=-1, **params)
    return model.fit(X, y, categorical_feature=CATEGORICAL), {**params, "n_estimators": n_trees}


def main():
    df = pd.read_parquet(FEATURES_PATH)
    train, test = train_test_split(df, test_size=0.2, random_state=SEED)
    X_train, y_train = train[FEATURES], train["Rating"]
    X_test, y_test = test[FEATURES], test["Rating"]

    ridge = fit_ridge(X_train, y_train)
    gbm, gbm_params = fit_lgb(X_train, y_train)

    preds = {
        "Baseline (mean)": np.full(len(test), y_train.mean()),
        "Ridge": ridge.predict(X_test),
        "LightGBM": gbm.predict(X_test),
    }
    metrics = {name: scores(y_test, p) for name, p in preds.items()}
    for name, m in metrics.items():
        print(f"{name:16s} RMSE {m['RMSE']:7.1f}  MAE {m['MAE']:7.1f}  R2 {m['R2']:5.2f}")

    ARTIFACTS.mkdir(exist_ok=True)
    meta = {"n_train": len(train), "n_test": len(test), "lightgbm_params": gbm_params}
    (ARTIFACTS / "metrics.json").write_text(json.dumps({"models": metrics, **meta}, indent=2))

    pd.DataFrame(
        {
            "id": test["PuzzleId"].values,
            "FEN": test["FEN"].values,
            "moves": test["Moves"].values,
            "actual": y_test.values,
            "predicted": preds["LightGBM"],
            "predicted_ridge": preds["Ridge"],
        }
    ).to_parquet(ARTIFACTS / "predictions.parquet", index=False)

    shap_values = shap.TreeExplainer(gbm).shap_values(X_test)
    pd.DataFrame(
        {"feature": FEATURES, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False).to_csv(
        ARTIFACTS / "importance.csv", index=False
    )

    gbm.booster_.save_model(str(ARTIFACTS / "model.txt"))
    print(f"Saved artifacts to {ARTIFACTS}")


if __name__ == "__main__":
    sys.exit(main())
