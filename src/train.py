"""Обучение модели. Модель выбирается в params.yaml → train.model."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(train_params: dict):
    name = train_params["model"]
    if name == "linear_ridge":
        cfg = train_params.get("linear_ridge", {})
        steps = [StandardScaler()]
        k = int(cfg.get("select_k", 0))
        if k > 0:
            steps.append(SelectKBest(f_regression, k=k))
        steps.append(Ridge(alpha=float(cfg.get("alpha", 10.0))))
        return make_pipeline(*steps)
    if name == "random_forest":
        cfg = train_params.get("random_forest", {})
        return RandomForestRegressor(
            n_estimators=int(cfg.get("n_estimators", 300)),
            max_depth=int(cfg.get("max_depth", 20)),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 3)),
            random_state=int(cfg.get("random_state", 42)),
            n_jobs=int(cfg.get("n_jobs", -1)),
        )
    if name == "hist_gb":
        cfg = train_params.get("hist_gb", {})
        return HistGradientBoostingRegressor(
            max_iter=int(cfg.get("max_iter", 400)),
            max_depth=int(cfg.get("max_depth", 6)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            random_state=int(cfg.get("random_state", 42)),
        )
    raise ValueError(f"Неизвестная модель: {name!r}")


def train(
    features_dir: str = "data/features",
    model_path: str = "models/model.joblib",
    params_path: str = "params.yaml",
) -> None:
    with open(params_path, encoding="utf-8") as fh:
        params = yaml.safe_load(fh)["train"]

    feat = Path(features_dir)
    X_train = pd.read_csv(feat / "X_train.csv")
    y_train = pd.read_csv(feat / "y_train.csv").squeeze("columns")

    if len(X_train) != len(y_train):
        raise ValueError("Размеры X_train и y_train не совпадают.")
    if X_train.isna().any().any() or y_train.isna().any():
        raise ValueError("Пропуски в данных недопустимы.")
    if (y_train < 0).any():
        raise ValueError("shares не может быть отрицательным.")

    model = make_model(params)
    # log1p стабилизирует тяжёлый хвост shares
    model.fit(X_train, np.log1p(y_train))

    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)

    print(f"Модель: {params['model']}")
    print(f"Обучено на {len(X_train)} статьях, {X_train.shape[1]} признаках.")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    train()
