"""Оценка модели: метрики, baseline и проверка порога."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def evaluate(
    features_dir: str = "data/features",
    model_path: str   = "models/model.joblib",
    report_path: str  = "reports/metrics.json",
    params_path: str  = "params.yaml",
) -> None:
    with open(params_path, encoding="utf-8") as fh:
        params = yaml.safe_load(fh)
    min_r2 = float(params.get("evaluate", {}).get("min_r2", 0.0))

    feat = Path(features_dir)
    X_train = pd.read_csv(feat / "X_train.csv")
    y_train = pd.read_csv(feat / "y_train.csv").squeeze("columns")
    X_test  = pd.read_csv(feat / "X_test.csv")
    y_test  = pd.read_csv(feat / "y_test.csv").squeeze("columns")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Признаки train и test не совпадают.")

    model = joblib.load(model_path)
    pred = np.maximum(0, np.expm1(model.predict(X_test)))

    # Baseline в том же пространстве, что и предсказание модели:
    # среднее log1p(shares) на train, возвращённое через expm1.
    baseline_value = float(np.expm1(np.log1p(y_train).mean()))
    baseline = np.full(len(y_test), baseline_value)

    m = _metrics(y_test, pred)
    b = _metrics(y_test, baseline)

    # ПЛОСКИЙ JSON: DVC читает только плоские ключи и числа.
    results = {
        "test_size":            int(len(y_test)),
        "n_features":           int(X_test.shape[1]),
        "baseline_mean_shares": baseline_value,
        "model_mae":   m["mae"],
        "model_rmse":  m["rmse"],
        "model_r2":    m["r2"],
        "baseline_mae":  b["mae"],
        "baseline_rmse": b["rmse"],
        "baseline_r2":   b["r2"],
    }

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Модель:")
    print(f"  MAE: {m['mae']:.4f}")
    print(f"  RMSE: {m['rmse']:.4f}")
    print(f"  R2: {m['r2']:.4f}")
    print("Baseline (среднее log1p(shares) на train):")
    print(f"  MAE: {b['mae']:.4f}")
    print(f"  RMSE: {b['rmse']:.4f}")
    print(f"  R2: {b['r2']:.4f}")
    print(f"Отчёт: {report}")

    if m["r2"] < min_r2:
        raise ValueError(f"R2 = {m['r2']:.4f} ниже порога {min_r2}")


if __name__ == "__main__":
    evaluate()
