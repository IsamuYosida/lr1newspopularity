"""Признаки: убираем url и неинформативные колонки, оставляем числовые."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "shares"
DROP = {"url"}


def featurize(
    processed_dir: str = "data/processed",
    features_dir: str = "data/features",
) -> None:
    proc = Path(processed_dir)
    feat = Path(features_dir)
    feat.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(proc / "train.csv")
    test = pd.read_csv(proc / "test.csv")

    if TARGET not in train.columns or TARGET not in test.columns:
        raise ValueError("Целевая колонка shares отсутствует.")

    feature_cols = [c for c in train.columns if c not in DROP and c != TARGET]
    test_cols = [c for c in test.columns if c not in DROP and c != TARGET]
    if feature_cols != test_cols:
        raise ValueError("Наборы признаков train и test не совпадают.")

    x_train, x_test = train[feature_cols], test[feature_cols]
    y_train, y_test = train[TARGET], test[TARGET]

    for name, frame in (("X_train", x_train), ("X_test", x_test)):
        non_num = [
            c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])
        ]
        if non_num:
            raise ValueError(f"{name}: нечисловые признаки {non_num}")
        if frame.isna().any().any():
            raise ValueError(f"{name}: пропуски недопустимы.")

    x_train.to_csv(feat / "X_train.csv", index=False)
    x_test.to_csv(feat / "X_test.csv", index=False)
    y_train.to_frame().to_csv(feat / "y_train.csv", index=False)
    y_test.to_frame().to_csv(feat / "y_test.csv", index=False)

    print(f"Признаков: {len(feature_cols)}")
    print(f"X_train {x_train.shape} | X_test {x_test.shape}")


if __name__ == "__main__":
    featurize()
