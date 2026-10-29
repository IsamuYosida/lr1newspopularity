"""Подготовка данных: проверка схемы и разбиение по времени.

timedelta — часы от первой статьи; большее значение == более старая статья.
Тест — самые новые статьи (меньший timedelta).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.schema import validate_raw


def split_by_time(
    data: pd.DataFrame, test_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит по уникальным timedelta, чтобы точка не попала в обе выборки."""
    times = sorted(data["timedelta"].unique())
    target_test = int(len(data) * test_fraction)
    cutoff = min(
        times,
        key=lambda v: abs(int((data["timedelta"] <= v).sum()) - target_test),
    )
    test = data[data["timedelta"] <= cutoff].copy()
    train = data[data["timedelta"] > cutoff].copy()
    return train, test


def prepare_data(
    input_path: str = "data/external/OnlineNewsPopularity.csv",
    output_dir: str = "data/processed",
    params_path: str = "params.yaml",
) -> None:
    with open(params_path, encoding="utf-8") as fh:
        params = yaml.safe_load(fh)["prepare"]
    test_fraction = float(params["test_fraction"])
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction должен быть между 0 и 1.")

    raw = pd.read_csv(input_path)
    raw.columns = raw.columns.str.strip()
    data = validate_raw(raw)

    train, test = split_by_time(data, test_fraction)

    assert not train.empty and not test.empty
    assert train["timedelta"].min() > test["timedelta"].max()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    cutoff_value = float(train["timedelta"].min())
    print(f"Схема пройдена: {len(data)} строк")
    print(
        f"Train: {len(train)} | Test: {len(test)} "
        f"| cutoff timedelta = {cutoff_value}"
    )


if __name__ == "__main__":
    prepare_data()
