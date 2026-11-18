"""Схема данных Online News Popularity (pandera >=0.20)."""

from __future__ import annotations

import pandas as pd

# pandera >=0.20: старый корневой API удалён, используем pandera.pandas.
from pandera.pandas import Check, Column, DataFrameSchema

RAW_SCHEMA = DataFrameSchema(
    {
        "timedelta": Column(float, Check.ge(0)),
        "shares": Column(int, Check.ge(0)),
    },
    strict=False,
    coerce=True,
)


def validate_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Проверяет обязательные колонки и числовой тип остальных."""
    validated = RAW_SCHEMA.validate(frame, lazy=True)
    non_numeric = [
        c
        for c in validated.columns
        if c not in {"url"} and not pd.api.types.is_numeric_dtype(validated[c])
    ]
    if non_numeric:
        raise ValueError(f"Нечисловые колонки: {non_numeric}")
    return validated
