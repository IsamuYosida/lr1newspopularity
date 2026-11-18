"""Работа с MLflow. Трекинг включается только при заданном MLFLOW_TRACKING_URI.

В ЛР1 переменная не задана и этапы работают без трекинга; начиная с ЛР2 она
указывает на сервер MLflow, встроенный в репозиторий DagsHub.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

RUN_ID_FILE = Path("reports/run_id.txt")
MODEL_URI_FILE = Path("reports/model_uri.txt")


def enabled() -> bool:
    """Возвращает True, если MLflow трекинг настроен."""
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def start_run(experiment: str, params: dict[str, Any]):
    """Открывает запуск, записывает параметры и сохраняет его идентификатор."""
    if not enabled():
        return None
    import mlflow

    mlflow.set_experiment(experiment)
    run = mlflow.start_run()
    # Логируем только скалярные значения; вложенные словари пропускаем
    flat_params = {k: v for k, v in params.items() if not isinstance(v, dict)}
    mlflow.log_params(flat_params)
    RUN_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUN_ID_FILE.write_text(run.info.run_id, encoding="utf-8")
    return run


def log_metrics(metrics: dict[str, float]) -> None:
    """Дописывает метрики в запуск, открытый на этапе train."""
    if not enabled() or not RUN_ID_FILE.exists():
        return
    import mlflow

    run_id = RUN_ID_FILE.read_text(encoding="utf-8").strip()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)


def log_model(model, features: list[str], example) -> None:
    """Сохраняет модель вместе с сигнатурой входа и запоминает её адрес.

    Адрес модели нужен реестру. Запись в файл избавляет от различий
    между версиями MLflow-сервера.
    """
    if not enabled():
        return
    import cloudpickle
    import mlflow
    import pandas
    import sklearn
    from mlflow.models import infer_signature

    signature = infer_signature(example, model.predict(example))
    # Окружение модели задаётся явно — по тому же принципу, что и lock-файл
    # проекта: тот, кто загрузит модель, получит те же версии библиотек.
    requirements = [
        f"scikit-learn=={sklearn.__version__}",
        f"pandas=={pandas.__version__}",
        f"cloudpickle=={cloudpickle.__version__}",
        f"numpy",
    ]
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=example.head(3),
        # cloudpickle работает одинаково на всех версиях сервера и не требует
        # явного перечисления доверенных типов, в отличие от skops.
        serialization_format="cloudpickle",
        pip_requirements=requirements,
    )
    MODEL_URI_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_URI_FILE.write_text(info.model_uri, encoding="utf-8")
    print(f"Модель сохранена в MLflow: {info.model_uri}")
