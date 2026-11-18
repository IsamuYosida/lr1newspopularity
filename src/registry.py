"""Реестр моделей MLflow: регистрация кандидата и продвижение в champion.

    python -m src.registry register     # версия из последнего запуска -> challenger
    python -m src.registry promote      # challenger лучше champion -> меняем алиас

Алиас champion указывает на версию, которую использует пакетное предсказание;
challenger — на кандидата, который ещё не прошёл сравнение.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

RUN_ID_FILE = Path("reports/run_id.txt")
MODEL_URI_FILE = Path("reports/model_uri.txt")


def _load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _client():
    from mlflow import MlflowClient
    return MlflowClient()


def _metric(client, version, name: str) -> float | None:
    run = client.get_run(version.run_id)
    return run.data.metrics.get(name)


def register(model_name: str) -> str:
    """Регистрирует последнюю модель и ставит алиас challenger."""
    import mlflow

    if MODEL_URI_FILE.exists():
        model_uri = MODEL_URI_FILE.read_text(encoding="utf-8").strip()
    else:
        run_id = RUN_ID_FILE.read_text(encoding="utf-8").strip()
        model_uri = f"runs:/{run_id}/model"

    version = mlflow.register_model(model_uri, model_name)
    _client().set_registered_model_alias(model_name, "challenger", version.version)
    print(f"registry   зарегистрирована версия {version.version}, алиас challenger")
    return version.version


def promote(model_name: str, metric: str, greater_is_better: bool = True) -> bool:
    """Продвигает challenger в champion, если его метрика лучше."""
    client = _client()
    challenger = client.get_model_version_by_alias(model_name, "challenger")
    new_value = _metric(client, challenger, metric)

    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
    except Exception:
        champion = None

    if champion is None:
        client.set_registered_model_alias(model_name, "champion", challenger.version)
        print(f"registry   champion не был задан: версия {challenger.version} становится champion")
        return True

    old_value = _metric(client, champion, metric)
    better = new_value > old_value if greater_is_better else new_value < old_value
    action = "продвигаем" if better else "оставляем прежнего"
    print(
        f"registry   {metric}: challenger v{challenger.version} = {new_value:.4f}, "
        f"champion v{champion.version} = {old_value:.4f} -> {action}"
    )
    if better:
        client.set_registered_model_alias(model_name, "champion", challenger.version)
    return better


def main() -> None:
    params = _load_params()["mlflow"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["register", "promote"])
    parser.add_argument("--model-name", default=params["model_name"])
    parser.add_argument("--metric", default=params["primary_metric"])
    args = parser.parse_args()

    if args.action == "register":
        register(args.model_name)
    else:
        promote(args.model_name, args.metric)


if __name__ == "__main__":
    main()
