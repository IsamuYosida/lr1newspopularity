# lr1newspopularity

**Вариант 7** — Online News Popularity (UCI).  
Задача: регрессия — предсказать число репостов статьи (`shares`).  
Разбиение — по времени (`timedelta`): тест — самые новые статьи.  
`timedelta` — часы от первой статьи; большее значение == более старая статья.  
Поэтому `test = timedelta <= cutoff` (новые), `train = timedelta > cutoff` (старые).  
Это соответствует варианту 7: «старые статьи — обучение, новые — поток».

## Воспроизведение

Требуется: Python ≥ 3.10, git, доступ к GitHub и DagsHub.

```bash
git clone https://github.com/IsamuYosida/lr1newspopularity.git
cd lr1newspopularity

# 1. Окружение строго по lock-файлу
pip install uv
uv export --frozen --no-hashes -o requirements.lock.txt
pip install -r requirements.lock.txt

# 2. Доступ к DVC remote (DagsHub S3) — токен вводится локально
dvc remote modify --local origin access_key_id     <DAGSHUB_TOKEN>
dvc remote modify --local origin secret_access_key <DAGSHUB_TOKEN>

# 3. Данные + артефакты и полный прогон пайплайна
dvc pull
dvc repro
cat reports/metrics.json
```

`<DAGSHUB_TOKEN>` — Personal Access Token из DagsHub (Settings → Tokens).
Команды `dvc remote modify --local` пишут в `.dvc/config.local`, который в `.gitignore`
и никогда не попадает в репозиторий.

## Граф пайплайна

```
prepare → featurize → train → evaluate
```

Этапы описаны в `dvc.yaml`. `dvc repro` строит весь путь от исходных данных
до `reports/metrics.json`; повторный запуск ничего не пересчитывает
(«didn't change, skipping»). `dvc dag` печатает ASCII-граф зависимостей.

### Стадии

| Стадия | Вход | Выход | Параметры |
|--------|------|-------|-----------|
| `prepare` | `data/external/OnlineNewsPopularity.csv` | `data/processed/{train,test}.csv` | `prepare.*` |
| `featurize` | `data/processed/*.csv` | `data/features/{X,y}_{train,test}.csv` | — |
| `train` | `data/features/{X,y}_train.csv` | `models/model.joblib` | `train.*` |
| `evaluate` | модель + все `data/features/*.csv` | `reports/metrics.json` | `evaluate.*` |

## Модель

По умолчанию `train.model = hist_gb` (`HistGradientBoostingRegressor`),
обучение идёт на `log1p(shares)`, предсказание возвращается через `expm1`
с обрезкой снизу нулём. Это стабилизирует обучение из-за тяжёлого хвоста `shares`.

```yaml
# params.yaml
train:
  model: hist_gb   # linear_ridge | random_forest | hist_gb
  target_transform: log1p
```

После смены `model` достаточно запустить `dvc repro` — пересчитаются только
`train` и `evaluate` (у `prepare` и `featurize` deps не изменились).

## Итоговые метрики

Замеры на тестовой выборке (самые новые статьи на момент cutoff).

| Версия данных | Строк  | Тест, строк | model_mae | model_rmse | model_r2 |
|---------------|--------|-------------|-----------|------------|----------|
| v1 (тег `data-v1`) | 39644 | 7951 | 2087.53 | 8128.65 | 0.0012 |
| v2 (тег `data-v2`) | 40644 | 8106 | 2081.58 | 8089.39 | 0.0012 |

Baseline (константа = `expm1(mean(log1p(shares_train)))`):

| Версия | baseline_mae | baseline_rmse | baseline_r2 |
|--------|--------------|---------------|-------------|
| v1 | 2216.37 | 8224.00 | -0.0224 |
| v2 | 2215.69 | 8186.54 | -0.0230 |

Модель лучше baseline по MAE/RMSE, но R² близок к нулю — `shares` имеет
тяжёлый хвост и слабо предсказуем по метаданным;
что яыляется ожидаемым поведением набора. Метрики записаны плоским JSON
(DVC читает только скалярные ключи), поэтому их можно сравнивать между
ревизиями через `dvc metrics diff data-v1 main`.

## Сценарий смены версии данных

Версии данных фиксируются тегами Git и метафайлами `.dvc`:

- `data-v1` — исходный набор (39 644 строки), скачанный с UCI.
- `data-v2` — к исходному набору добавлено 1000 строк: берутся 1000 статей
  с наибольшим `timedelta`, и им присваивается `timedelta = max_td + 1 + i`
  (то есть искусственно «удревняются» и остаются в train). Итог — 40 644 строки.
  Цель: продемонстрировать воспроизводимый переход между двумя версиями
  данных и корректный пересчёт метрик, а не моделировать реальный поток.

Переход v2 → v1 → v2:

```bash
# На v1
git checkout data-v1  # код + .dvc-метафайл v1
dvc checkout  # по md5 из *.dvc достаёт CSV v1 из локального кэша
dvc repro # метрики совпадают с v1 до последнего знака
cat reports/metrics.json

# Обратно на v2
git checkout main
dvc checkout
dvc repro
```

Без `dvc checkout` в рабочем каталоге останутся данные одной версии при
коде/метафайле другой — пайплайн отработает и молча даст неверный результат.
Поэтому `git checkout` + `dvc checkout` обязательны.

## Проверки перед коммитом

```bash
pre-commit install # один раз — ставит git-хук
pre-commit run --all-files   # ruff-check, ruff-format, large-files, yaml, private-key
pytest -q  # smoke-тесты из tests/
```

Хуки настроены в `.pre-commit-config.yaml` (ruff v0.11.13, pre-commit-hooks v5.0.0).
Это защита от форматных правок в истории и от случайных секретов/бинарников.

## Структура

```
src/
  __init__.py
  schema.py     # pandera-схема исходных данных
  prepare.py     # валидация + разбиение по времени
  featurize.py  # X/y train/test
  train.py     # выбор модели + обучение на log1p(shares)
  evaluate.py   # MAE/RMSE/R² + baseline + порог min_r2
tests/
  test_smoke.py
notebooks/
data/
  external/OnlineNewsPopularity.csv{,.dvc}
  processed/     # train.csv, test.csv
  features/      # X_train.csv, X_test.csv, y_train.csv, y_test.csv
models/model.joblib
reports/metrics.json

params.yaml     # все гиперпараметры
dvc.yaml       # граф пайплайна
dvc.lock       # фиксация md5 deps/outs
pyproject.toml     # зависимости + ruff/pytest конфиг
uv.lock       # точные версии всех пакетов
.pre-commit-config.yaml # хуки
.gitignore  # данные/модели/reports — под DVC, не в git
.dvc/
  config  # remote origin (endpointurl)
  config.local   # токены (не коммитится)
```

## Что версионируется, а что нет

- **В git**: код (`src/`, `tests/`), `params.yaml`, `dvc.yaml`, `dvc.lock`,
  `*.dvc`-метафайлы, `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, README.
- **В DVC remote (DagsHub S3)**: `data/external/*.csv`, `data/processed/*`,
  `data/features/*`, `models/model.joblib`.
- **Нигде**: `data/processed/`, `data/features/`, `models/`, `reports/` —
  пересоздаются `dvc repro`; `reports/metrics.json` дополнительно коммитится
  в git как точка сравнения между ревизиями.

## Отказоустойчивость и проверки

- `src/schema.py` (pandera) — пайплайн падает при нарушении схемы,
  но не выдает неверные метрики.
- `prepare.py` — ассерт `train.min(timedelta) > test.max(timedelta)`
  (нет временной утечки).
- `featurize.py` — проверка совпадения колонок train/test, отсутствия NaN
  и нечисловых признаков.
- `evaluate.py` — бросает `ValueError`, если `model_r2 < evaluate.min_r2`.
- `tests/test_smoke.py` — контракт структуры и ключей `metrics.json`.
- pre-commit — стиль и запрет секретов/больших файлов.
