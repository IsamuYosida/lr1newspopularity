"""Smoke-тесты: модули на месте, метрики содержат ключи."""
import json
from pathlib import Path


def test_src_modules_exist():
    for mod in ("prepare", "featurize", "train", "evaluate", "schema"):
        assert Path(f"src/{mod}.py").exists()


def test_metrics_file_has_keys():
    p = Path("reports/metrics.json")
    if not p.exists():
        return  # до первого dvc repro
    m = json.loads(p.read_text(encoding="utf-8"))
    expected = {
        "model_mae", "model_rmse", "model_r2",
        "baseline_mae", "baseline_rmse", "baseline_r2",
    }
    assert expected <= set(m)
