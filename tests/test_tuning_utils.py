"""Tests for tuning result selection and CLI application."""
from __future__ import annotations

import json

import pytest

from app.data.schema import initialize_database
from app.utils.config import DataPaths, get_config
from app.utils.db import db_session
from app.utils.tuning import select_best_tuning_result

import scripts.apply_best_weights as apply_best_weights


@pytest.fixture()
def isolated_env(tmp_path):
    cfg = get_config()
    original_paths = cfg.data_paths
    tmp_root = tmp_path / "data"
    tmp_root.mkdir(parents=True, exist_ok=True)
    cfg.data_paths = DataPaths(root=tmp_root)
    initialize_database()
    try:
        yield cfg
    finally:
        cfg.data_paths = original_paths


def _insert_result(experiment: str, reward: float, metrics: dict, weights: dict | None = None, action: dict | None = None) -> None:
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO tuning_results (experiment_id, strategy, action, weights, reward, metrics)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                experiment,
                "test",
                json.dumps(action or {}, ensure_ascii=False),
                json.dumps(weights or {}, ensure_ascii=False),
                reward,
                json.dumps(metrics, ensure_ascii=False),
            ),
        )


def test_select_best_by_reward(isolated_env):
    _insert_result("exp", 0.1, {"risk_count": 2}, {"A_mom": 0.3})
    _insert_result("exp", 0.25, {"risk_count": 4}, {"A_mom": 0.6})

    best = select_best_tuning_result("exp")
    assert best is not None
    assert best["reward"] == pytest.approx(0.25)
    assert best["weights"]["A_mom"] == pytest.approx(0.6)


def test_select_best_by_metric(isolated_env):
    _insert_result("exp_metric", 0.2, {"risk_count": 5}, {"A_mom": 0.4})
    _insert_result("exp_metric", 0.1, {"risk_count": 2}, {"A_mom": 0.7})

    best = select_best_tuning_result("exp_metric", metric="risk_count", descending=False)
    assert best is not None
    assert best["weights"]["A_mom"] == pytest.approx(0.7)
    assert best["metrics"]["risk_count"] == 2


def test_apply_best_weights_cli_updates_config(isolated_env, capsys):
    cfg = isolated_env
    _insert_result("exp_cli", 0.3, {"risk_count": 1}, {"A_mom": 0.65, "A_val": 0.2})
    exit_code = apply_best_weights.run_cli([
        "exp_cli",
        "--apply-config",
    ])
    assert exit_code == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["metric"] == "reward"
    updated = cfg.agent_weights.as_dict()
    assert updated["A_mom"] == pytest.approx(0.65)
    assert updated["A_val"] == pytest.approx(0.2)
