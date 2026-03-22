from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_compute_stability_score_returns_zero_for_inactive_windows():
    from backend.services.strategy_evaluation_service import StrategyEvaluationService

    svc = StrategyEvaluationService.__new__(StrategyEvaluationService)
    score = svc.compute_stability_score(
        [
            {"test_metrics": {"sharpe": float("nan"), "annual_return": float("nan"), "active_ratio": 0.0}},
            {"test_metrics": {"sharpe": float("nan"), "annual_return": float("nan"), "active_ratio": 0.0}},
        ]
    )

    # After fix: returns dict with evaluation_status="invalid" for all-NaN windows
    if isinstance(score, dict):
        assert score.get("evaluation_status") == "invalid"
    else:
        assert score == 0.0


def test_single_stock_execute_search_raises_when_no_usable_candidate(tmp_path: Path):
    from backend.services.champion_strategy_service import ChampionStrategyService

    svc = ChampionStrategyService.__new__(ChampionStrategyService)
    svc._search_svc = MagicMock()
    svc._eval_svc = MagicMock()
    svc._registry = MagicMock()

    svc._search_svc.search_for_stock.return_value = [
        {
            "candidate_id": "candidate_1",
            "profile_id": "mean_reversion_v1",
            "mode": "temporal_pool",
            "source": "search_result",
        }
    ]
    svc._eval_svc.evaluate_temporal_pool_candidate.return_value = {
        "train_metrics": {},
        "valid_metrics": {},
        "test_metrics": {
            "sharpe": float("nan"),
            "annual_return": float("nan"),
            "active_ratio": 0.0,
        },
        "stability_score": 0.0,
        "window_metrics": [],
    }

    with pytest.raises(RuntimeError, match="未找到可用的 single_stock champion"):
        svc._execute_search(
            task_id="task-id",
            task_dir=tmp_path,
            scope_type="single_stock",
            scope_key="000001.SZ",
            search_space_id="temporal_default",
            start_date="2022-01-01",
            end_date="2026-03-21",
            codes=["000001.SZ"],
            universe=None,
            max_workers=1,
        )

    assert (tmp_path / "candidate_leaderboard.csv").exists()
    assert (tmp_path / "candidate_configs.json").exists()
