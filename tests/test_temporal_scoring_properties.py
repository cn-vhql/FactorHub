"""
属性测试：日频时序打分模块（temporal-scoring）
"""
import threading
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from hypothesis import given, settings as h_settings
from hypothesis import strategies as st

from backend.services.temporal_scoring_service import TemporalScoringService

FACTOR_NAMES = [
    "price_vs_sma20", "momentum_20", "price_vwma_ratio", "force_index_ma",
    "bollinger_position", "stochastic_d", "atr_norm", "volume_ma_ratio",
]


# ---------------------------------------------------------------------------
# Property 7: 百分位正确性（范围 + 单调性）
# Validates: Requirements 6.1, 6.4
# ---------------------------------------------------------------------------

# Feature: temporal-scoring, Property 7: 百分位正确性（范围 + 单调性）
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=20))
def test_compute_percentile_range_and_monotonicity(values):
    """
    Validates: Requirements 6.1, 6.4

    Property 7: For any factor time series of length >= 20,
    _compute_percentile always returns a value in [0, 1], and for the same
    history window a larger factor value gets a percentile >= smaller value's
    percentile (monotonicity).
    """
    service = TemporalScoringService()
    series = pd.Series(values)

    # Skip if all values are identical (percentile is still valid but
    # monotonicity is trivially satisfied; we still verify range)
    current_value = values[0]
    result = service._compute_percentile(series, current_value)

    # Range invariant: output must be in [0, 1]
    assert 0.0 <= result <= 1.0, f"Percentile {result} out of [0, 1] for value={current_value}"

    # Monotonicity: pick two distinct values from the series and verify ordering
    if len(values) >= 2:
        v_small = min(values)
        v_large = max(values)
        pct_small = service._compute_percentile(series, v_small)
        pct_large = service._compute_percentile(series, v_large)

        # Both must be in range
        assert 0.0 <= pct_small <= 1.0
        assert 0.0 <= pct_large <= 1.0

        # Monotonicity: larger value => percentile >= smaller value's percentile
        assert pct_large >= pct_small, (
            f"Monotonicity violated: pct({v_large})={pct_large} < pct({v_small})={pct_small}"
        )


# ---------------------------------------------------------------------------
# Property 6: 评分公式正确性
# Validates: Requirements 5.3, 5.4
# ---------------------------------------------------------------------------

# Feature: temporal-scoring, Property 6: 评分公式正确性
@given(st.fixed_dictionaries({factor: st.floats(0, 1) for factor in FACTOR_NAMES}))
@h_settings(max_examples=200)
def test_compute_score_formula_correctness(pcts):
    """
    Validates: Requirements 5.3, 5.4

    Property 6: For any combination of 8 factor percentiles in [0, 1],
    _compute_score output equals the manually computed
    clip(raw_score - penalties, 0, 100) and is always in [0, 100].
    """
    service = TemporalScoringService()

    # Manually compute expected score
    raw_score = 100 * (
        0.22 * pcts["price_vs_sma20"]
        + 0.18 * pcts["momentum_20"]
        + 0.16 * pcts["price_vwma_ratio"]
        + 0.14 * pcts["force_index_ma"]
        + 0.14 * pcts["bollinger_position"]
        + 0.16 * pcts["stochastic_d"]
    )

    trend_break = 1 if (pcts["price_vs_sma20"] < 0.35 or pcts["stochastic_d"] < 0.20) else 0
    high_volatility = 1 if pcts["atr_norm"] > 0.80 else 0
    liquidity_risk = 1 if pcts["volume_ma_ratio"] < 0.30 else 0
    penalties = 12 * trend_break + 10 * high_volatility + 8 * liquidity_risk

    expected = float(np.clip(raw_score - penalties, 0, 100))
    actual = service._compute_score(pcts)

    # Formula correctness
    assert abs(actual - expected) < 1e-9, (
        f"Score mismatch: got {actual}, expected {expected} for pcts={pcts}"
    )

    # Range invariant: output must be in [0, 100]
    assert 0.0 <= actual <= 100.0, f"Score {actual} out of [0, 100]"


# ---------------------------------------------------------------------------
# Property 2: 错误隔离不中断
# Validates: Requirements 1.4, 7.2, 7.3
# ---------------------------------------------------------------------------

_valid_code = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Nd")),
    min_size=1, max_size=8,
).filter(lambda c: not c.startswith("ERR_"))

_error_code = st.builds(lambda s: "ERR_" + s, st.text(min_size=1, max_size=6))

_mixed_codes = st.lists(
    st.one_of(_valid_code, _error_code),
    min_size=1,
    max_size=20,
)


# Feature: temporal-scoring, Property 2: 错误隔离不中断
@given(_mixed_codes)
def test_score_many_stocks_error_isolation(codes):
    """
    Validates: Requirements 1.4, 7.2, 7.3

    Property 2: When score_one_stock raises for codes starting with "ERR_",
    score_many_stocks must:
    - conserve total count: len(results) + len(errors) == len(codes)
    - route ERR_ codes to errors, others to results
    """
    service = TemporalScoringService()

    def mock_score_one_stock(code, trade_date):
        if code.startswith("ERR_"):
            raise ValueError(f"Simulated failure for {code}")
        return {"date": trade_date, "code": code, "score": 50.0, "factors": {}, "pcts": {}}

    with patch.object(service, "score_one_stock", side_effect=mock_score_one_stock):
        results, errors = service.score_many_stocks(codes, "2024-01-01")

    # Total conservation
    assert len(results) + len(errors) == len(codes), (
        f"Total mismatch: {len(results)} results + {len(errors)} errors != {len(codes)} codes"
    )

    result_codes = {r["code"] for r in results}
    error_codes = {e["code"] for e in errors}

    for code in codes:
        if code.startswith("ERR_"):
            assert code in error_codes, f"ERR_ code {code!r} should be in errors"
        else:
            assert code in result_codes, f"Valid code {code!r} should be in results"


# ---------------------------------------------------------------------------
# Property 8: 结果文件格式正确性
# Validates: Requirements 8.5, 9.2
# ---------------------------------------------------------------------------

import tempfile
from pathlib import Path

from backend.services.temporal_scoring_service import TaskManager

# Feature: temporal-scoring, Property 8: 结果文件格式正确性
@given(
    st.lists(
        st.fixed_dictionaries({
            "date": st.just("2024-01-01"),
            "code": st.from_regex(r"[0-9]{6}", fullmatch=True),
            "score": st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            "factors": st.just({
                "price_vs_sma20": 1.0, "momentum_20": 0.0, "price_vwma_ratio": 1.0,
                "force_index_ma": 0.0, "bollinger_position": 0.5, "stochastic_d": 0.5,
                "atr_norm": 0.02, "volume_ma_ratio": 1.0,
            }),
            "pcts": st.just({
                "price_vs_sma20": 0.5, "momentum_20": 0.5, "price_vwma_ratio": 0.5,
                "force_index_ma": 0.5, "bollinger_position": 0.5, "stochastic_d": 0.5,
                "atr_norm": 0.5, "volume_ma_ratio": 0.5,
            }),
        }),
        min_size=1,
        max_size=20,
    )
)
def test_score_file_format_correctness(results):
    """
    Validates: Requirements 8.5, 9.2

    Property 8: After writing N score rows via _append_score_row,
    daily_scores.csv must have:
    - first line == "date,code,score"
    - each score value formatted to exactly 1 decimal place
    - data row count == number of written records
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        task_id = "test-task"
        task_dir = Path(tmpdir) / task_id
        task_dir.mkdir()

        manager = TaskManager()
        manager.results_dir = Path(tmpdir)
        manager._file_locks[task_id] = threading.Lock()

        for result in results:
            manager._append_score_row(task_id, result)

        scores_path = task_dir / "daily_scores.csv"
        assert scores_path.exists(), "daily_scores.csv should exist after writing"

        lines = scores_path.read_text(encoding="utf-8").splitlines()

        # Header check
        assert lines[0] == "date,code,score", (
            f"Expected header 'date,code,score', got {lines[0]!r}"
        )

        data_lines = lines[1:]

        # Row count check
        assert len(data_lines) == len(results), (
            f"Expected {len(results)} data rows, got {len(data_lines)}"
        )

        # Score format check: each score must be formatted to 1 decimal place
        for i, line in enumerate(data_lines):
            parts = line.split(",")
            assert len(parts) == 3, f"Row {i+1} should have 3 columns: {line!r}"
            score_str = parts[2]
            # Must be parseable as float
            score_val = float(score_str)
            # Must have exactly 1 decimal place
            assert "." in score_str, f"Score {score_str!r} has no decimal point"
            decimal_part = score_str.split(".")[1]
            assert len(decimal_part) == 1, (
                f"Score {score_str!r} does not have exactly 1 decimal place"
            )


# ---------------------------------------------------------------------------
# Property 9: 异步任务 profile_id 持久化
# Validates: async job path correctly persists and reads profile_id
# ---------------------------------------------------------------------------

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.services.temporal_scoring_service import TaskManager


def test_create_task_persists_profile_id():
    """
    create_task(codes, date, profile_id="momentum_v1") must write profile_id
    into status.json so run_task can read it back.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TaskManager()
        manager.results_dir = Path(tmpdir)

        task_id = manager.create_task(["600519", "000651"], "2024-01-05", profile_id="momentum_v1")

        status_path = Path(tmpdir) / task_id / "status.json"
        assert status_path.exists()
        status = json.loads(status_path.read_text())
        assert status["profile_id"] == "momentum_v1", (
            f"profile_id not persisted: {status}"
        )


def test_create_task_persists_none_profile_id():
    """
    create_task without profile_id must write profile_id: null into status.json.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TaskManager()
        manager.results_dir = Path(tmpdir)

        task_id = manager.create_task(["600519"], "2024-01-05")

        status = json.loads((Path(tmpdir) / task_id / "status.json").read_text())
        assert "profile_id" in status
        assert status["profile_id"] is None


def test_run_task_uses_profile_id_from_status():
    """
    When run_task is called without profile_id argument, it must read profile_id
    from status.json and dispatch to score_one_stock_with_profile.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TaskManager()
        manager.results_dir = Path(tmpdir)
        manager.max_workers = 1

        task_id = manager.create_task(["600519"], "2024-01-05", profile_id="trend_flow_v1")

        mock_result = {
            "date": "2024-01-05",
            "code": "600519",
            "score": 72.0,
            "factors": {},
            "pcts": {},
        }
        mock_scoring = MagicMock()
        mock_scoring.score_one_stock_with_profile.return_value = mock_result
        mock_scoring.score_one_stock.return_value = mock_result
        manager.scoring_service = mock_scoring

        # Call run_task WITHOUT passing profile_id — it must read from status.json
        manager.run_task(task_id, ["600519"], "2024-01-05")

        mock_scoring.score_one_stock_with_profile.assert_called_once_with(
            "600519", "2024-01-05", "trend_flow_v1"
        )
        mock_scoring.score_one_stock.assert_not_called()


def test_run_task_uses_explicit_profile_id_over_status():
    """
    When run_task is called WITH profile_id argument, it must use that value
    (not the one in status.json).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TaskManager()
        manager.results_dir = Path(tmpdir)
        manager.max_workers = 1

        # status.json has profile_id="old_profile"
        task_id = manager.create_task(["600519"], "2024-01-05", profile_id="old_profile")

        mock_result = {
            "date": "2024-01-05",
            "code": "600519",
            "score": 55.0,
            "factors": {},
            "pcts": {},
        }
        mock_scoring = MagicMock()
        mock_scoring.score_one_stock_with_profile.return_value = mock_result
        manager.scoring_service = mock_scoring

        # Explicit profile_id="new_profile" should win
        manager.run_task(task_id, ["600519"], "2024-01-05", profile_id="new_profile")

        mock_scoring.score_one_stock_with_profile.assert_called_once_with(
            "600519", "2024-01-05", "new_profile"
        )


def test_run_task_without_profile_id_uses_score_one_stock():
    """
    When neither run_task argument nor status.json has a profile_id,
    it must fall back to score_one_stock (not score_one_stock_with_profile).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TaskManager()
        manager.results_dir = Path(tmpdir)
        manager.max_workers = 1

        task_id = manager.create_task(["600519"], "2024-01-05")  # no profile_id

        mock_result = {
            "date": "2024-01-05",
            "code": "600519",
            "score": 60.0,
            "factors": {},
            "pcts": {},
        }
        mock_scoring = MagicMock()
        mock_scoring.score_one_stock.return_value = mock_result
        manager.scoring_service = mock_scoring

        manager.run_task(task_id, ["600519"], "2024-01-05")

        mock_scoring.score_one_stock.assert_called_once_with("600519", "2024-01-05")
        mock_scoring.score_one_stock_with_profile.assert_not_called()


def test_auto_upgrade_path_passes_profile_id():
    """
    The auto-upgrade path in POST /temporal-score (codes > threshold) must
    pass profile_id to both create_task and run_task.
    This test verifies the router logic by calling the service layer directly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TaskManager()
        manager.results_dir = Path(tmpdir)
        manager.max_workers = 1

        codes = [f"{i:06d}" for i in range(1, 6)]  # 5 codes
        profile_id = "breakout_v1"

        task_id = manager.create_task(codes, "2024-01-05", profile_id)

        status = json.loads((Path(tmpdir) / task_id / "status.json").read_text())
        assert status["profile_id"] == profile_id

        mock_result_template = {
            "date": "2024-01-05",
            "score": 50.0,
            "factors": {},
            "pcts": {},
        }
        mock_scoring = MagicMock()
        mock_scoring.score_one_stock_with_profile.side_effect = lambda code, date, pid: {
            **mock_result_template, "code": code
        }
        manager.scoring_service = mock_scoring

        manager.run_task(task_id, codes, "2024-01-05")

        assert mock_scoring.score_one_stock_with_profile.call_count == len(codes)
        for call in mock_scoring.score_one_stock_with_profile.call_args_list:
            assert call.args[2] == profile_id
