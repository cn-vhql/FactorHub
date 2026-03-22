"""
评估链路稳健性修复 — 属性测试

Task 1: Bug 条件探索测试（P0-2：无效指标仍可产出 champion）

**Validates: Requirements 1.3**

Bug 条件：compute_stability_score 接收到全部窗口指标均为 NaN 的 window_metrics 列表时，
当前代码对 NaN 使用默认值 0.0 进行计算，返回固定的非零分（约 27.5），而非标记为无效。

此测试预期在未修复代码上 **失败**，以证明 Bug 存在。
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from backend.services.strategy_evaluation_service import StrategyEvaluationService


# ---------------------------------------------------------------------------
# 策略：生成全 NaN test_metrics 的 window_metrics 列表
# ---------------------------------------------------------------------------

_NAN = float("nan")

_nan_test_metrics = st.just(
    {
        "sharpe": _NAN,
        "annual_return": _NAN,
        "max_drawdown": _NAN,
        "ir": _NAN,
        "excess_return": _NAN,
        "turnover_rate": _NAN,
        "active_ratio": _NAN,
        "win_rate": _NAN,
        "volatility": _NAN,
        "total_return": _NAN,
        "active_days": _NAN,
        "trading_days": _NAN,
        "active_rebalance_count": _NAN,
        "rebalance_count": _NAN,
    }
)

_nan_window = st.fixed_dictionaries(
    {
        "window_id": st.integers(min_value=0, max_value=100),
        "train_start": st.just("2020-01-01"),
        "train_end": st.just("2020-12-31"),
        "valid_start": st.just("2021-01-01"),
        "valid_end": st.just("2021-03-31"),
        "test_start": st.just("2021-04-01"),
        "test_end": st.just("2021-06-30"),
        "train_metrics": _nan_test_metrics,
        "valid_metrics": _nan_test_metrics,
        "test_metrics": _nan_test_metrics,
        "overfitting_flag": st.just(False),
    }
)

# 生成 1~10 个全 NaN 窗口的列表（均少于 min_valid_windows=3 个有效窗口）
_all_nan_window_metrics = st.lists(_nan_window, min_size=1, max_size=10)


# ---------------------------------------------------------------------------
# Bug 条件探索测试（预期在未修复代码上失败）
# ---------------------------------------------------------------------------

@given(window_metrics=_all_nan_window_metrics)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_all_nan_windows_bug_condition(window_metrics):
    """
    **Validates: Requirements 1.3**

    属性：当所有窗口的 test_metrics 均为 NaN 时，
    compute_stability_score 应返回 NaN（或 evaluation_status="invalid"）。

    此测试在未修复代码上 **预期失败**：
    当前实现将 NaN 替换为 0.0 默认值，返回约 27.5 的固定分，而非 NaN。
    测试失败即证明 Bug 存在。
    """
    svc = StrategyEvaluationService.__new__(StrategyEvaluationService)
    result = svc.compute_stability_score(window_metrics)

    # 修复后的期望行为：返回 NaN 或包含 evaluation_status="invalid" 的结构
    if isinstance(result, dict):
        # 修复后可能返回结构体
        score = result.get("stability_score", result)
        status = result.get("evaluation_status", "")
        assert math.isnan(float(score)) or status == "invalid", (
            f"全 NaN 窗口指标应返回 NaN 稳定性分或 evaluation_status='invalid'，"
            f"但得到: score={score}, status={status}"
        )
    else:
        # 当前返回 float，修复后应为 NaN
        assert math.isnan(float(result)), (
            f"全 NaN 窗口指标应返回 NaN 稳定性分，"
            f"但得到: {result}（Bug：NaN 被替换为 0.0 默认值，产出约 27.5 的固定分）"
        )


# ---------------------------------------------------------------------------
# Task 2: 保留属性测试（正常窗口指标下的稳定性计算）
# ---------------------------------------------------------------------------

# 生成有效（非 NaN、有限）的 test_metrics 字典
_valid_test_metrics = st.fixed_dictionaries(
    {
        "sharpe": st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        "annual_return": st.floats(min_value=-0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
        "max_drawdown": st.floats(min_value=-0.9, max_value=0.0, allow_nan=False, allow_infinity=False),
        "ir": st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        "excess_return": st.floats(min_value=-0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
        "turnover_rate": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "active_ratio": st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        "win_rate": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "volatility": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "total_return": st.floats(min_value=-0.9, max_value=5.0, allow_nan=False, allow_infinity=False),
        "active_days": st.floats(min_value=1.0, max_value=252.0, allow_nan=False, allow_infinity=False),
        "trading_days": st.floats(min_value=1.0, max_value=252.0, allow_nan=False, allow_infinity=False),
        "active_rebalance_count": st.floats(min_value=0.0, max_value=52.0, allow_nan=False, allow_infinity=False),
        "rebalance_count": st.floats(min_value=1.0, max_value=52.0, allow_nan=False, allow_infinity=False),
    }
)

_valid_window = st.fixed_dictionaries(
    {
        "window_id": st.integers(min_value=0, max_value=100),
        "train_start": st.just("2020-01-01"),
        "train_end": st.just("2020-12-31"),
        "valid_start": st.just("2021-01-01"),
        "valid_end": st.just("2021-03-31"),
        "test_start": st.just("2021-04-01"),
        "test_end": st.just("2021-06-30"),
        "train_metrics": _valid_test_metrics,
        "valid_metrics": _valid_test_metrics,
        "test_metrics": _valid_test_metrics,
        "overfitting_flag": st.just(False),
    }
)

# 生成 ≥3 个有效窗口的列表（满足 min_valid_windows=3）
_valid_window_metrics = st.lists(_valid_window, min_size=3, max_size=10)


@given(window_metrics=_valid_window_metrics)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_valid_windows_preservation(window_metrics):
    """
    **Validates: Requirements 3.2**

    保留属性：当有效窗口数 ≥ min_valid_windows（3）且所有 test_metrics 均为有效数值时，
    compute_stability_score 应按原有 7 项权重公式计算，返回值范围 [0.0, 100.0]。

    此测试在未修复代码和修复后代码上均应 **通过**（保留行为不变）。
    """
    svc = StrategyEvaluationService.__new__(StrategyEvaluationService)
    result = svc.compute_stability_score(window_metrics)

    # 处理修复后可能返回结构体的情况
    if isinstance(result, dict):
        score = result.get("stability_score", result)
        status = result.get("evaluation_status", "valid")
        assert status == "valid", (
            f"有效窗口数 ≥ 3 时 evaluation_status 应为 'valid'，但得到: {status}"
        )
        score = float(score)
    else:
        score = float(result)

    assert not math.isnan(score), (
        f"有效窗口数 ≥ 3 时稳定性分不应为 NaN，但得到: {score}"
    )
    assert 0.0 <= score <= 100.0, (
        f"稳定性分应在 [0.0, 100.0] 范围内，但得到: {score}"
    )


# ---------------------------------------------------------------------------
# Task 3.1: P0-1 schema 迁移测试
# ---------------------------------------------------------------------------

def test_schema_migration_p0_1():
    """
    测试 init_db() 中的幂等 schema 迁移逻辑（P0-1）。

    1. 创建不含 formula_type 列的旧版 factors 表（模拟旧数据库）
    2. 调用迁移逻辑
    3. 断言 formula_type 列现在存在，应用可正常启动
    4. 再次调用迁移逻辑（幂等性验证），不应报错
    """
    import sqlite3
    from sqlalchemy import create_engine, text
    from backend.core.database import _run_migrations, _column_exists

    # 使用内存数据库模拟旧版 schema（不含 formula_type 列）
    old_schema_sql = """
        CREATE TABLE factors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL UNIQUE,
            code TEXT NOT NULL,
            description TEXT DEFAULT '',
            source VARCHAR(20) NOT NULL DEFAULT 'user',
            category VARCHAR(50) DEFAULT '',
            is_active INTEGER DEFAULT 1,
            created_at DATETIME,
            updated_at DATETIME
        );
    """

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    with engine.begin() as conn:
        conn.execute(text(old_schema_sql))
        # 插入一条历史数据（无 formula_type）
        conn.execute(text(
            "INSERT INTO factors (name, code, source) VALUES ('test_factor', 'close/open', 'user')"
        ))

    # 验证迁移前 formula_type 列不存在
    with engine.connect() as conn:
        assert not _column_exists(conn, "factors", "formula_type"), \
            "测试前提：formula_type 列不应存在"

    # 执行迁移
    with engine.begin() as conn:
        _run_migrations(conn)  # 不应抛出异常

    # 验证迁移后 formula_type 列存在
    with engine.connect() as conn:
        assert _column_exists(conn, "factors", "formula_type"), \
            "迁移后 formula_type 列应存在"

        # 验证历史数据已回填默认值
        result = conn.execute(text("SELECT formula_type FROM factors WHERE name='test_factor'"))
        row = result.fetchone()
        assert row is not None
        assert row[0] == "expression", \
            f"历史数据应回填 'expression'，但得到: {row[0]}"

        # 验证 schema_migrations 表已记录
        result = conn.execute(text(
            "SELECT migration_id FROM schema_migrations WHERE migration_id='add_formula_type_to_factors'"
        ))
        assert result.fetchone() is not None, "schema_migrations 表应记录已完成的迁移"

    # 幂等性验证：再次调用迁移不应报错
    with engine.begin() as conn:
        _run_migrations(conn)  # 应静默跳过，不抛出异常

    # 验证列仍然存在（未被破坏）
    with engine.connect() as conn:
        assert _column_exists(conn, "factors", "formula_type"), \
            "幂等调用后 formula_type 列应仍然存在"


# ---------------------------------------------------------------------------
# Task 5.2: 覆盖率门禁测试（P1-1）
# ---------------------------------------------------------------------------

def test_coverage_gate_p1_1():
    """
    **Validates: Requirements 2.7**

    测试覆盖率门禁：当 get_multiple_stocks_data_with_report 返回的覆盖率低于
    EVAL_MIN_CODE_COVERAGE（0.85）时，StrategyEvaluationService.evaluate_cross_sectional_candidate
    应直接返回 result_status="invalid_run"，reason_codes 包含 CODE_COVERAGE_TOO_LOW。
    """
    from unittest.mock import MagicMock, patch
    from backend.services.data_service import DataFetchReport
    from backend.services.strategy_evaluation_service import StrategyEvaluationService

    # 构造低覆盖率报告：请求 10 只股票，只成功 5 只（覆盖率 50% < 85%）
    requested = [f"00000{i}" for i in range(10)]
    success = requested[:5]
    failed = requested[5:]
    low_coverage_report = DataFetchReport(
        requested_codes=requested,
        success_codes=success,
        failed_codes=failed,
        data={code: MagicMock() for code in success},
        failures=[{"code": c, "error_type": "ValueError", "message": "mock failure"} for c in failed],
    )

    svc = StrategyEvaluationService.__new__(StrategyEvaluationService)

    candidate = {
        "profile_id": "test_profile",
        "universe_codes": requested,
        "config": {
            "factors": [{"name": "momentum", "weight": 1.0, "role": "signal"}],
            "params": {"top_percentile": 0.2},
        },
    }

    # mock get_multiple_stocks_data_with_report 返回低覆盖率报告
    with patch(
        "backend.services.data_service.DataService.get_multiple_stocks_data_with_report",
        return_value=low_coverage_report,
    ), patch(
        "backend.services.vectorbt_backtest_service.VectorBTBacktestService",
        MagicMock(),
    ):
        result = svc.evaluate_cross_sectional_candidate(
            candidate=candidate,
            universe="test_universe",
            start_date="2023-01-01",
            end_date="2024-12-31",
        )

    assert isinstance(result, dict), f"低覆盖率时应返回 dict，但得到: {type(result)}"
    assert result.get("result_status") == "invalid_run", (
        f"result_status 应为 'invalid_run'，但得到: {result.get('result_status')}"
    )
    assert "CODE_COVERAGE_TOO_LOW" in result.get("reason_codes", []), (
        f"reason_codes 应包含 'CODE_COVERAGE_TOO_LOW'，但得到: {result.get('reason_codes')}"
    )


# ---------------------------------------------------------------------------
# Task 6.1: 缓存版本指纹测试（P1-2）
# ---------------------------------------------------------------------------

def test_cache_version_fingerprint_p1_2():
    """
    **Validates: Requirements 2.8, 3.5**

    测试缓存版本指纹：
    1. 相同输入两次调用 build_versioned_cache_key → 相同 key（缓存命中场景）
    2. 不同因子表达式 → 不同 key（缓存失效场景）
    3. 不同 profile 定义 → 不同 key（缓存失效场景）
    """
    from backend.services.cache_service import build_versioned_cache_key

    base_key = "stock_001:2024-01-01:profile_A"
    factor_expr = "close / open"
    profile_def = '{"top_percentile": 0.2}'

    # 1. 相同输入两次调用 → 相同 key（缓存命中）
    key1 = build_versioned_cache_key(base_key, factor_expr, profile_def)
    key2 = build_versioned_cache_key(base_key, factor_expr, profile_def)
    assert key1 == key2, (
        f"相同输入应生成相同缓存 key，但得到: key1={key1}, key2={key2}"
    )

    # 2. 不同因子表达式 → 不同 key（缓存失效）
    key_diff_factor = build_versioned_cache_key(base_key, "close / high", profile_def)
    assert key1 != key_diff_factor, (
        f"不同因子表达式应生成不同缓存 key，但两者相同: {key1}"
    )

    # 3. 不同 profile 定义 → 不同 key（缓存失效）
    key_diff_profile = build_versioned_cache_key(base_key, factor_expr, '{"top_percentile": 0.3}')
    assert key1 != key_diff_profile, (
        f"不同 profile 定义应生成不同缓存 key，但两者相同: {key1}"
    )


# ---------------------------------------------------------------------------
# Task 7.1: run_manifest.json 写出测试（P2-1）
# ---------------------------------------------------------------------------

def test_run_manifest_p2_1(tmp_path):
    """
    **Validates: Requirements 2.9, 3.7**

    测试 _write_run_manifest 辅助函数：
    1. 创建临时目录
    2. 调用 _write_run_manifest 写出 run_manifest.json
    3. 读取并验证所有必需字段存在
    4. 验证 quality_gate.passed 在 reason_codes 为空时为 True，非空时为 False
    """
    import json
    from backend.services.champion_strategy_service import _write_run_manifest

    # --- 场景 1：quality_gate 通过（reason_codes 为空）---
    task_dir = tmp_path / "task_passed"
    task_dir.mkdir()

    _write_run_manifest(
        task_dir=task_dir,
        module="champion_strategy",
        data_coverage=0.95,
        quality_gate=[],
        result_status="completed",
    )

    manifest_path = task_dir / "run_manifest.json"
    assert manifest_path.exists(), "run_manifest.json 应被写出"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 验证所有必需字段存在
    required_fields = [
        "run_id", "module", "generated_at", "schema_version",
        "settings_fingerprint", "data_coverage", "quality_gate", "result_status",
    ]
    for field in required_fields:
        assert field in manifest, f"run_manifest.json 缺少必需字段: {field}"

    # 验证字段值
    assert manifest["module"] == "champion_strategy"
    assert manifest["schema_version"] == "1.0"
    assert manifest["data_coverage"] == 0.95
    assert manifest["result_status"] == "completed"
    assert len(manifest["settings_fingerprint"]) == 12, "settings_fingerprint 应为 12 位 hex"

    # quality_gate 字段结构
    qg = manifest["quality_gate"]
    assert "passed" in qg, "quality_gate 应包含 passed 字段"
    assert "reason_codes" in qg, "quality_gate 应包含 reason_codes 字段"
    assert qg["passed"] is True, "reason_codes 为空时 quality_gate.passed 应为 True"
    assert qg["reason_codes"] == [], "reason_codes 应为空列表"

    # --- 场景 2：quality_gate 未通过（reason_codes 非空）---
    task_dir2 = tmp_path / "task_failed"
    task_dir2.mkdir()

    _write_run_manifest(
        task_dir=task_dir2,
        module="temporal_filter_validation",
        data_coverage=0.60,
        quality_gate=["CODE_COVERAGE_TOO_LOW", "INSUFFICIENT_IC_SAMPLES"],
        result_status="invalid_run",
    )

    manifest2 = json.loads((task_dir2 / "run_manifest.json").read_text(encoding="utf-8"))

    qg2 = manifest2["quality_gate"]
    assert qg2["passed"] is False, "reason_codes 非空时 quality_gate.passed 应为 False"
    assert "CODE_COVERAGE_TOO_LOW" in qg2["reason_codes"]
    assert "INSUFFICIENT_IC_SAMPLES" in qg2["reason_codes"]
    assert manifest2["result_status"] == "invalid_run"
    assert manifest2["data_coverage"] == 0.60
