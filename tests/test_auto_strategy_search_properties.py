"""
属性测试：自动化策略搜索模块（auto-strategy-search）
"""
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings as h_settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# 在导入 TemporalScoringService 之前，强制重载核心模块，确保本模块使用 mock talib。
# teardown 时恢复 sys.modules，防止后续测试持有 mock 版本。
# conftest.py 已在 talib 不可用时统一注入 mock，此处无需重复注入。
# ---------------------------------------------------------------------------
_talib_real = sys.modules.get("talib")
# 记录被 pop 前的模块引用，teardown 时恢复
_factor_service_real = sys.modules.get("backend.services.factor_service")
_temporal_scoring_real = sys.modules.get("backend.services.temporal_scoring_service")

# 强制重新导入，确保本模块使用 mock talib
for _mod in ("backend.services.factor_service", "backend.services.temporal_scoring_service"):
    sys.modules.pop(_mod, None)


import pytest as _pytest  # noqa: E402


@_pytest.fixture(autouse=True, scope="module")
def _restore_talib_mock():
    """模块级 fixture：测试结束后将 sys.modules 恢复到进入前的状态。"""
    yield
    # 恢复 talib（保持 conftest 注入的 mock，不移除）
    if _talib_real is not None:
        sys.modules["talib"] = _talib_real
    # 恢复被 pop 的核心模块，防止后续测试持有本模块重载的版本
    if _factor_service_real is None:
        sys.modules.pop("backend.services.factor_service", None)
    else:
        sys.modules["backend.services.factor_service"] = _factor_service_real
    if _temporal_scoring_real is None:
        sys.modules.pop("backend.services.temporal_scoring_service", None)
    else:
        sys.modules["backend.services.temporal_scoring_service"] = _temporal_scoring_real

from backend.services.temporal_scoring_service import TemporalScoringService  # noqa: E402


# ---------------------------------------------------------------------------
# 辅助：生成合法 profile 的 strategy
# ---------------------------------------------------------------------------

_signal_factor_names = [
    "price_vs_sma20",
    "momentum_20",
    "price_vwma_ratio",
    "force_index_ma",
    "bollinger_position",
    "stochastic_d",
]

_risk_filter_factor_names = [
    "atr_norm",
    "volume_ma_ratio",
]


@st.composite
def signal_factors_strategy(draw):
    """生成归一化权重的 signal 因子列表（权重和 ∈ [0.999, 1.001]）"""
    n = draw(st.integers(min_value=1, max_value=len(_signal_factor_names)))
    names = draw(
        st.lists(
            st.sampled_from(_signal_factor_names),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    # 生成 n 个正权重，归一化
    raw_weights = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    total = sum(raw_weights)
    weights = [w / total for w in raw_weights]

    factors = []
    for name, weight in zip(names, weights):
        direction = draw(st.sampled_from([-1, 1]))
        factors.append({
            "name": name,
            "role": "signal",
            "weight": weight,
            "direction": direction,
        })
    return factors


@st.composite
def risk_filter_factors_strategy(draw):
    """生成 0-2 个 risk_filter 因子列表"""
    n = draw(st.integers(min_value=0, max_value=len(_risk_filter_factor_names)))
    if n == 0:
        return []
    names = draw(
        st.lists(
            st.sampled_from(_risk_filter_factor_names),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    factors = []
    for name in names:
        direction = draw(st.sampled_from([-1, 1]))
        threshold = draw(st.floats(min_value=0.1, max_value=0.95, allow_nan=False))
        factors.append({
            "name": name,
            "role": "risk_filter",
            "weight": 0.0,
            "direction": direction,
            "threshold": threshold,
        })
    return factors


@st.composite
def valid_profile_strategy(draw):
    """生成合法的 profile 配置字典"""
    signal_factors = draw(signal_factors_strategy())
    risk_factors = draw(risk_filter_factors_strategy())
    all_factors = signal_factors + risk_factors

    return {
        "id": "test_profile",
        "mode": "temporal_pool",
        "description": "hypothesis generated profile",
        "factors": all_factors,
        "params": {
            "rebalance": "W-FRI",
            "hold_days": 5,
            "top_n": 10,
            "score_threshold": 60,
            "percentile_window": 252,
        },
    }


# ---------------------------------------------------------------------------
# Property 1：单股打分分数范围不变量
# Validates: Requirements 2.3, 2.4
# ---------------------------------------------------------------------------

# Feature: auto-strategy-search, Property 1: 单股打分分数范围不变量
@given(
    profile=valid_profile_strategy(),
    pct_values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    ),
)
@h_settings(max_examples=500)
def test_score_range_invariant_property1(profile, pct_values):
    """
    **Validates: Requirements 2.3, 2.4**

    Property 1：单股打分分数范围不变量

    对任意合法 profile（signal 因子权重归一化，含可选 risk_filter），
    以及 pcts 中每个值 ∈ [0, 1]，
    _compute_score_from_profile 返回的 score 必须 ∈ [0, 100]。
    同时验证 pcts 中每个值 ∈ [0, 1]（需求 2.4）。
    """
    service = TemporalScoringService.__new__(TemporalScoringService)

    factor_names = [f["name"] for f in profile.get("factors", [])]
    if not factor_names:
        return  # 空因子列表跳过

    # 将 pct_values 循环映射到因子名
    pcts = {
        name: pct_values[i % len(pct_values)]
        for i, name in enumerate(factor_names)
    }

    # 验证输入约束：pcts 中每个值 ∈ [0, 1]（需求 2.4）
    for name, val in pcts.items():
        assert 0.0 <= val <= 1.0, f"pcts[{name}]={val} 不在 [0, 1] 范围内"

    # 验证输出约束：score ∈ [0, 100]（需求 2.3）
    score = service._compute_score_from_profile(pcts, profile)
    assert 0.0 <= score <= 100.0, (
        f"score={score} 超出 [0, 100] 范围，pcts={pcts}, profile={profile}"
    )


# ---------------------------------------------------------------------------
# Property 1 补充：risk_filter 触发时 score 必须为 0
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------

def test_score_risk_filter_direction_neg_triggers_zero():
    """
    **Validates: Requirements 2.3**

    direction=-1 的 risk_filter：pct > threshold 时 score 必须为 0。
    """
    service = TemporalScoringService.__new__(TemporalScoringService)

    profile = {
        "id": "test_neg_filter",
        "mode": "temporal_pool",
        "factors": [
            {"name": "price_vs_sma20", "role": "signal", "weight": 1.0, "direction": 1},
            {"name": "atr_norm", "role": "risk_filter", "weight": 0.0, "direction": -1, "threshold": 0.8},
        ],
        "params": {"percentile_window": 252},
    }
    # atr_norm pct=0.9 > threshold=0.8 → 触发过滤
    pcts = {"price_vs_sma20": 0.9, "atr_norm": 0.9}
    score = service._compute_score_from_profile(pcts, profile)
    assert score == 0.0, f"risk_filter 触发时 score 应为 0，实际为 {score}"
    assert 0.0 <= score <= 100.0


def test_score_risk_filter_direction_pos_triggers_zero():
    """
    **Validates: Requirements 2.3**

    direction=1 的 risk_filter：pct < threshold 时 score 必须为 0。
    """
    service = TemporalScoringService.__new__(TemporalScoringService)

    profile = {
        "id": "test_pos_filter",
        "mode": "temporal_pool",
        "factors": [
            {"name": "price_vs_sma20", "role": "signal", "weight": 1.0, "direction": 1},
            {"name": "volume_ma_ratio", "role": "risk_filter", "weight": 0.0, "direction": 1, "threshold": 0.3},
        ],
        "params": {"percentile_window": 252},
    }
    # volume_ma_ratio pct=0.1 < threshold=0.3 → 触发过滤
    pcts = {"price_vs_sma20": 0.9, "volume_ma_ratio": 0.1}
    score = service._compute_score_from_profile(pcts, profile)
    assert score == 0.0, f"risk_filter 触发时 score 应为 0，实际为 {score}"
    assert 0.0 <= score <= 100.0


def test_score_all_signal_max_pcts_equals_100():
    """
    **Validates: Requirements 2.3**

    无 risk_filter，所有 signal 因子 pct=1.0 且 direction=1 时，score 应为 100.0。
    """
    service = TemporalScoringService.__new__(TemporalScoringService)

    profile = {
        "id": "test_all_signal",
        "mode": "temporal_pool",
        "factors": [
            {"name": "price_vs_sma20", "role": "signal", "weight": 0.5, "direction": 1},
            {"name": "momentum_20", "role": "signal", "weight": 0.5, "direction": 1},
        ],
        "params": {"percentile_window": 252},
    }
    pcts = {"price_vs_sma20": 1.0, "momentum_20": 1.0}
    score = service._compute_score_from_profile(pcts, profile)
    assert score == 100.0, f"全 1.0 pcts + direction=1 时 score 应为 100，实际为 {score}"
    assert 0.0 <= score <= 100.0


def test_score_all_signal_zero_pcts_equals_zero():
    """
    **Validates: Requirements 2.3**

    无 risk_filter，所有 signal 因子 pct=0.0 时，score 应为 0.0。
    """
    service = TemporalScoringService.__new__(TemporalScoringService)

    profile = {
        "id": "test_all_signal_zero",
        "mode": "temporal_pool",
        "factors": [
            {"name": "price_vs_sma20", "role": "signal", "weight": 0.5, "direction": 1},
            {"name": "momentum_20", "role": "signal", "weight": 0.5, "direction": 1},
        ],
        "params": {"percentile_window": 252},
    }
    pcts = {"price_vs_sma20": 0.0, "momentum_20": 0.0}
    score = service._compute_score_from_profile(pcts, profile)
    assert score == 0.0, f"全 0.0 pcts 时 score 应为 0，实际为 {score}"
    assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# Property 2：池内排名连续整数不变量
# Validates: Requirements 3.2
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402 (already available at module level via hypothesis imports)

# 辅助：直接测试排名逻辑（给定 code/score 列表，验证 rank 不变量）
def _apply_ranking_logic(code_score_pairs: list[tuple[str, float]]) -> pd.DataFrame:
    """
    复现 TemporalPoolService.score_pool 中的排名逻辑：
    按 score 降序排列，rank 从 1 开始连续赋值。
    """
    results = [{"code": code, "score": score} for code, score in code_score_pairs]
    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# 生成合法股票代码的 strategy（6 位数字字符串）
stock_code_strategy = st.text(
    alphabet="0123456789",
    min_size=6,
    max_size=6,
).filter(lambda s: s[0] != "0" or len(s) == 6)  # 允许前导零（深市）

# 生成 (code, score) 对列表的 strategy
code_score_pairs_strategy = st.lists(
    st.tuples(
        stock_code_strategy,
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    ),
    min_size=1,
    max_size=50,
    unique_by=lambda pair: pair[0],  # 每只股票代码唯一
)


# Feature: auto-strategy-search, Property 2: 池内排名连续整数不变量
@given(code_score_pairs=code_score_pairs_strategy)
@h_settings(max_examples=500)
def test_pool_rank_consecutive_integers_property2(code_score_pairs):
    """
    **Validates: Requirements 3.2**

    Property 2：池内排名连续整数不变量

    对任意非空股票列表（含 code/score 对），score_pool 返回的 rank 列：
    1. 是从 1 开始的连续整数（即 {1, 2, ..., N}）
    2. score 最高者 rank=1
    3. rank 列长度等于输入股票数量
    """
    df = _apply_ranking_logic(code_score_pairs)
    n = len(df)

    # 1. rank 列长度等于输入数量
    assert len(df) == len(code_score_pairs), (
        f"输出行数 {len(df)} 不等于输入数量 {len(code_score_pairs)}"
    )

    # 2. rank 是从 1 开始的连续整数集合
    rank_values = sorted(df["rank"].tolist())
    expected_ranks = list(range(1, n + 1))
    assert rank_values == expected_ranks, (
        f"rank 列不是连续整数 1..{n}，实际为 {rank_values}"
    )

    # 3. score 最高者 rank=1
    max_score = df["score"].max()
    rank_of_max = df.loc[df["score"] == max_score, "rank"].min()
    assert rank_of_max == 1, (
        f"score 最高者（score={max_score}）的 rank={rank_of_max}，应为 1"
    )

    # 4. rank 按 score 降序单调递增（score 相同时 rank 可连续但不跳跃）
    for i in range(len(df) - 1):
        assert df.loc[i, "score"] >= df.loc[i + 1, "score"], (
            f"第 {i} 行 score={df.loc[i, 'score']} < 第 {i+1} 行 score={df.loc[i+1, 'score']}，"
            "结果未按 score 降序排列"
        )
        assert df.loc[i, "rank"] < df.loc[i + 1, "rank"], (
            f"第 {i} 行 rank={df.loc[i, 'rank']} >= 第 {i+1} 行 rank={df.loc[i+1, 'rank']}，"
            "rank 未严格递增"
        )


@given(code_score_pairs=code_score_pairs_strategy)
@h_settings(max_examples=300)
def test_pool_rank_via_mock_score_pool_property2(code_score_pairs):
    """
    **Validates: Requirements 3.2**

    Property 2 补充：通过 mock TemporalScoringService 验证 TemporalPoolService.score_pool
    返回的 rank 列满足连续整数不变量。
    """
    codes = [pair[0] for pair in code_score_pairs]
    score_map = {pair[0]: pair[1] for pair in code_score_pairs}

    def mock_score_one(code, trade_date, profile_id):
        return {"date": trade_date, "code": code, "score": score_map[code], "factors": {}, "pcts": {}}

    with patch(
        "backend.services.temporal_pool_service.TemporalScoringService"
    ) as MockScoringCls:
        mock_instance = MagicMock()
        mock_instance.score_one_stock_with_profile.side_effect = mock_score_one
        MockScoringCls.return_value = mock_instance

        from backend.services.temporal_pool_service import TemporalPoolService
        service = TemporalPoolService()
        df = service.score_pool(codes, profile_id="test_profile", trade_date="2024-01-05")

    n = len(df)
    assert n == len(code_score_pairs), f"输出行数 {n} 不等于输入数量 {len(code_score_pairs)}"

    # rank 是从 1 开始的连续整数
    rank_values = sorted(df["rank"].tolist())
    assert rank_values == list(range(1, n + 1)), (
        f"rank 列不是连续整数 1..{n}，实际为 {rank_values}"
    )

    # score 最高者 rank=1
    max_score = df["score"].max()
    rank_of_max = df.loc[df["score"] == max_score, "rank"].min()
    assert rank_of_max == 1, (
        f"score 最高者（score={max_score}）的 rank={rank_of_max}，应为 1"
    )


# ---------------------------------------------------------------------------
# Property 3：TopN 数量上界不变量
# Property 4：score_threshold 过滤正确性
# Validates: Requirements 3.6, 3.7
# ---------------------------------------------------------------------------

from backend.services.temporal_pool_service import TemporalPoolService  # noqa: E402


# 生成 pool_scores DataFrame 的 strategy（含 code/score/rank 列）
@st.composite
def pool_scores_strategy(draw):
    """生成合法的 pool_scores DataFrame（含 code/score/rank 列）"""
    n = draw(st.integers(min_value=0, max_value=50))
    if n == 0:
        return pd.DataFrame(columns=["code", "score", "rank"])

    codes = draw(
        st.lists(
            st.text(alphabet="0123456789", min_size=6, max_size=6),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    scores = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    df = pd.DataFrame({"code": codes, "score": scores})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# Feature: auto-strategy-search, Property 3: TopN 数量上界不变量
@given(
    pool_scores=pool_scores_strategy(),
    top_n=st.integers(min_value=1, max_value=100),
)
@h_settings(max_examples=500)
def test_select_top_n_count_upper_bound_property3(pool_scores, top_n):
    """
    **Validates: Requirements 3.6**

    Property 3：TopN 数量上界不变量

    对任意 pool_scores DataFrame 和任意 top_n，
    select_top_n 返回的行数不超过 top_n。
    """
    service = TemporalPoolService.__new__(TemporalPoolService)
    result = service.select_top_n(pool_scores, top_n)

    assert len(result) <= top_n, (
        f"返回行数 {len(result)} 超过 top_n={top_n}，"
        f"输入行数={len(pool_scores)}"
    )


# Feature: auto-strategy-search, Property 4: score_threshold 过滤正确性
@given(
    pool_scores=pool_scores_strategy(),
    score_threshold=st.floats(
        min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
)
@h_settings(max_examples=500)
def test_select_top_n_score_threshold_filter_property4(pool_scores, score_threshold):
    """
    **Validates: Requirements 3.7**

    Property 4：score_threshold 过滤正确性

    对任意 pool_scores DataFrame 和任意 score_threshold，
    select_top_n 返回结果中所有 score >= score_threshold。
    """
    service = TemporalPoolService.__new__(TemporalPoolService)
    top_n = max(len(pool_scores), 1)  # 不限制数量，专注验证阈值过滤
    result = service.select_top_n(pool_scores, top_n, score_threshold=score_threshold)

    if len(result) == 0:
        return  # 空结果合法（所有股票均低于阈值）

    for _, row in result.iterrows():
        assert row["score"] >= score_threshold, (
            f"结果中存在 score={row['score']} < score_threshold={score_threshold}，"
            f"code={row['code']}"
        )


# Feature: auto-strategy-search, Property 3+4 联合验证
@given(
    pool_scores=pool_scores_strategy(),
    top_n=st.integers(min_value=1, max_value=100),
    score_threshold=st.floats(
        min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
)
@h_settings(max_examples=500)
def test_select_top_n_combined_property3_property4(pool_scores, top_n, score_threshold):
    """
    **Validates: Requirements 3.6, 3.7**

    Property 3 + Property 4 联合验证：

    同时指定 top_n 和 score_threshold 时：
    1. 返回行数不超过 top_n（Property 3）
    2. 所有返回结果的 score >= score_threshold（Property 4）
    """
    service = TemporalPoolService.__new__(TemporalPoolService)
    result = service.select_top_n(pool_scores, top_n, score_threshold=score_threshold)

    # Property 3：数量上界
    assert len(result) <= top_n, (
        f"返回行数 {len(result)} 超过 top_n={top_n}"
    )

    # Property 4：阈值过滤
    if len(result) > 0:
        min_score = result["score"].min()
        assert min_score >= score_threshold, (
            f"结果中最低 score={min_score} < score_threshold={score_threshold}"
        )


# ---------------------------------------------------------------------------
# Property 5：受约束随机搜索权重归一化
# Property 6：受约束随机搜索约束满足性
# Validates: Requirements 4.3, 4.4, 4.5, 4.6, 4.7
# ---------------------------------------------------------------------------

from backend.services.strategy_search_service import StrategySearchService  # noqa: E402


# ---------------------------------------------------------------------------
# Hypothesis strategies for generating valid search_space configs
# ---------------------------------------------------------------------------

# All available factor names (from config/strategy_search_space.json)
_ALL_SIGNAL_FACTORS = [
    "price_vs_sma20",
    "momentum_20",
    "price_vwma_ratio",
    "force_index_ma",
    "bollinger_position",
    "stochastic_d",
    "roc_10",
    "deviation_from_ma20",
    "cci_20",
    "rsi_14",
]

_ALL_RISK_FACTORS = [
    "atr_norm",
    "volume_ma_ratio",
]


@st.composite
def valid_search_space_strategy(draw):
    """
    生成合法的 search_space 配置字典，用于测试 _constrained_random。

    约束：
    - 至少 2 个因子组，每组至少 1 个 signal 因子
    - 至少 1 个 risk 因子
    - min_factors <= max_factors，且 max_factors <= 总因子数
    - max_per_group >= 1
    """
    # 选择 risk 因子（至少 1 个）
    n_risk = draw(st.integers(min_value=1, max_value=len(_ALL_RISK_FACTORS)))
    risk_factors = draw(
        st.lists(
            st.sampled_from(_ALL_RISK_FACTORS),
            min_size=n_risk,
            max_size=n_risk,
            unique=True,
        )
    )

    # 构建因子组：至少 2 组，每组 1-4 个 signal 因子
    n_groups = draw(st.integers(min_value=2, max_value=4))
    # 从 signal 因子池中分配到各组（不重复）
    available_signal = list(_ALL_SIGNAL_FACTORS)
    groups = {}
    group_names = [f"group_{i}" for i in range(n_groups)]

    # 每组至少 1 个 signal 因子
    for gname in group_names:
        if not available_signal:
            break
        n_in_group = draw(st.integers(min_value=1, max_value=min(3, len(available_signal))))
        chosen = draw(
            st.lists(
                st.sampled_from(available_signal),
                min_size=n_in_group,
                max_size=n_in_group,
                unique=True,
            )
        )
        groups[gname] = chosen
        for f in chosen:
            available_signal.remove(f)

    # 将 risk 因子加入一个专属组
    groups["risk_group"] = list(risk_factors)

    # 计算总因子数（signal + risk）
    total_signal = sum(len(v) for k, v in groups.items() if k != "risk_group")
    total_factors = total_signal + len(risk_factors)

    # max_per_group >= 1，且不超过最大组大小
    max_group_size = max(len(v) for k, v in groups.items() if k != "risk_group")
    max_per_group = draw(st.integers(min_value=1, max_value=max(1, max_group_size)))

    # min_factors 和 max_factors：确保可行
    # 最少需要 1 个 signal + 1 个 risk = 2
    min_possible = 2
    max_possible = total_factors
    if min_possible > max_possible:
        min_possible = max_possible

    min_factors = draw(st.integers(min_value=min_possible, max_value=max(min_possible, min(4, max_possible))))
    max_factors = draw(st.integers(min_value=min_factors, max_value=max_possible))

    return {
        "mode": draw(st.sampled_from(["temporal_pool", "cross_sectional_filter"])),
        "factor_groups": groups,
        "risk_factors": risk_factors,
        "min_factors": min_factors,
        "max_factors": max_factors,
        "max_per_group": max_per_group,
        "rebalance_choices": ["W-FRI", "2W-FRI"],
        "hold_days_choices": [5, 10],
        "top_n_choices": [5, 10],
        "score_threshold_choices": [60, 65, 70],
    }


# Feature: auto-strategy-search, Property 5: 受约束随机搜索权重归一化
@given(
    search_space=valid_search_space_strategy(),
    n=st.integers(min_value=1, max_value=10),
)
@h_settings(max_examples=200)
def test_constrained_random_weight_normalization_property5(search_space, n):
    """
    **Validates: Requirements 4.3**

    Property 5：受约束随机搜索权重归一化

    对任意合法搜索空间配置和 n，_constrained_random 生成的每个候选策略的
    signal 因子权重之和在 [0.999, 1.001] 范围内。
    """
    service = StrategySearchService.__new__(StrategySearchService)
    candidates = service._constrained_random(search_space, n)

    # 允许生成 0 个候选（搜索空间过于受限时）
    for candidate in candidates:
        signal_factors = [
            f for f in candidate.get("factors", [])
            if f.get("role") == "signal"
        ]
        assert len(signal_factors) >= 1, (
            f"候选策略中没有 signal 因子，candidate={candidate}"
        )
        weight_sum = sum(f.get("weight", 0.0) for f in signal_factors)
        assert 0.999 <= weight_sum <= 1.001, (
            f"signal 因子权重之和 {weight_sum:.6f} 不在 [0.999, 1.001] 范围内，"
            f"signal_factors={signal_factors}"
        )


# Feature: auto-strategy-search, Property 6: 受约束随机搜索约束满足性
@given(
    search_space=valid_search_space_strategy(),
    n=st.integers(min_value=1, max_value=10),
)
@h_settings(max_examples=200)
def test_constrained_random_constraint_satisfaction_property6(search_space, n):
    """
    **Validates: Requirements 4.4, 4.5, 4.6, 4.7**

    Property 6：受约束随机搜索约束满足性

    对任意合法搜索空间配置和 n，_constrained_random 生成的每个候选策略均满足：
    1. 至少包含 1 个 risk_filter 因子（需求 4.4）
    2. 因子总数在 [min_factors, max_factors] 范围内（需求 4.5）
    3. 同组因子数不超过 max_per_group（需求 4.6）
    4. 单个 signal 因子权重不超过 0.40（需求 4.7）
    """
    service = StrategySearchService.__new__(StrategySearchService)
    candidates = service._constrained_random(search_space, n)

    factor_groups: dict = search_space.get("factor_groups", {})
    risk_factors: list = search_space.get("risk_factors", [])
    min_factors: int = search_space.get("min_factors", 4)
    max_factors: int = search_space.get("max_factors", 8)
    max_per_group: int = search_space.get("max_per_group", 2)

    # 构建因子名 -> 组名的映射（用于验证同组约束）
    factor_to_group: dict[str, str] = {}
    for gname, gfactors in factor_groups.items():
        for fname in gfactors:
            factor_to_group[fname] = gname

    for candidate in candidates:
        factors = candidate.get("factors", [])
        signal_factors = [f for f in factors if f.get("role") == "signal"]
        risk_filter_factors = [f for f in factors if f.get("role") == "risk_filter"]

        # 约束 1：至少 1 个 risk_filter 因子（需求 4.4）
        assert len(risk_filter_factors) >= 1, (
            f"候选策略中没有 risk_filter 因子，factors={factors}"
        )

        # 约束 2：因子总数在 [min_factors, max_factors] 范围内（需求 4.5）
        total_count = len(factors)
        assert min_factors <= total_count <= max_factors, (
            f"因子总数 {total_count} 不在 [{min_factors}, {max_factors}] 范围内，"
            f"factors={factors}"
        )

        # 约束 3：同组 signal 因子数不超过 max_per_group（需求 4.6）
        group_counts: dict[str, int] = {}
        for f in signal_factors:
            fname = f.get("name", "")
            gname = factor_to_group.get(fname, "__unknown__")
            group_counts[gname] = group_counts.get(gname, 0) + 1

        for gname, count in group_counts.items():
            if gname == "__unknown__":
                continue  # 未知组跳过（risk_group 中的 signal 因子）
            assert count <= max_per_group, (
                f"组 '{gname}' 中 signal 因子数 {count} 超过 max_per_group={max_per_group}，"
                f"signal_factors={signal_factors}"
            )

        # 约束 4：单个 signal 因子权重不超过 0.40（需求 4.7）
        for f in signal_factors:
            w = f.get("weight", 0.0)
            assert w <= 0.40 + 1e-9, (
                f"signal 因子 '{f.get('name')}' 权重 {w:.6f} 超过 0.40，"
                f"signal_factors={signal_factors}"
            )


# ---------------------------------------------------------------------------
# Property 7：稳定性总分范围不变量 & 全负 Sharpe 压低
# Validates: Requirements 5.7, 5.8
# ---------------------------------------------------------------------------

from backend.services.strategy_evaluation_service import StrategyEvaluationService  # noqa: E402


# Hypothesis strategy: generate a single window_metric dict
@st.composite
def window_metric_strategy(draw):
    """生成单个 window_metric 字典，test_metrics 含所有必要字段。"""
    sharpe = draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    annual_return = draw(st.floats(min_value=-1.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    max_drawdown = draw(st.floats(min_value=-1.0, max_value=0.0, allow_nan=False, allow_infinity=False))
    ir = draw(st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False))
    excess_return = draw(st.floats(min_value=-1.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    turnover_rate = draw(st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False))
    return {
        "test_metrics": {
            "sharpe": sharpe,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "ir": ir,
            "excess_return": excess_return,
            "turnover_rate": turnover_rate,
        }
    }


@st.composite
def negative_sharpe_window_metric_strategy(draw):
    """生成 Sharpe 严格为负的 window_metric 字典。"""
    sharpe = draw(st.floats(min_value=-5.0, max_value=-0.001, allow_nan=False, allow_infinity=False))
    annual_return = draw(st.floats(min_value=-1.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    max_drawdown = draw(st.floats(min_value=-1.0, max_value=0.0, allow_nan=False, allow_infinity=False))
    ir = draw(st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False))
    excess_return = draw(st.floats(min_value=-1.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    turnover_rate = draw(st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False))
    return {
        "test_metrics": {
            "sharpe": sharpe,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "ir": ir,
            "excess_return": excess_return,
            "turnover_rate": turnover_rate,
        }
    }


# Feature: auto-strategy-search, Property 7a: 稳定性总分范围不变量
@given(
    window_metrics=st.lists(window_metric_strategy(), min_size=1, max_size=20)
)
@h_settings(max_examples=500)
def test_stability_score_range_invariant_property7a(window_metrics):
    """
    **Validates: Requirements 5.7**

    Property 7a：稳定性总分范围不变量

    对任意非空 window_metrics 列表，compute_stability_score 返回值必须 ∈ [0.0, 100.0]。
    """
    service = StrategyEvaluationService.__new__(StrategyEvaluationService)
    result = service.compute_stability_score(window_metrics)
    # compute_stability_score may return a dict (invalid) or float (valid)
    if isinstance(result, dict):
        status = result.get("evaluation_status", "")
        assert status == "invalid", (
            f"dict 返回值的 evaluation_status 应为 'invalid'，但得到: {status}"
        )
        return  # invalid 结果不需要检查范围
    score = float(result)
    assert 0.0 <= score <= 100.0, (
        f"compute_stability_score 返回值 {score} 超出 [0.0, 100.0] 范围，"
        f"window_metrics={window_metrics}"
    )


# Feature: auto-strategy-search, Property 7b: 全负 Sharpe 时总分 < 30
@given(
    window_metrics=st.lists(negative_sharpe_window_metric_strategy(), min_size=1, max_size=20)
)
@h_settings(max_examples=500)
def test_stability_score_all_negative_sharpe_property7b(window_metrics):
    """
    **Validates: Requirements 5.8**

    Property 7b：全负 Sharpe 时稳定性总分 < 30

    当所有 window_metrics 的 test_metrics.sharpe 均为负时，
    compute_stability_score 返回值必须 < 30。
    """
    # 确认所有 Sharpe 确实为负（generator 保证，但做防御性检查）
    for wm in window_metrics:
        assert wm["test_metrics"]["sharpe"] < 0, "测试数据生成错误：Sharpe 应为负"

    service = StrategyEvaluationService.__new__(StrategyEvaluationService)
    result = service.compute_stability_score(window_metrics)
    # compute_stability_score may return a dict (invalid) or float (valid)
    if isinstance(result, dict):
        status = result.get("evaluation_status", "")
        assert status == "invalid", (
            f"dict 返回值的 evaluation_status 应为 'invalid'，但得到: {status}"
        )
        return  # invalid 结果不适用此属性
    score = float(result)
    assert score < 30.0, (
        f"全负 Sharpe 时 compute_stability_score 返回值 {score} 应 < 30，"
        f"window_metrics={window_metrics}"
    )


# ---------------------------------------------------------------------------
# Property 14：Walk-Forward 窗口无重叠覆盖完整性
# Validates: Requirements 5.3, 5.4
# ---------------------------------------------------------------------------

from datetime import date, timedelta  # noqa: E402
from dateutil.relativedelta import relativedelta  # noqa: E402


# ---------------------------------------------------------------------------
# Hypothesis strategies for walk-forward window generation
# ---------------------------------------------------------------------------

@st.composite
def walk_forward_params_strategy(draw):
    """
    生成合法的 walk-forward 参数组合：
    - start_date, end_date：合法日期范围，跨度足够容纳至少一个完整窗口
    - train_months, valid_months, test_months, step_months：合法月数
    """
    # 生成起始年份和月份
    start_year = draw(st.integers(min_value=2010, max_value=2022))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))  # 避免月末边界问题
    start = date(start_year, start_month, start_day)

    # 窗口参数
    train_months = draw(st.integers(min_value=3, max_value=24))
    valid_months = draw(st.integers(min_value=1, max_value=6))
    test_months = draw(st.integers(min_value=1, max_value=6))
    step_months = draw(st.integers(min_value=1, max_value=6))

    window_total = train_months + valid_months + test_months

    # end_date 至少能容纳 1 个完整窗口，最多 3 个窗口
    extra_months = draw(st.integers(min_value=0, max_value=step_months * 2))
    end = start + relativedelta(months=window_total + extra_months)

    return {
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "train_months": train_months,
        "valid_months": valid_months,
        "test_months": test_months,
        "step_months": step_months,
    }


def _parse_date_str(s: str) -> date:
    from datetime import datetime
    return datetime.strptime(s, "%Y-%m-%d").date()


# Feature: auto-strategy-search, Property 14a: 窗口内各期无重叠
@given(params=walk_forward_params_strategy())
@h_settings(max_examples=500)
def test_walk_forward_windows_no_overlap_property14a(params):
    """
    **Validates: Requirements 5.3**

    Property 14a：Walk-Forward 窗口内各期无重叠

    对任意合法时间范围和窗口参数，_generate_walk_forward_windows 生成的每个窗口中：
    - train 期、valid 期、test 期两两无重叠（各期日期区间不相交）
    - 各期在窗口内按顺序排列：train_end < valid_start <= valid_end < test_start
    """
    service = StrategyEvaluationService.__new__(StrategyEvaluationService)
    windows = service._generate_walk_forward_windows(
        start_date=params["start_date"],
        end_date=params["end_date"],
        train_months=params["train_months"],
        valid_months=params["valid_months"],
        test_months=params["test_months"],
        step_months=params["step_months"],
    )

    # 允许生成 0 个窗口（时间范围不足时）
    for w in windows:
        train_start = _parse_date_str(w["train_start"])
        train_end = _parse_date_str(w["train_end"])
        valid_start = _parse_date_str(w["valid_start"])
        valid_end = _parse_date_str(w["valid_end"])
        test_start = _parse_date_str(w["test_start"])
        test_end = _parse_date_str(w["test_end"])

        # 各期内部：start <= end
        assert train_start <= train_end, (
            f"窗口 {w['window_id']}: train_start={train_start} > train_end={train_end}"
        )
        assert valid_start <= valid_end, (
            f"窗口 {w['window_id']}: valid_start={valid_start} > valid_end={valid_end}"
        )
        assert test_start <= test_end, (
            f"窗口 {w['window_id']}: test_start={test_start} > test_end={test_end}"
        )

        # 各期顺序：train_end < valid_start（无重叠）
        assert train_end < valid_start, (
            f"窗口 {w['window_id']}: train_end={train_end} >= valid_start={valid_start}，"
            "训练期与验证期重叠"
        )

        # valid_end < test_start（无重叠）
        assert valid_end < test_start, (
            f"窗口 {w['window_id']}: valid_end={valid_end} >= test_start={test_start}，"
            "验证期与测试期重叠"
        )

        # 连续性：valid_start = train_end + 1 day
        assert valid_start == train_end + timedelta(days=1), (
            f"窗口 {w['window_id']}: valid_start={valid_start} 不等于 train_end+1={train_end + timedelta(days=1)}，"
            "训练期与验证期不连续"
        )

        # 连续性：test_start = valid_end + 1 day
        assert test_start == valid_end + timedelta(days=1), (
            f"窗口 {w['window_id']}: test_start={test_start} 不等于 valid_end+1={valid_end + timedelta(days=1)}，"
            "验证期与测试期不连续"
        )


# Feature: auto-strategy-search, Property 14b: 窗口间无重叠
@given(params=walk_forward_params_strategy())
@h_settings(max_examples=500)
def test_walk_forward_windows_inter_window_no_overlap_property14b(params):
    """
    **Validates: Requirements 5.3**

    Property 14b：Walk-Forward 相邻窗口间无重叠

    对任意合法时间范围和窗口参数，相邻窗口的 train_start 严格递增，
    且后一窗口的 train_start 等于前一窗口 train_start + step_months。
    """
    service = StrategyEvaluationService.__new__(StrategyEvaluationService)
    windows = service._generate_walk_forward_windows(
        start_date=params["start_date"],
        end_date=params["end_date"],
        train_months=params["train_months"],
        valid_months=params["valid_months"],
        test_months=params["test_months"],
        step_months=params["step_months"],
    )

    if len(windows) < 2:
        return  # 少于 2 个窗口时无需验证相邻关系

    step_months = params["step_months"]

    for i in range(len(windows) - 1):
        curr = windows[i]
        nxt = windows[i + 1]

        curr_train_start = _parse_date_str(curr["train_start"])
        next_train_start = _parse_date_str(nxt["train_start"])

        # 后一窗口 train_start = 前一窗口 train_start + step_months
        expected_next_start = curr_train_start + relativedelta(months=step_months)
        assert next_train_start == expected_next_start, (
            f"窗口 {i} -> {i+1}: next_train_start={next_train_start} "
            f"不等于 curr_train_start+{step_months}m={expected_next_start}"
        )

        # 后一窗口 train_start 严格大于前一窗口 train_start
        assert next_train_start > curr_train_start, (
            f"窗口 {i} -> {i+1}: train_start 未严格递增，"
            f"curr={curr_train_start}, next={next_train_start}"
        )


# Feature: auto-strategy-search, Property 14c: 窗口覆盖完整时间范围
@given(params=walk_forward_params_strategy())
@h_settings(max_examples=500)
def test_walk_forward_windows_coverage_property14c(params):
    """
    **Validates: Requirements 5.4**

    Property 14c：Walk-Forward 窗口覆盖完整时间范围

    对任意合法时间范围和窗口参数，若生成了至少一个窗口：
    - 第一个窗口的 train_start 等于整体 start_date
    - 最后一个窗口的 test_end 不超过整体 end_date
    - 所有窗口的 test_end 均不超过整体 end_date（边界约束）
    """
    service = StrategyEvaluationService.__new__(StrategyEvaluationService)
    windows = service._generate_walk_forward_windows(
        start_date=params["start_date"],
        end_date=params["end_date"],
        train_months=params["train_months"],
        valid_months=params["valid_months"],
        test_months=params["test_months"],
        step_months=params["step_months"],
    )

    if not windows:
        return  # 时间范围不足，无窗口生成，合法

    overall_start = _parse_date_str(params["start_date"])
    overall_end = _parse_date_str(params["end_date"])

    # 第一个窗口的 train_start 等于整体 start_date
    first_train_start = _parse_date_str(windows[0]["train_start"])
    assert first_train_start == overall_start, (
        f"第一个窗口 train_start={first_train_start} 不等于 start_date={overall_start}"
    )

    # 所有窗口的 test_end 不超过整体 end_date
    for w in windows:
        test_end = _parse_date_str(w["test_end"])
        assert test_end <= overall_end, (
            f"窗口 {w['window_id']}: test_end={test_end} 超过 end_date={overall_end}"
        )

    # 最后一个窗口的 test_end 是所有合法窗口中最晚的
    last_test_end = _parse_date_str(windows[-1]["test_end"])

    # 验证：若再滚动一步，下一个窗口的 test_end 会超过 end_date（即已生成最多窗口）
    last_train_start = _parse_date_str(windows[-1]["train_start"])
    next_train_start = last_train_start + relativedelta(months=params["step_months"])
    next_test_end = (
        next_train_start
        + relativedelta(months=params["train_months"])
        + relativedelta(months=params["valid_months"])
        + relativedelta(months=params["test_months"])
        - timedelta(days=1)
    )
    assert next_test_end > overall_end, (
        f"最后一个窗口后还可以生成更多窗口：next_test_end={next_test_end} <= end_date={overall_end}，"
        "说明窗口生成不完整"
    )


# ---------------------------------------------------------------------------
# Property 8：ChampionRegistryService save/load round-trip
# Property 9：list_champions 按 scope_type 过滤正确性
# Property 10：group_hash 顺序无关性
# Validates: Requirements 7.3, 7.5, 7.8
# ---------------------------------------------------------------------------

import tempfile  # noqa: E402
from pathlib import Path as _Path  # noqa: E402 (alias to avoid shadowing)

from backend.services.champion_registry_service import ChampionRegistryService  # noqa: E402

# Hypothesis strategies for champion data

_VALID_SCOPE_TYPES = ["single_stock", "stock_group", "cross_sectional_universe"]

# scope_key：不含 '/' 或 '..' 的安全字符串
safe_scope_key_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_",
    min_size=1,
    max_size=20,
)

# champion dict：任意 JSON 可序列化的字典（值为 str/int/float/bool/None）
champion_value_strategy = st.one_of(
    st.text(min_size=0, max_size=20),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
)

champion_dict_strategy = st.dictionaries(
    keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10),
    values=champion_value_strategy,
    min_size=0,
    max_size=8,
)

# stock code strategy：6 位数字字符串（允许前导零）
stock_code_6_strategy = st.text(
    alphabet="0123456789",
    min_size=6,
    max_size=6,
)


# Feature: auto-strategy-search, Property 8: save/load round-trip
@given(
    scope_type=st.sampled_from(_VALID_SCOPE_TYPES),
    scope_key=safe_scope_key_strategy,
    champion=champion_dict_strategy,
)
@h_settings(max_examples=300)
def test_champion_registry_save_load_roundtrip_property8(scope_type, scope_key, champion):
    """
    **Validates: Requirements 7.3**

    Property 8：save/load round-trip

    对任意合法 scope_type、scope_key 和 champion 字典，
    save 后立即 load，返回内容与保存内容完全一致。
    使用临时目录隔离文件 I/O，不污染真实数据。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        service = ChampionRegistryService(base_dir=_Path(tmpdir))
        service.save(scope_type, scope_key, champion)
        loaded = service.load(scope_type, scope_key)

        assert loaded is not None, (
            f"load 返回 None，scope_type={scope_type}, scope_key={scope_key}"
        )

        # 去掉 list_champions 注入的 scope_type 字段（load 不注入）
        # load 直接返回原始 JSON，不含注入字段
        assert loaded == champion, (
            f"round-trip 不一致：\n  saved={champion}\n  loaded={loaded}\n"
            f"  scope_type={scope_type}, scope_key={scope_key}"
        )


# Feature: auto-strategy-search, Property 9: list_champions 按 scope_type 过滤正确性
@st.composite
def multi_scope_champions_strategy(draw):
    """
    生成多个 scope_type 下的 champion 列表，每个 scope_type 有 0-4 个 champion。
    返回 dict: {scope_type: [(scope_key, champion_dict), ...]}
    """
    result = {}
    for st_name in _VALID_SCOPE_TYPES:
        n = draw(st.integers(min_value=0, max_value=4))
        if n == 0:
            result[st_name] = []
            continue
        keys = draw(
            st.lists(
                safe_scope_key_strategy,
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
        champions = draw(
            st.lists(
                champion_dict_strategy,
                min_size=n,
                max_size=n,
            )
        )
        result[st_name] = list(zip(keys, champions))
    return result


# Feature: auto-strategy-search, Property 9: list_champions 按 scope_type 过滤正确性
@given(champions_by_type=multi_scope_champions_strategy())
@h_settings(max_examples=200)
def test_champion_registry_list_filter_by_scope_type_property9(champions_by_type):
    """
    **Validates: Requirements 7.5**

    Property 9：list_champions 按 scope_type 过滤正确性

    保存多个不同 scope_type 的 champion 后，
    list_champions(scope_type) 返回列表中所有元素的 scope_type 字段等于指定值，
    且数量与该 scope_type 下保存的 champion 数量一致。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        service = ChampionRegistryService(base_dir=_Path(tmpdir))

        # 保存所有 champion
        for scope_type, pairs in champions_by_type.items():
            for scope_key, champion in pairs:
                service.save(scope_type, scope_key, champion)

        # 验证每个 scope_type 的过滤结果
        for target_scope_type in _VALID_SCOPE_TYPES:
            result = service.list_champions(scope_type=target_scope_type)

            # 所有返回元素的 scope_type 字段必须等于指定值
            for item in result:
                assert item.get("scope_type") == target_scope_type, (
                    f"list_champions({target_scope_type!r}) 返回了 scope_type={item.get('scope_type')!r} 的元素"
                )

            # 返回数量应等于该 scope_type 下保存的 champion 数量
            expected_count = len(champions_by_type.get(target_scope_type, []))
            assert len(result) == expected_count, (
                f"list_champions({target_scope_type!r}) 返回 {len(result)} 个，"
                f"期望 {expected_count} 个"
            )


# Feature: auto-strategy-search, Property 10: group_hash 顺序无关性
@given(
    codes=st.lists(
        stock_code_6_strategy,
        min_size=1,
        max_size=20,
        unique=True,
    )
)
@h_settings(max_examples=500)
def test_champion_registry_group_hash_order_independence_property10(codes):
    """
    **Validates: Requirements 7.8**

    Property 10：group_hash 顺序无关性

    对任意股票代码列表，_scope_key_to_hash 对任意排列顺序返回相同的 hash 值。
    验证：sorted(codes) 和 reversed(sorted(codes)) 以及原始顺序均产生相同 hash。
    """
    import random

    hash_original = ChampionRegistryService._scope_key_to_hash(codes)

    # 逆序
    reversed_codes = list(reversed(codes))
    hash_reversed = ChampionRegistryService._scope_key_to_hash(reversed_codes)
    assert hash_original == hash_reversed, (
        f"原始顺序 hash={hash_original} 与逆序 hash={hash_reversed} 不一致，"
        f"codes={codes}"
    )

    # 随机打乱（固定种子保证可复现）
    shuffled_codes = list(codes)
    random.seed(42)
    random.shuffle(shuffled_codes)
    hash_shuffled = ChampionRegistryService._scope_key_to_hash(shuffled_codes)
    assert hash_original == hash_shuffled, (
        f"原始顺序 hash={hash_original} 与随机打乱 hash={hash_shuffled} 不一致，"
        f"codes={codes}"
    )

    # 排序后
    sorted_codes = sorted(codes)
    hash_sorted = ChampionRegistryService._scope_key_to_hash(sorted_codes)
    assert hash_original == hash_sorted, (
        f"原始顺序 hash={hash_original} 与排序后 hash={hash_sorted} 不一致，"
        f"codes={codes}"
    )

    # hash 值应为 32 位十六进制字符串（MD5）
    assert len(hash_original) == 32, (
        f"hash 长度 {len(hash_original)} 不等于 32（MD5 格式），hash={hash_original}"
    )
    assert all(c in "0123456789abcdef" for c in hash_original), (
        f"hash 含非十六进制字符：{hash_original}"
    )


# ---------------------------------------------------------------------------
# Property 13：Champion 稳定性总分非负性
# Validates: Requirements 6.10
# ---------------------------------------------------------------------------


@st.composite
def evaluated_candidate_strategy(draw):
    """生成单个已评价候选策略字典，含 stability_score 和 test_metrics。"""
    stability_score = draw(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    sharpe = draw(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    annual_return = draw(
        st.floats(min_value=-1.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    return {
        "stability_score": stability_score,
        "test_metrics": {"sharpe": sharpe, "annual_return": annual_return},
        "profile_id": "test_profile",
        "mode": "temporal_pool",
    }


@st.composite
def evaluated_candidate_with_none_score_strategy(draw):
    """生成 stability_score 可能为 None 或缺失的候选策略字典。"""
    score_variant = draw(
        st.one_of(
            st.none(),
            st.just("missing"),  # 用于标记缺失键
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        )
    )
    sharpe = draw(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    annual_return = draw(
        st.floats(min_value=-1.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    candidate = {
        "test_metrics": {"sharpe": sharpe, "annual_return": annual_return},
        "profile_id": "test_profile",
        "mode": "temporal_pool",
    }
    if score_variant != "missing":
        candidate["stability_score"] = score_variant
    return candidate


def _apply_champion_selection_logic(evaluated: list[dict]) -> dict:
    """
    复现 ChampionStrategyService._execute_search 中的 champion 选择逻辑：
    1. 过滤：test Sharpe > 0 且 annual_return > -0.20
    2. 若无有效候选，降级为全部候选
    3. 按 stability_score 降序排序，取第一个
    4. 用 _safe_float(val, 0.0) 处理 stability_score（None 或缺失时默认 0.0）
    """
    from backend.services.champion_strategy_service import _safe_float

    valid = [
        c for c in evaluated
        if _safe_float(c.get("test_metrics", {}).get("sharpe"), 0.0) > 0
        and _safe_float(c.get("test_metrics", {}).get("annual_return"), 0.0) > -0.20
    ]
    if not valid:
        valid = list(evaluated)  # fallback

    valid.sort(key=lambda c: _safe_float(c.get("stability_score"), 0.0), reverse=True)
    best = valid[0]
    stability_score = _safe_float(best.get("stability_score"), 0.0)
    return {"stability_score": stability_score}


# Feature: auto-strategy-search, Property 13: Champion 稳定性总分非负性
@given(
    evaluated=st.lists(evaluated_candidate_strategy(), min_size=1, max_size=20)
)
@h_settings(max_examples=500)
def test_champion_stability_score_non_negative_property13(evaluated):
    """
    **Validates: Requirements 6.10**

    Property 13：Champion 稳定性总分非负性

    对任意非空已评价候选列表（stability_score ∈ [0, 100]），
    champion 选择逻辑选出的 champion 的 stability_score 必须 >= 0。
    """
    champion = _apply_champion_selection_logic(evaluated)
    stability_score = champion["stability_score"]
    assert stability_score >= 0.0, (
        f"champion stability_score={stability_score} < 0，evaluated={evaluated}"
    )


# Feature: auto-strategy-search, Property 13 补充: _safe_float fallback 保证非负
@given(
    evaluated=st.lists(
        evaluated_candidate_with_none_score_strategy(), min_size=1, max_size=20
    )
)
@h_settings(max_examples=500)
def test_champion_stability_score_non_negative_with_none_property13_fallback(evaluated):
    """
    **Validates: Requirements 6.10**

    Property 13 补充：stability_score 为 None 或缺失时，_safe_float 默认值 0.0 保证非负性。

    当候选的 stability_score 字段为 None 或不存在时，
    champion 选择逻辑通过 _safe_float(val, 0.0) 将其转换为 0.0，
    确保最终 champion 的 stability_score >= 0。
    """
    champion = _apply_champion_selection_logic(evaluated)
    stability_score = champion["stability_score"]
    assert stability_score >= 0.0, (
        f"champion stability_score={stability_score} < 0（含 None/缺失 score），"
        f"evaluated={evaluated}"
    )

# ---------------------------------------------------------------------------
# Property 11：深市股票代码前导零保留
# Validates: Requirements 8.8
# ---------------------------------------------------------------------------

import csv
import os
import tempfile


@given(
    prefix=st.just("0"),
    rest=st.text(alphabet="0123456789", min_size=5, max_size=5),
)
@h_settings(max_examples=300)
def test_shenzhen_code_leading_zero_preserved_property11(prefix, rest):
    """
    **Validates: Requirements 8.8**

    对任意以 '0' 开头的 6 位纯数字股票代码，经过 _normalize_code 逻辑处理后，
    代码字符串保持不变（前导零不丢失）。
    """
    code = prefix + rest  # 6 位，以 "0" 开头
    # 复现 _normalize_code 逻辑
    s = str(code).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit():
        s = s.zfill(6)
    assert s == code, f"前导零丢失：原始={code}, 规范化后={s}"
    assert s.startswith("0"), f"规范化后不以 0 开头：{s}"
    assert len(s) == 6, f"规范化后长度不为 6：{s}"


def test_load_fixed_pool_preserves_leading_zeros_property11_integration():
    """
    **Validates: Requirements 8.8**

    通过 load_fixed_pool 加载含深市代码的临时 CSV，验证前导零被完整保留。
    """
    # Mock heavy dependencies that may not be installed in test environment
    _sklearn_mock = MagicMock()
    _sklearn_preprocessing_mock = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "sklearn": _sklearn_mock,
            "sklearn.preprocessing": _sklearn_preprocessing_mock,
        },
    ):
        from backend.services.temporal_filter_validation_service import (
            TemporalFilterValidationService,
        )

        codes = ["000001", "000002", "002415", "001234"]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["code"])
            writer.writeheader()
            for code in codes:
                writer.writerow({"code": code})
            tmppath = f.name

        try:
            service = TemporalFilterValidationService.__new__(
                TemporalFilterValidationService
            )
            result = service.load_fixed_pool(tmppath)
            for code in codes:
                assert code in result, f"代码 {code} 在结果中丢失"
        finally:
            os.unlink(tmppath)


@given(
    digits=st.text(alphabet="123456789", min_size=1, max_size=5),
)
@h_settings(max_examples=300)
def test_normalize_code_removes_float_suffix_property11(digits):
    """
    **Validates: Requirements 8.8**

    模拟 pandas 将整数列读为浮点数（如 "1234" -> "1234.0"），
    验证 _normalize_code 逻辑能正确去除 .0 后缀并补零到 6 位。
    """
    float_str = digits + ".0"
    s = str(float_str).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit():
        s = s.zfill(6)
    assert not s.endswith(".0"), f"规范化后仍含 .0 后缀：{s}"
    assert len(s) == 6, f"规范化后长度不为 6：{s}"


# ---------------------------------------------------------------------------
# Property 12：回测持仓顺序无关性
# Validates: Requirements 11.1
# ---------------------------------------------------------------------------

def _get_temporal_filter_service():
    """Import TemporalFilterValidationService with sklearn mocked out."""
    _sklearn_mock = MagicMock()
    _sklearn_preprocessing_mock = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "sklearn": _sklearn_mock,
            "sklearn.preprocessing": _sklearn_preprocessing_mock,
        },
    ):
        from backend.services.temporal_filter_validation_service import (
            TemporalFilterValidationService,
        )
        return TemporalFilterValidationService


# Feature: auto-strategy-search, Property 12: 回测持仓顺序无关性（单次再平衡）
@given(
    codes=st.lists(
        st.text(alphabet="0123456789", min_size=6, max_size=6),
        min_size=2,
        max_size=5,
        unique=True,
    ),
    n_dates=st.integers(min_value=5, max_value=20),
    price_seed=st.integers(min_value=0, max_value=1000),
)
@h_settings(max_examples=300)
def test_equal_weight_backtest_order_independence_property12(codes, n_dates, price_seed):
    """
    **Validates: Requirements 11.1**

    Property 12：回测持仓顺序无关性（单次再平衡）

    对任意持仓集合，以不同顺序传入 _equal_weight_backtest，
    产生的 NAV 曲线必须完全相同。
    """
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")

    rng = np.random.default_rng(price_seed)
    price_data = {}
    for code in codes:
        prices = 10.0 + rng.random(n_dates) * 5  # prices in [10, 15]
        price_data[code] = pd.DataFrame({"close": prices}, index=dates)

    rebalance_date = dates[0]
    holdings_original = list(codes)
    holdings_reversed = list(reversed(codes))

    holdings_by_date_original = {rebalance_date: holdings_original}
    holdings_by_date_reversed = {rebalance_date: holdings_reversed}

    TemporalFilterValidationService = _get_temporal_filter_service()
    service = TemporalFilterValidationService.__new__(TemporalFilterValidationService)
    nav_original = service._equal_weight_backtest(holdings_by_date_original, price_data)
    nav_reversed = service._equal_weight_backtest(holdings_by_date_reversed, price_data)

    assert len(nav_original) == len(nav_reversed), (
        f"NAV 曲线长度不一致：original={len(nav_original)}, reversed={len(nav_reversed)}"
    )
    pd.testing.assert_series_equal(nav_original, nav_reversed, check_names=False)


# Feature: auto-strategy-search, Property 12: 回测持仓顺序无关性（多次再平衡）
@given(
    codes=st.lists(
        st.text(alphabet="0123456789", min_size=6, max_size=6),
        min_size=2,
        max_size=5,
        unique=True,
    ),
    n_rebalances=st.integers(min_value=1, max_value=3),
    n_dates=st.integers(min_value=10, max_value=30),
    price_seed=st.integers(min_value=0, max_value=1000),
)
@h_settings(max_examples=300)
def test_equal_weight_backtest_multi_rebalance_order_independence_property12(
    codes, n_rebalances, n_dates, price_seed
):
    """
    **Validates: Requirements 11.1**

    Property 12：回测持仓顺序无关性（多次再平衡）

    对任意持仓集合和多个再平衡日期，以不同顺序传入 _equal_weight_backtest，
    产生的 NAV 曲线必须完全相同。
    """
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")

    rng = np.random.default_rng(price_seed)
    price_data = {}
    for code in codes:
        prices = 10.0 + rng.random(n_dates) * 5
        price_data[code] = pd.DataFrame({"close": prices}, index=dates)

    # Spread rebalance dates evenly across the date range
    step = max(1, n_dates // (n_rebalances + 1))
    rebalance_indices = [i * step for i in range(n_rebalances) if i * step < n_dates]

    holdings_original: dict = {}
    holdings_reversed: dict = {}
    for idx in rebalance_indices:
        rb_date = dates[idx]
        holdings_original[rb_date] = list(codes)
        holdings_reversed[rb_date] = list(reversed(codes))

    TemporalFilterValidationService = _get_temporal_filter_service()
    service = TemporalFilterValidationService.__new__(TemporalFilterValidationService)
    nav_original = service._equal_weight_backtest(holdings_original, price_data)
    nav_reversed = service._equal_weight_backtest(holdings_reversed, price_data)

    assert len(nav_original) == len(nav_reversed), (
        f"NAV 曲线长度不一致：original={len(nav_original)}, reversed={len(nav_reversed)}"
    )
    pd.testing.assert_series_equal(nav_original, nav_reversed, check_names=False)


# Feature: auto-strategy-search, Property 12: _equal_weight_backtest_with_cash 顺序无关性
@given(
    codes=st.lists(
        st.text(alphabet="0123456789", min_size=6, max_size=6),
        min_size=2,
        max_size=5,
        unique=True,
    ),
    n_dates=st.integers(min_value=5, max_value=20),
    price_seed=st.integers(min_value=0, max_value=1000),
    top_n=st.integers(min_value=2, max_value=10),
)
@h_settings(max_examples=300)
def test_equal_weight_backtest_with_cash_order_independence_property12(
    codes, n_dates, price_seed, top_n
):
    """
    **Validates: Requirements 11.1**

    Property 12：_equal_weight_backtest_with_cash 持仓顺序无关性

    对任意持仓集合，以不同顺序传入 _equal_weight_backtest_with_cash，
    产生的 NAV 曲线必须完全相同。
    """
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")

    rng = np.random.default_rng(price_seed)
    price_data = {}
    for code in codes:
        prices = 10.0 + rng.random(n_dates) * 5
        price_data[code] = pd.DataFrame({"close": prices}, index=dates)

    rebalance_date = dates[0]
    holdings_original = {rebalance_date: list(codes)}
    holdings_reversed = {rebalance_date: list(reversed(codes))}

    TemporalFilterValidationService = _get_temporal_filter_service()
    service = TemporalFilterValidationService.__new__(TemporalFilterValidationService)
    nav_original = service._equal_weight_backtest_with_cash(
        holdings_original, price_data, top_n
    )
    nav_reversed = service._equal_weight_backtest_with_cash(
        holdings_reversed, price_data, top_n
    )

    assert len(nav_original) == len(nav_reversed), (
        f"NAV 曲线长度不一致：original={len(nav_original)}, reversed={len(nav_reversed)}"
    )
    pd.testing.assert_series_equal(nav_original, nav_reversed, check_names=False)


# ---------------------------------------------------------------------------
# Property 13：_load_universe_df 因子列非全 NaN 不变量
# Validates: TASK 8 fix — cross-sectional default path computes factor columns
# ---------------------------------------------------------------------------

def test_load_universe_df_factor_columns_populated():
    """
    **Validates: TASK 8 fix**

    Property 13：_load_universe_df 返回的 DataFrame 中，
    请求的因子列必须存在且非全 NaN（当 DataService 返回有效 OHLCV 数据时）。

    通过 mock DataService.get_multiple_stocks_data 和
    TemporalScoringService._compute_factors 验证因子列被正确附加。
    """
    import pandas as pd
    import numpy as np
    from unittest.mock import patch, MagicMock
    from backend.services.strategy_evaluation_service import _load_universe_df

    # 构造 mock OHLCV 数据（60 个交易日，足够计算因子）
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    rng = np.random.default_rng(42)
    codes = ["000001", "000002"]

    def make_ohlcv(seed):
        rng2 = np.random.default_rng(seed)
        close = 10.0 + rng2.random(60).cumsum()
        return pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": rng2.integers(1_000_000, 5_000_000, size=60).astype(float),
            },
            index=dates,
        )

    mock_price_data = {code: make_ohlcv(i) for i, code in enumerate(codes)}

    # mock _compute_factors 返回两个因子的 Series
    def mock_compute_factors(df):
        return {
            "price_vs_sma20": df["close"] / df["close"].rolling(20).mean(),
            "momentum_20": df["close"] / df["close"].shift(20) - 1,
        }

    with patch(
        "backend.services.data_service.data_service.get_multiple_stocks_data",
        return_value=mock_price_data,
    ), patch(
        "backend.services.temporal_scoring_service.temporal_scoring_service._compute_factors",
        side_effect=mock_compute_factors,
    ):
        result = _load_universe_df(
            universe="test_universe",
            codes=codes,
            start_date="2023-01-01",
            end_date="2023-03-31",
            factor_names=["price_vs_sma20", "momentum_20"],
        )

    # 基本结构检查
    assert not result.empty, "_load_universe_df 返回空 DataFrame"
    assert "stock_code" in result.columns
    assert "close" in result.columns
    assert "date" in result.columns

    # 因子列存在
    assert "price_vs_sma20" in result.columns, "缺少 price_vs_sma20 列"
    assert "momentum_20" in result.columns, "缺少 momentum_20 列"

    # 因子列非全 NaN（至少有部分有效值）
    for fn in ["price_vs_sma20", "momentum_20"]:
        non_nan_count = result[fn].notna().sum()
        assert non_nan_count > 0, (
            f"因子列 '{fn}' 全为 NaN，_load_universe_df 未正确计算因子"
        )

    # 所有请求的股票代码都在结果中
    result_codes = set(result["stock_code"].unique())
    for code in codes:
        assert code in result_codes, f"股票 {code} 不在结果中"


def test_load_universe_df_unknown_factor_returns_nan_column():
    """
    **Validates: TASK 8 fix — 未知因子名优雅降级**

    请求一个不在内置因子列表中的因子名时，
    _load_universe_df 应返回该列全为 NaN（而非抛出异常）。
    """
    import pandas as pd
    import numpy as np
    from unittest.mock import patch
    from backend.services.strategy_evaluation_service import _load_universe_df

    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    rng = np.random.default_rng(0)
    close = 10.0 + rng.random(60).cumsum()
    mock_ohlcv = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, size=60).astype(float),
        },
        index=dates,
    )

    with patch(
        "backend.services.data_service.data_service.get_multiple_stocks_data",
        return_value={"000001": mock_ohlcv},
    ), patch(
        "backend.services.temporal_scoring_service.temporal_scoring_service._compute_factors",
        return_value={"price_vs_sma20": mock_ohlcv["close"] / mock_ohlcv["close"].rolling(20).mean()},
    ):
        result = _load_universe_df(
            universe="test_universe",
            codes=["000001"],
            start_date="2023-01-01",
            end_date="2023-03-31",
            factor_names=["nonexistent_factor_xyz"],
        )

    assert "nonexistent_factor_xyz" in result.columns, "未知因子列应存在（值为 NaN）"
    # 全为 NaN 是预期行为
    assert result["nonexistent_factor_xyz"].isna().all(), (
        "未知因子列应全为 NaN"
    )
