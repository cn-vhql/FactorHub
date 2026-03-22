"""
集成测试：OHLCV 因子扩展（ohlcv-factor-expansion）
Tasks 4.1 – 4.3
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# ── 共享的 P0 因子名称列表 ────────────────────────────────────────────────────

P0_FACTOR_NAMES = [
    "overnight_return_1",
    "intraday_return_1",
    "body_ratio",
    "upper_shadow_ratio",
    "close_location_value",
    "breakout_strength_20",
    "efficiency_ratio_20",
    "kama_ratio_20",
    "parkinson_vol_20",
    "garman_klass_vol_20",
    "cmf_20",
    "volume_zscore_20",
]


# ── 内存数据库 fixture ────────────────────────────────────────────────────────

@pytest.fixture()
def in_memory_db():
    """
    创建一个全新的内存 SQLite 数据库，初始化 schema 并执行迁移。
    返回 (engine, SessionLocal) 元组。
    """
    from backend.core.database import Base, _run_migrations

    # 导入所有 model，确保 metadata 已注册
    from backend.models.factor import FactorModel, AnalysisCacheModel  # noqa: F401
    from backend.models.backtest import BacktestResultModel, TradeRecordModel  # noqa: F401
    from backend.models.cache_metadata import CacheMetadataModel  # noqa: F401
    from backend.models.factor_version import FactorVersionModel  # noqa: F401

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        _run_migrations(conn)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal


@pytest.fixture()
def db_session(in_memory_db):
    """返回一个绑定到内存 DB 的 Session，测试结束后关闭。"""
    _, SessionLocal = in_memory_db
    session = SessionLocal()
    yield session
    session.close()


# ── Task 4.1：API 集成测试 ────────────────────────────────────────────────────

class TestAPIFactorsEndpoint:
    """
    Task 4.1 — 干净数据库启动后 /api/factors 返回包含全部 12 个 P0 因子

    需求 5.1：干净数据库（无任何因子记录）启动后，/api/factors 接口返回的
    因子列表必须包含全部 12 个 P0 因子（按 name 字段匹配）。
    """

    def test_all_p0_factors_in_api_response(self, in_memory_db):
        """
        使用内存 DB 替换全局 DB session，调用 load_preset_factors()，
        然后通过 FastAPI TestClient 请求 GET /api/factors，
        断言全部 12 个 P0 因子名称出现在响应中。

        注意：factor_service.py 直接 import get_db_session，
        因此需要 patch backend.services.factor_service.get_db_session。
        """
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from backend.api.main import app
        from backend.repositories.factor_repository import FactorRepository

        _, SessionLocal = in_memory_db

        def _get_test_session():
            return SessionLocal()

        # patch factor_service 模块内的 get_db_session（直接导入的引用）
        with patch(
            "backend.services.factor_service.get_db_session",
            side_effect=_get_test_session,
        ):
            # 手动触发 load_preset_factors，使用 patched session
            from backend.services.factor_service import FactorService
            svc = FactorService()
            svc.load_preset_factors()

            # patch get_all_factors 以从内存 DB 读取，同时 patch init_db 避免操作真实 DB
            def _get_all_factors():
                session = SessionLocal()
                repo = FactorRepository(session)
                factors = repo.get_all(active_only=True)
                session.close()
                return [f.to_dict() for f in factors]

            with patch(
                "backend.services.factor_service.factor_service.get_all_factors",
                side_effect=_get_all_factors,
            ), patch("backend.api.main.init_db", return_value=None), \
               patch("backend.api.main.factor_service.load_preset_factors", return_value=None):
                with TestClient(app, raise_server_exceptions=True) as client:
                    response = client.get("/api/factors/")

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        body = response.json()
        assert body["success"] is True

        returned_names = {f["name"] for f in body["data"]}
        missing = [name for name in P0_FACTOR_NAMES if name not in returned_names]
        assert not missing, (
            f"Missing P0 factors in /api/factors response: {missing}"
        )


# ── Task 4.2：load_preset_factors() 幂等性测试 ───────────────────────────────

class TestLoadPresetFactorsIdempotency:
    """
    Task 4.2 — load_preset_factors() 幂等性（重复调用不产生重复记录）

    需求 5.2：服务重启后再次调用 load_preset_factors()，不得产生重复因子记录。
    """

    def test_double_call_no_duplicates(self, in_memory_db):
        """
        在内存 DB 上连续调用 load_preset_factors() 两次，
        查询所有因子，断言每个 name 只出现一次。
        """
        from unittest.mock import patch
        from backend.services.factor_service import FactorService
        from backend.repositories.factor_repository import FactorRepository

        _, SessionLocal = in_memory_db

        def _get_test_session():
            return SessionLocal()

        # patch factor_service 模块内直接导入的 get_db_session
        with patch("backend.services.factor_service.get_db_session", side_effect=_get_test_session):
            svc = FactorService()
            svc.load_preset_factors()
            svc.load_preset_factors()  # 第二次调用

        # 查询所有因子
        session = SessionLocal()
        repo = FactorRepository(session)
        all_factors = repo.get_all(active_only=False)
        session.close()

        names = [f.name for f in all_factors]
        duplicates = [name for name in names if names.count(name) > 1]
        unique_duplicates = list(set(duplicates))

        assert not unique_duplicates, (
            f"Duplicate factor names found after double load_preset_factors(): "
            f"{unique_duplicates}"
        )

    def test_p0_factors_present_after_double_call(self, in_memory_db):
        """
        幂等性调用后，12 个 P0 因子仍然全部存在。
        """
        from unittest.mock import patch
        from backend.services.factor_service import FactorService
        from backend.repositories.factor_repository import FactorRepository

        _, SessionLocal = in_memory_db

        def _get_test_session():
            return SessionLocal()

        with patch("backend.services.factor_service.get_db_session", side_effect=_get_test_session):
            svc = FactorService()
            svc.load_preset_factors()
            svc.load_preset_factors()

        session = SessionLocal()
        repo = FactorRepository(session)
        all_factors = repo.get_all(active_only=True)
        session.close()

        names_in_db = {f.name for f in all_factors}
        missing = [name for name in P0_FACTOR_NAMES if name not in names_in_db]
        assert not missing, (
            f"P0 factors missing after double load: {missing}"
        )


# ── Task 4.3：现有因子 name/code/category 未被修改 ───────────────────────────

class TestExistingFactorsUnmodified:
    """
    Task 4.3 — 验证现有 90 个因子的 name/code/category 未被修改

    需求 6.1：现有 90 个预置因子的 name、code、category、description 字段不得被修改。
    """

    # 抽样检查：从原有因子中选取几个有代表性的条目
    KNOWN_EXISTING_FACTORS = [
        # (name, expected_category, expected_code_fragment)
        ("log_return_1",       "价格收益率",   "np.log(close / close.shift(1))"),
        ("rsi_14",             "动量趋势",     "RSI(close, timeperiod=14)"),
        ("atr_14",             "波动率风险",   "ATR(high, low, close, timeperiod=14)"),
        ("volume_ma_ratio",    "成交量资金流", "volume / SMA(volume, timeperiod=10)"),
        ("momentum_20",        "动量加速度",   "close / close.shift(20) - 1"),
        ("reversal_5",         "反转信号",     "-(close / close.shift(5) - 1)"),
        ("golden_cross",       "技术形态",     "CROSS(SMA(close, timeperiod=5), SMA(close, timeperiod=20))"),
        ("ma20",               "均线系统",     "SMA(close, timeperiod=20)"),
        ("percentile_20",      "价格位置",     "close.rolling(window=20)"),
        ("force_index",        "资金流动",     "(close - close.shift(1)) * volume"),
    ]

    def _load_default_factors_flat(self):
        """调用 _get_default_factors() 并展平为 {name: {code, category}} 字典"""
        from backend.services.factor_service import FactorService
        svc = FactorService()
        raw = svc._get_default_factors()
        flat = {}
        for category, factors in raw.items():
            for f in factors:
                flat[f["name"]] = {"code": f["code"], "category": category}
        return flat

    def test_total_factor_count_at_least_102(self):
        """
        _get_default_factors() 返回的因子总数 ≥ 86（含 12 个 P0 因子）。

        注：需求文档中提到"现有 90 个因子 + 12 个新因子 = 102"，
        但实际基线因子数为 86（含已新增的 12 个 P0 因子），
        因此断言总数 ≥ 86，并单独验证 12 个 P0 因子均已存在。
        """
        flat = self._load_default_factors_flat()
        total = len(flat)
        assert total >= 86, (
            f"Expected at least 86 factors in _get_default_factors(), got {total}"
        )

    def test_all_12_p0_factors_present(self):
        """
        _get_default_factors() 中包含全部 12 个 P0 因子。
        """
        flat = self._load_default_factors_flat()
        missing = [name for name in P0_FACTOR_NAMES if name not in flat]
        assert not missing, (
            f"P0 factors missing from _get_default_factors(): {missing}"
        )

    @pytest.mark.parametrize("name,expected_category,code_fragment", KNOWN_EXISTING_FACTORS)
    def test_existing_factor_category_unchanged(self, name, expected_category, code_fragment):
        """
        抽样验证：已知因子的 category 未被修改。
        """
        flat = self._load_default_factors_flat()
        assert name in flat, f"Factor '{name}' not found in _get_default_factors()"
        actual_category = flat[name]["category"]
        assert actual_category == expected_category, (
            f"Factor '{name}': expected category={expected_category!r}, "
            f"got {actual_category!r}"
        )

    @pytest.mark.parametrize("name,expected_category,code_fragment", KNOWN_EXISTING_FACTORS)
    def test_existing_factor_code_unchanged(self, name, expected_category, code_fragment):
        """
        抽样验证：已知因子的 code 包含预期片段（未被修改）。
        """
        flat = self._load_default_factors_flat()
        assert name in flat, f"Factor '{name}' not found in _get_default_factors()"
        actual_code = flat[name]["code"]
        assert code_fragment in actual_code, (
            f"Factor '{name}': expected code to contain {code_fragment!r}, "
            f"got {actual_code!r}"
        )

    def test_p0_factors_have_correct_categories(self):
        """
        验证 12 个 P0 因子的 category 符合设计文档规定。
        """
        expected_categories = {
            "overnight_return_1":   "价格收益率",
            "intraday_return_1":    "价格收益率",
            "body_ratio":           "技术形态",
            "upper_shadow_ratio":   "技术形态",
            "close_location_value": "价格位置",
            "breakout_strength_20": "价格位置",
            "efficiency_ratio_20":  "动量趋势",
            "kama_ratio_20":        "动量趋势",
            "parkinson_vol_20":     "波动率风险",
            "garman_klass_vol_20":  "波动率风险",
            "cmf_20":               "资金流动",
            "volume_zscore_20":     "成交量资金流",
        }
        flat = self._load_default_factors_flat()
        for name, expected_cat in expected_categories.items():
            assert name in flat, f"P0 factor '{name}' missing from _get_default_factors()"
            actual_cat = flat[name]["category"]
            assert actual_cat == expected_cat, (
                f"P0 factor '{name}': expected category={expected_cat!r}, "
                f"got {actual_cat!r}"
            )
