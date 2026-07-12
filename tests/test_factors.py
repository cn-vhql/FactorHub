import asyncio

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.background import BackgroundTasks
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api.routers.factors import validate_factor
from backend.api.routers import mining as mining_router
from backend.api.routers.mining import GeneticMiningRequest, start_genetic_mining
from backend.core.database import Base
from backend.models.factor import FactorModel
from backend.formula_engine.code_normalizer import normalize_formula_code
from backend.services import factor_service as factor_service_module
from backend.services.factor_service import factor_service
from backend.services.genetic_factor_mining_service import GeneticFactorMiningService


def test_update_factor_rejects_duplicate_name(monkeypatch, tmp_path):
    db_path = tmp_path / "factor_test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    session = SessionLocal()
    alpha = FactorModel(name="alpha", code="close", source="user", category="test", is_active=1)
    beta = FactorModel(name="beta", code="open", source="user", category="test", is_active=1)
    session.add_all([alpha, beta])
    session.commit()
    session.refresh(alpha)
    session.refresh(beta)
    session.close()

    monkeypatch.setattr(
        factor_service_module,
        "get_db_session",
        SessionLocal,
    )

    with pytest.raises(ValueError, match="已存在"):
        factor_service.update_factor(alpha.id, name="beta", create_version=False)


def test_genetic_mining_requires_deap(monkeypatch):
    from backend.services import genetic_factor_mining_service as mining_service_module

    monkeypatch.setattr(mining_service_module, "DEAP_AVAILABLE", False)

    request = GeneticMiningRequest(
        stock_code="000001.SZ",
        base_factors=[],
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(start_genetic_mining(request, BackgroundTasks()))

    assert exc_info.value.status_code == 503


def test_genetic_mining_marks_task_failed_when_selected_base_factors_are_unavailable(monkeypatch):
    class DummySession:
        def close(self):
            return None

    class EmptyFactorRepository:
        def __init__(self, db):
            self.db = db

        def get_by_name(self, name):
            return None

    class DummyDataService:
        @staticmethod
        def get_stock_data(stock_code, start_date, end_date):
            return pd.DataFrame(
                {
                    "close": [10.0, 10.5, 10.8],
                    "open": [9.8, 10.2, 10.6],
                    "high": [10.1, 10.7, 10.9],
                    "low": [9.7, 10.1, 10.5],
                    "volume": [1000, 1100, 1200],
                }
            )

    monkeypatch.setattr("backend.core.database.get_db_session", lambda: DummySession())
    monkeypatch.setattr("backend.repositories.factor_repository.FactorRepository", EmptyFactorRepository)
    monkeypatch.setattr("backend.services.data_service.data_service", DummyDataService())

    task_id = "missing-base-factor-task"
    mining_router.mining_tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "result": None,
        "error": None,
    }

    request = GeneticMiningRequest(
        stock_code="000001.SZ",
        base_factors=["not_exists"],
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    asyncio.run(mining_router._run_genetic_mining(task_id, request))

    assert mining_router.mining_tasks[task_id]["status"] == "failed"
    assert "所选基础因子均不可用" in mining_router.mining_tasks[task_id]["error"]


def test_create_factor_persists_formula_type(monkeypatch, tmp_path):
    db_path = tmp_path / "factor_formula_type.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(
        factor_service_module,
        "get_db_session",
        SessionLocal,
    )

    created = factor_service.create_factor(
        name="alpha_mylang",
        code="XG:CLOSE/REF(CLOSE,1)",
        category="test",
        formula_type="mylanguage",
    )

    assert created["formula_type"] == "mylanguage"


def test_create_factor_preserves_mylanguage_program_for_storage(monkeypatch, tmp_path):
    db_path = tmp_path / "factor_mylanguage_program.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(
        factor_service_module,
        "get_db_session",
        SessionLocal,
    )

    code = "DIFF:=EMA(CLOSE,12)-EMA(CLOSE,26); DEA:=EMA(DIFF,9); MACD:(DIFF-DEA)*2; XG:DIFF"
    created = factor_service.create_factor(
        name="alpha_macd_program",
        code=code,
        category="test",
        formula_type="mylanguage",
    )

    assert created["formula_type"] == "mylanguage"
    assert created["code"] == code


def test_load_preset_factors_updates_existing_preset_definition(monkeypatch, tmp_path):
    db_path = tmp_path / "preset_sync.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    session = SessionLocal()
    session.add(
        FactorModel(
            name="percentile_20",
            code="close.rolling(window=20).apply(lambda x: 0.0)",
            formula_type="python",
            description="旧实现",
            source="preset",
            category="旧分类",
            is_active=1,
        )
    )
    session.commit()
    session.close()

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "factors.yaml").write_text("", encoding="utf-8")

    monkeypatch.setattr(factor_service_module, "get_db_session", SessionLocal)
    monkeypatch.setattr(factor_service_module.settings, "CONFIG_DIR", config_dir)

    factor_service.load_preset_factors()

    session = SessionLocal()
    updated = session.query(FactorModel).filter(FactorModel.name == "percentile_20").one()
    session.close()

    assert updated.code == "RANGEPOS(close, 20)"
    assert updated.formula_type == "mylanguage"
    assert updated.category == "价格位置"
    assert "20日价格分位数" in updated.description


def test_create_factor_rejects_invalid_code(monkeypatch, tmp_path):
    db_path = tmp_path / "invalid_factor.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(factor_service_module, "get_db_session", SessionLocal)

    with pytest.raises(ValueError, match="因子代码校验失败"):
        factor_service.create_factor(
            name="broken_factor",
            code="UNKNOWNFUNC(close)",
            category="test",
            formula_type="mylanguage",
        )


def test_cleanup_legacy_generated_factors_migrates_mined_and_deletes_composite(monkeypatch, tmp_path):
    db_path = tmp_path / "legacy_generated_factors.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    session = SessionLocal()
    session.add_all(
        [
            FactorModel(
                name="legacy_mined_factor",
                code="""def calculate_factor(df):
    \"\"\"
    遗传算法挖掘因子
    表达式: np.sqrt(((HHV(high, 20) - close) / HHV(high, 20)))
    IC: -0.3972
    IR: -1.9073
    \"\"\"
    import pandas as pd
    import numpy as np
    return np.sqrt(((HHV(df['high'], 20) - df['close']) / HHV(df['high'], 20)))
""",
                formula_type="python",
                description="通过遗传算法挖掘的因子",
                source="user",
                category="遗传挖掘",
                is_active=1,
            ),
            FactorModel(
                name="legacy_composite_factor",
                code="""def calculate_factor(df):
    \"\"\"
    组合因子 - 最大夏普优化
    \"\"\"
    composite = factor_1 + factor_2
    return composite
""",
                formula_type="python",
                description="旧版组合因子包装",
                source="user",
                category="组合因子",
                is_active=1,
            ),
            FactorModel(
                name="broken_mined_factor",
                code="((np.log(close / close.shift(1))) * (def calculate_factor(df): return close))",
                formula_type="python",
                description="旧版挖掘表达式残片",
                source="user",
                category="遗传挖掘",
                is_active=1,
            ),
            FactorModel(
                name="manual_python_factor",
                code="""def calculate_factor(df):
    return EMA(df['close'], timeperiod=12)
""",
                formula_type="python",
                description="手写 Python 因子",
                source="user",
                category="自定义",
                is_active=1,
            ),
        ]
    )
    session.commit()
    session.close()

    monkeypatch.setattr(factor_service_module, "get_db_session", SessionLocal)

    summary = factor_service.cleanup_legacy_generated_factors()

    assert summary["migrated"] == 1
    assert summary["deleted"] == 2
    assert "legacy_mined_factor" in summary["migrated_names"]
    assert "legacy_composite_factor" in summary["deleted_names"]
    assert "broken_mined_factor" in summary["deleted_names"]

    session = SessionLocal()
    migrated = session.query(FactorModel).filter(FactorModel.name == "legacy_mined_factor").one()
    preserved = session.query(FactorModel).filter(FactorModel.name == "manual_python_factor").one()
    deleted = session.query(FactorModel).filter(FactorModel.name == "legacy_composite_factor").all()
    broken_deleted = session.query(FactorModel).filter(FactorModel.name == "broken_mined_factor").all()
    session.close()

    assert migrated.code == "np.sqrt(((HHV(high, 20) - close) / HHV(high, 20)))"
    assert migrated.formula_type == "python"
    assert preserved.code.startswith("def calculate_factor")
    assert deleted == []
    assert broken_deleted == []


def test_validate_factor_code_returns_friendly_message_for_lambda():
    is_valid, message = factor_service.validate_factor_code(
        "close.rolling(window=20).apply(lambda x: x.iloc[-1])",
        formula_type="python",
    )

    assert is_valid is False
    assert "rolling(...).apply(...)" in message


def test_validate_factor_code_returns_friendly_message_for_mylanguage_dot_syntax():
    is_valid, message = factor_service.validate_factor_code(
        "close.rolling(window=20).mean()",
        formula_type="mylanguage",
    )

    assert is_valid is False
    assert "不支持对象点号语法" in message


def test_validate_factor_code_returns_friendly_message_for_python_loop():
    is_valid, message = factor_service.validate_factor_code(
        """def calculate_factor(df):
    total = 0
    for value in close:
        total += value
    return total
""",
        formula_type="python",
    )

    assert is_valid is False
    assert "不支持 for / while 循环" in message


def test_validate_api_returns_resolved_formula_type():
    response = asyncio.run(
        validate_factor(
            {
                "code": "CLOSE / MA(CLOSE, 20)",
                "formula_type": "auto",
            }
        )
    )

    assert response["success"] is True
    assert response["data"]["formula_type"] == "mylanguage"
    assert "验证通过（类型: mylanguage）" in response["message"]


def test_genetic_mining_converts_mylanguage_primary_output_to_composable_expression():
    service = GeneticFactorMiningService.__new__(GeneticFactorMiningService)
    service.factor_calculator = factor_service.calculator

    expression = service._build_composable_factor_code(
        "PREV:=REF(CLOSE,1); XG:(CLOSE-PREV)/PREV"
    )

    assert expression == "((CLOSE - (REF(CLOSE, 1))) / (REF(CLOSE, 1)))"


def test_genetic_mining_replaces_internal_factor_with_composable_expression():
    service = GeneticFactorMiningService.__new__(GeneticFactorMiningService)
    service.factor_calculator = factor_service.calculator
    service.base_factor_values = {
        "factor_0": {
            "code": "((CLOSE - (REF(CLOSE, 1))) / (REF(CLOSE, 1)))",
            "source_code": "PREV:=REF(CLOSE,1); XG:(CLOSE-PREV)/PREV",
            "values": pd.Series([1.0, 2.0, 3.0]),
        }
    }

    converted = service._convert_expression_to_code("np.log(np.sqrt(factor_0))")

    assert converted == "np.log(np.sqrt((((CLOSE - (REF(CLOSE, 1))) / (REF(CLOSE, 1))))))"


def test_genetic_mining_skips_python_function_factor_for_composition():
    service = GeneticFactorMiningService.__new__(GeneticFactorMiningService)
    service.factor_calculator = factor_service.calculator

    expression = service._build_composable_factor_code(
        """def calculate_factor(df):
    return close / open
"""
    )

    assert expression is None


def test_normalize_formula_code_rewrites_embedded_mylanguage_program():
    normalized = normalize_formula_code(
        "np.log(np.sqrt((PREV:=REF(CLOSE,1); XG:(CLOSE-PREV)/PREV)))",
        formula_type="python",
    )

    assert normalized == "np.log(np.sqrt((((CLOSE - (REF(CLOSE, 1))) / (REF(CLOSE, 1))))))"


def test_create_factor_normalizes_legacy_mined_expression_before_validation(monkeypatch, tmp_path):
    db_path = tmp_path / "normalized_mined_factor.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(factor_service_module, "get_db_session", SessionLocal)

    created = factor_service.create_factor(
        name="normalized_mined_factor",
        code="np.log(np.sqrt((PREV:=REF(CLOSE,1); XG:(CLOSE-PREV)/PREV)))",
        category="遗传挖掘",
        formula_type="auto",
    )

    assert created["formula_type"] == "python"
    assert created["code"] == "np.log(np.sqrt((((CLOSE - (REF(CLOSE, 1))) / (REF(CLOSE, 1))))))"


def test_create_factor_normalizes_composite_expression_with_embedded_mylanguage(monkeypatch, tmp_path):
    db_path = tmp_path / "normalized_composite_factor.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(factor_service_module, "get_db_session", SessionLocal)

    composite_code = (
        "(((PREV:=REF(CLOSE,1); XG:(CLOSE-PREV)/PREV) - "
        "(PREV:=REF(CLOSE,1); XG:(CLOSE-PREV)/PREV).expanding(min_periods=1).mean()) / "
        "((PREV:=REF(CLOSE,1); XG:(CLOSE-PREV)/PREV).expanding(min_periods=1).std() + 1e-8))"
    )

    created = factor_service.create_factor(
        name="normalized_composite_factor",
        code=composite_code,
        category="组合因子",
        formula_type="python",
    )

    assert created["formula_type"] == "python"
    assert "PREV:=" not in created["code"]
    assert "XG:" not in created["code"]
    assert "REF(CLOSE, 1)" in created["code"]


def test_validate_factor_code_normalizes_legacy_escaped_python_quotes():
    code = """def calculate_factor(df):
    import pandas as pd
    import numpy as np
    try:
        result = (df[\\"close\\"] - df[\\"close\\"].shift(1)).fillna(0)
        return result
    except Exception as e:
        return pd.Series(0, index=df.index)
"""

    is_valid, message = factor_service.validate_factor_code(code, formula_type="python")

    assert is_valid is True
    assert "验证通过（类型: python）" in message


def test_repair_stored_factor_codes_persists_historical_fixes_without_touching_valid_mylanguage(monkeypatch, tmp_path):
    db_path = tmp_path / "repair_stored_factors.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    session = SessionLocal()
    session.add_all(
        [
            FactorModel(
                name="legacy_escaped_python",
                code="""def calculate_factor(df):
    import pandas as pd
    result = (df[\\"close\\"] - df[\\"open\\"]).fillna(0)
    return result
""",
                formula_type="python",
                description="历史错误转义",
                source="user",
                category="QA",
                is_active=1,
            ),
            FactorModel(
                name="embedded_mylanguage_python",
                code="np.log(np.sqrt((PREV:=REF(CLOSE,1); XG:(CLOSE-PREV)/PREV)))",
                formula_type="python",
                description="历史嵌入式麦语言组合表达式",
                source="user",
                category="组合因子",
                is_active=1,
            ),
            FactorModel(
                name="legacy_generated_wrapper",
                code="""def calculate_factor(df):
    \"\"\"
    遗传算法挖掘因子
    表达式: np.sqrt(((HHV(high, 20) - close) / HHV(high, 20)))
    \"\"\"
    import pandas as pd
    import numpy as np
    return np.sqrt(((HHV(df['high'], 20) - df['close']) / HHV(df['high'], 20)))
""",
                formula_type="python",
                description="通过遗传算法挖掘的因子",
                source="user",
                category="遗传挖掘",
                is_active=1,
            ),
            FactorModel(
                name="valid_mylanguage_program",
                code="DIFF:=EMA(CLOSE,12)-EMA(CLOSE,26); DEA:=EMA(DIFF,9); MACD:(DIFF-DEA)*2; XG:DIFF",
                formula_type="mylanguage",
                description="合法多输出麦语言程序",
                source="user",
                category="副图",
                is_active=1,
            ),
        ]
    )
    session.commit()
    session.close()

    monkeypatch.setattr(factor_service_module, "get_db_session", SessionLocal)

    summary = factor_service.repair_stored_factor_codes()

    assert summary["scanned"] == 4
    assert summary["repaired"] == 3
    assert summary["unchanged"] == 1
    assert summary["failed"] == 0

    session = SessionLocal()
    escaped = session.query(FactorModel).filter(FactorModel.name == "legacy_escaped_python").one()
    embedded = session.query(FactorModel).filter(FactorModel.name == "embedded_mylanguage_python").one()
    generated = session.query(FactorModel).filter(FactorModel.name == "legacy_generated_wrapper").one()
    mylanguage_program = session.query(FactorModel).filter(FactorModel.name == "valid_mylanguage_program").one()
    session.close()

    assert '\\"' not in escaped.code
    assert "PREV:=" not in embedded.code
    assert "XG:" not in embedded.code
    assert embedded.code == "np.log(np.sqrt((((CLOSE - (REF(CLOSE, 1))) / (REF(CLOSE, 1))))))"
    assert generated.code == "np.sqrt(((HHV(high, 20) - close) / HHV(high, 20)))"
    assert generated.formula_type == "python"
    assert mylanguage_program.code == "DIFF:=EMA(CLOSE,12)-EMA(CLOSE,26); DEA:=EMA(DIFF,9); MACD:(DIFF-DEA)*2; XG:DIFF"
