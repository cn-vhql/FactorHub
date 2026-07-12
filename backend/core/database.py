"""
数据库连接管理模块
"""
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator

from backend.core.settings import settings


# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

# 创建 Session 工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """数据库模型基类"""
    pass


def init_db() -> None:
    """初始化数据库，创建所有表"""
    from backend.models.factor import FactorModel, AnalysisCacheModel
    from backend.models.backtest import BacktestResultModel, TradeRecordModel
    from backend.models.cache_metadata import CacheMetadataModel
    from backend.models.factor_version import FactorVersionModel

    Base.metadata.create_all(bind=engine)
    _migrate_legacy_columns()


def _migrate_legacy_columns() -> None:
    """补齐历史数据库缺失列。"""
    inspector = inspect(engine)

    if not inspector.has_table("factors"):
        return

    columns = {column["name"] for column in inspector.get_columns("factors")}
    if "formula_type" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE factors "
                    "ADD COLUMN formula_type VARCHAR(20) NOT NULL DEFAULT 'mylanguage'"
                )
            )
            connection.execute(
                text(
                    """
                    UPDATE factors
                    SET formula_type = CASE
                        WHEN lower(trim(code)) LIKE 'def calculate_factor%'
                             OR instr(lower(code), 'df[') > 0
                             OR instr(lower(code), 'np.') > 0
                             OR instr(lower(code), 'pd.') > 0
                             OR instr(lower(code), '.rolling(') > 0
                             OR instr(lower(code), '.expanding(') > 0
                             OR instr(lower(code), '.astype(') > 0
                             OR instr(lower(code), '.shift(') > 0
                             OR instr(lower(code), 'lambda ') > 0
                        THEN 'python'
                        ELSE 'mylanguage'
                    END
                    """
                )
            )


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """获取数据库会话的上下文管理器"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """获取数据库会话（非上下文管理器方式）"""
    return SessionLocal()
