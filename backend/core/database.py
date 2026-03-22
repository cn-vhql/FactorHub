"""
数据库连接管理模块
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import logging

from backend.core.settings import settings

logger = logging.getLogger(__name__)

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


def _column_exists(conn, table: str, column: str) -> bool:
    """检查指定表中是否存在某列（SQLite PRAGMA）"""
    result = conn.execute(text(f"PRAGMA table_info({table})"))
    return any(row[1] == column for row in result)


def _migration_applied(conn, migration_id: str) -> bool:
    """检查 schema_migrations 表中是否已记录某次迁移"""
    try:
        result = conn.execute(
            text("SELECT 1 FROM schema_migrations WHERE migration_id = :mid"),
            {"mid": migration_id},
        )
        return result.fetchone() is not None
    except Exception:
        return False


def _ensure_schema_migrations_table(conn) -> None:
    """确保 schema_migrations 表存在"""
    conn.execute(text(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_id VARCHAR(100) PRIMARY KEY,
            applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    ))


def _run_migrations(conn) -> None:
    """执行所有幂等 schema 迁移，失败时抛出异常阻止启动"""
    _ensure_schema_migrations_table(conn)

    migration_id = "add_formula_type_to_factors"

    if _migration_applied(conn, migration_id):
        logger.debug("Migration '%s' already applied, skipping.", migration_id)
        return

    if not _column_exists(conn, "factors", "formula_type"):
        logger.info("Applying migration '%s': adding formula_type column to factors table.", migration_id)
        try:
            conn.execute(text(
                "ALTER TABLE factors ADD COLUMN formula_type VARCHAR(20) NOT NULL DEFAULT 'expression'"
            ))
            # 回填历史数据（ALTER TABLE DEFAULT 已覆盖，此处显式 UPDATE 确保一致性）
            conn.execute(text(
                "UPDATE factors SET formula_type = 'expression' WHERE formula_type IS NULL OR formula_type = ''"
            ))
        except Exception as exc:
            raise RuntimeError(
                f"Schema migration '{migration_id}' failed: {exc}. "
                "Application startup aborted."
            ) from exc

    # 记录迁移已完成
    conn.execute(
        text("INSERT INTO schema_migrations (migration_id) VALUES (:mid)"),
        {"mid": migration_id},
    )
    logger.info("Migration '%s' applied successfully.", migration_id)


def init_db() -> None:
    """初始化数据库，创建所有表，并执行幂等 schema 迁移"""
    from backend.models.factor import FactorModel, AnalysisCacheModel
    from backend.models.backtest import BacktestResultModel, TradeRecordModel
    from backend.models.cache_metadata import CacheMetadataModel
    from backend.models.factor_version import FactorVersionModel

    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        _run_migrations(conn)


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
