"""
因子相关数据模型
"""
from datetime import datetime
from sqlalchemy import String, DateTime, Text, Integer, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class FactorModel(Base):
    """因子模型"""

    __tablename__ = "factors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    code: Mapped[str] = mapped_column(Text, nullable=False)  # 因子计算代码
    description: Mapped[str] = mapped_column(Text, nullable="", default="")
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="user")  # preset 或 user
    category: Mapped[str] = mapped_column(String(50), nullable="", default="")  # 因子分类
    formula_type: Mapped[str] = mapped_column(String(20), nullable=False, default="expression")  # expression 或 function
    is_active: Mapped[int] = mapped_column(Integer, default=1)  # 0: 禁用, 1: 启用
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "description": self.description,
            "source": self.source,
            "category": self.category,
            "formula_type": self.formula_type,
            "is_active": bool(self.is_active),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AnalysisCacheModel(Base):
    """分析结果缓存模型"""

    __tablename__ = "analysis_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cache_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    stock_codes: Mapped[str] = mapped_column(String(500), nullable=False)  # 逗号分隔的股票代码
    factor_names: Mapped[str] = mapped_column(String(1000), nullable=False)  # 逗号分隔的因子名
    start_date: Mapped[str] = mapped_column(String(20), nullable=False)
    end_date: Mapped[str] = mapped_column(String(20), nullable=False)
    result_data: Mapped[dict] = mapped_column(JSON, nullable=False)  # 存储分析结果
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "cache_key": self.cache_key,
            "stock_codes": self.stock_codes,
            "factor_names": self.factor_names,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "result_data": self.result_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
