"""
应用配置模块
"""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # 项目路径
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CACHE_DIR: Path = DATA_DIR / "cache"
    DB_DIR: Path = DATA_DIR / "db"
    CONFIG_DIR: Path = BASE_DIR / "config"
    REPORTS_DIR: Path = DATA_DIR / "reports"

    # 数据库配置
    DATABASE_URL: str = f"sqlite:///{DB_DIR}/factorflow.db"

    # akshare 配置
    AKSHARE_CACHE_ENABLED: bool = True
    AKSHARE_CACHE_DIR: Path = CACHE_DIR / "akshare"

    # 缓存配置
    CACHE_DEFAULT_TTL: int = 7 * 24 * 60 * 60  # 默认TTL: 7天（秒）
    CACHE_AUTO_CLEANUP: bool = True  # 是否自动清理过期缓存
    CACHE_CLEANUP_INTERVAL: int = 24 * 60 * 60  # 清理间隔: 24小时（秒）

    # 数据预处理配置
    DATA_OUTLIER_DETECTION: bool = True  # 是否启用异常值检测
    DATA_OUTLIER_N_SIGMA: float = 3.0  # 异常值检测的σ倍数
    DATA_OUTLIER_METHOD: str = "clip"  # 异常值处理方法: clip/remove/replace
    DATA_FILL_MISSING: bool = True  # 是否填充缺失值
    DATA_FILL_METHOD: str = "ffill"  # 缺失值填充方法: ffill/bfill/interpolate

    # 分析配置
    DEFAULT_START_DATE: str = "2020-01-01"
    DEFAULT_END_DATE: str = "2024-12-31"

    # 滚动窗口配置
    ROLLING_WINDOW: int = 252  # 一年交易日

    # 时序打分配置
    SCORING_MAX_WORKERS: int = 4
    SCORING_SYNC_THRESHOLD: int = 20
    SCORING_RESULTS_DIR: Path = Path("data/reports/scoring_jobs")
    SCORING_LOOKBACK_DAYS: int = 372

    # 评估链路质量门禁配置
    EVAL_MIN_CODE_COVERAGE: float = 0.85
    EVAL_MIN_IC_SAMPLES: int = 30

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 创建必要的目录
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.DB_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.AKSHARE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.SCORING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# 全局配置实例
settings = Settings()
