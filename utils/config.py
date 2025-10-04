"""
全局配置文件
"""
import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 数据目录
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
FACTORS_DIR = DATA_DIR / "factors"
EXPORTS_DIR = DATA_DIR / "exports"

# 创建必要的目录
for dir_path in [DATA_DIR, CACHE_DIR, FACTORS_DIR, EXPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 默认配置
DEFAULT_CONFIG = {
    "data": {
        "default_start_date": "2020-01-01",
        "default_end_date": "2023-12-31",
        "frequency": "daily",
        "adjust_type": "qfq",  # 前复权
        "fill_method": "ffill",  # 前向填充
        "outlier_method": "3sigma",  # 3σ原则
    },
    "analysis": {
        "default_periods": 5,  # 分层层数
        "rolling_window": 60,  # 滚动窗口天数
        "benchmark": "000300",  # 沪深300
    },
    "backtest": {
        "commission": 0.0003,  # 手续费0.03%
        "slippage": 0.001,     # 滑点0.1%
        "rebalance_frequency": "monthly",  # 调仓频率
    },
    "mining": {
        "population_size": 50,
        "generations": 10,
        "mutation_rate": 0.1,
        "crossover_rate": 0.7,
        "max_complexity": 3,  # 最大因子复杂度
        "top_factors": 10,  # 输出因子数量
    }
}

# 预置因子列表
PRESET_FACTORS = {
    "trend": {
        "MACD": "MACD指标",
        "MA5": "5日移动平均线",
        "MA10": "10日移动平均线",
        "MA20": "20日移动平均线",
        "MA60": "60日移动平均线",
    },
    "momentum": {
        "RSI": "相对强弱指标",
        "MOM": "动量指标",
        "BIAS": "乖离率",
    },
    "volatility": {
        "STD": "价格标准差",
        "ATR": "平均真实波幅",
        "VOL": "成交量波动率",
    },
    "value": {
        "PE": "市盈率",
        "PB": "市净率",
        "PS": "市销率",
        "EV": "企业价值倍数",
    }
}

# 市场配置
MARKETS = {
    "沪市": "SH",
    "深市": "SZ",
    "全部": "ALL"
}

# 股票池配置
STOCK_POOLS = {
    "全市场": "all",
    "沪深300": "hs300",
    "中证500": "zz500",
    "创业板": "cyb"
}