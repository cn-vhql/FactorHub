"""
辅助函数
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

def validate_stock_code(code: str) -> bool:
    """验证股票代码格式"""
    code = str(code).upper()
    return (code.isdigit() and len(code) == 6 and
            (code.startswith('0') or code.startswith('3') or code.startswith('6')))

def normalize_stock_code(code: str) -> str:
    """标准化股票代码格式"""
    code = str(code).zfill(6).upper()
    return code

def format_date(date_str: str) -> str:
    """格式化日期为YYYY-MM-DD"""
    if pd.isna(date_str):
        return ""

    if isinstance(date_str, str):
        try:
            date_obj = pd.to_datetime(date_str)
            return date_obj.strftime('%Y-%m-%d')
        except:
            return ""
    elif isinstance(date_str, (datetime, pd.Timestamp)):
        return date_str.strftime('%Y-%m-%d')

    return ""

def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """计算收益率"""
    return prices.pct_change(periods=periods)

def remove_outliers_3sigma(data: pd.Series) -> pd.Series:
    """3σ原则剔除异常值"""
    mean = data.mean()
    std = data.std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    return data.clip(lower=lower_bound, upper=upper_bound)

def forward_fill(data: pd.DataFrame) -> pd.DataFrame:
    """前向填充缺失值"""
    return data.ffill()

def linear_interpolate(data: pd.DataFrame) -> pd.DataFrame:
    """线性插值填充缺失值"""
    return data.interpolate(method='linear')

def calculate_ic(predicted: pd.Series, actual: pd.Series, method: str = 'spearman') -> float:
    """计算IC值"""
    # 对齐数据
    aligned_data = pd.DataFrame({'pred': predicted, 'act': actual}).dropna()
    if len(aligned_data) < 2:
        return np.nan

    if method == 'spearman':
        return aligned_data['pred'].corr(aligned_data['act'], method='spearman')
    elif method == 'pearson':
        return aligned_data['pred'].corr(aligned_data['act'], method='pearson')
    else:
        raise ValueError("method must be 'spearman' or 'pearson'")

def calculate_ir(ic_series: pd.Series) -> float:
    """计算IR值"""
    if len(ic_series) < 2:
        return np.nan
    return ic_series.mean() / ic_series.std()

def calculate_ic_win_rate(ic_series: pd.Series) -> float:
    """计算IC胜率"""
    if len(ic_series) == 0:
        return np.nan
    return (ic_series > 0).sum() / len(ic_series)

def create_date_range(start_date: str, end_date: str, frequency: str = 'daily') -> pd.DatetimeIndex:
    """创建日期范围"""
    if frequency == 'daily':
        return pd.date_range(start=start_date, end=end_date, freq='D')
    elif frequency == 'weekly':
        return pd.date_range(start=start_date, end=end_date, freq='W')
    elif frequency == 'monthly':
        return pd.date_range(start=start_date, end=end_date, freq='M')
    else:
        raise ValueError("frequency must be 'daily', 'weekly', or 'monthly'")

def save_to_cache(data: Any, cache_key: str, cache_dir: str = None) -> None:
    """保存数据到缓存"""
    import pickle
    from pathlib import Path

    if cache_dir is None:
        from .config import CACHE_DIR
        cache_dir = CACHE_DIR

    cache_file = Path(cache_dir) / f"{cache_key}.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

def load_from_cache(cache_key: str, cache_dir: str = None, max_age_days: int = 7) -> Any:
    """从缓存加载数据"""
    import pickle
    from pathlib import Path
    from datetime import datetime

    if cache_dir is None:
        from .config import CACHE_DIR
        cache_dir = CACHE_DIR

    cache_file = Path(cache_dir) / f"{cache_key}.pkl"

    if not cache_file.exists():
        return None

    # 检查缓存文件是否过期
    file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
    age_days = (datetime.now() - file_mtime).days

    if age_days > max_age_days:
        # 缓存过期，删除文件
        try:
            cache_file.unlink()
        except:
            pass
        return None

    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            return data
    except:
        # 缓存文件损坏，删除文件
        try:
            cache_file.unlink()
        except:
            pass
        return None

def format_number(num: float, decimal_places: int = 4) -> str:
    """格式化数字显示"""
    if pd.isna(num):
        return "N/A"
    return f"{num:.{decimal_places}f}"

def format_percentage(num: float, decimal_places: int = 2) -> str:
    """格式化百分比显示"""
    if pd.isna(num):
        return "N/A"
    return f"{num*100:.{decimal_places}f}%"