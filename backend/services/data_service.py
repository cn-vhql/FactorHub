"""
数据服务模块 - 股票数据获取与缓存
"""
import hashlib
import ssl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
import akshare as ak


# ---------------------------------------------------------------------------
# 新浪财经 SSL 兼容补丁
# akshare 直接调用 requests.get，需要 patch 全局函数
# 同时清空代理（proxies={}），避免系统代理干扰
# ---------------------------------------------------------------------------

class _LegacySSLAdapter(HTTPAdapter):
    """降级 SSL 安全级别，兼容新浪财经等旧 TLS 服务器（OpenSSL 3.x 兼容）"""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


_SINA_HOSTS = ("stock.finance.sina.com.cn", "finance.sina.com.cn", "hq.sinajs.cn")
_legacy_session = requests.Session()
_legacy_session.mount("https://", _LegacySSLAdapter())
_legacy_session.trust_env = False  # 忽略系统代理环境变量

_orig_requests_get = requests.get


def _patched_requests_get(url, **kwargs):
    """对新浪财经域名使用降级 SSL + 无代理；其余请求保持原样"""
    if any(host in url for host in _SINA_HOSTS):
        kwargs.setdefault("proxies", {})
        kwargs.setdefault("verify", False)
        kwargs.setdefault("timeout", 30)
        return _legacy_session.get(url, **kwargs)
    return _orig_requests_get(url, **kwargs)


requests.get = _patched_requests_get

from backend.core.settings import settings
from backend.services.cache_service import cache_service
from backend.services.data_preprocessing_service import data_preprocessing_service


@dataclass
class DataFetchReport:
    """数据拉取报告，包含覆盖率和失败详情"""

    requested_codes: list
    success_codes: list
    failed_codes: list
    data: dict
    failures: list = field(default_factory=list)

    @property
    def code_coverage(self) -> float:
        """成功拉取的股票占请求总数的比例；请求为空时返回 0.0"""
        if not self.requested_codes:
            return 0.0
        return len(self.success_codes) / len(self.requested_codes)


class DataService:
    """数据服务类 - 负责股票数据获取和缓存"""

    def __init__(self):
        self.cache_dir = settings.AKSHARE_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_service = cache_service
        self.preprocessing = data_preprocessing_service

    def _get_cache_key(self, stock_code: str, start_date: str, end_date: str) -> str:
        """生成缓存键"""
        cache_key = f"{stock_code}_{start_date}_{end_date}"
        return hashlib.md5(cache_key.encode()).hexdigest()

    def _get_cache_path(self, stock_code: str, start_date: str, end_date: str) -> Path:
        """生成缓存文件路径（保留向后兼容）"""
        cache_hash = self._get_cache_key(stock_code, start_date, end_date)
        return self.cache_dir / f"{cache_hash}.pkl"

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从智能缓存加载数据"""
        return self.cache_service.get(cache_key)

    def _save_to_cache(self, data: pd.DataFrame, cache_key: str, ttl: Optional[int] = None) -> None:
        """保存数据到智能缓存"""
        if ttl is None:
            ttl = settings.CACHE_DEFAULT_TTL
        self.cache_service.set(cache_key, data, ttl=ttl)

    def get_stock_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        获取股票历史数据

        Args:
            stock_code: 股票代码，如 "000001" 或 "000001.SZ"
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"
            use_cache: 是否使用缓存

        Returns:
            包含OHLCV数据的DataFrame
        """
        # 标准化股票代码
        stock_code = self._normalize_stock_code(stock_code)

        # 检查智能缓存
        if use_cache and settings.AKSHARE_CACHE_ENABLED:
            cache_key = self._get_cache_key(stock_code, start_date, end_date)
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

        # 从 akshare 获取数据
        try:
            if stock_code.endswith(".SH"):
                symbol = "sh" + stock_code.replace(".SH", "")
                df = ak.stock_zh_a_daily(
                    symbol=symbol,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",  # 前复权
                )
            elif stock_code.endswith(".SZ"):
                symbol = "sz" + stock_code.replace(".SZ", "")
                df = ak.stock_zh_a_daily(
                    symbol=symbol,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )
            else:
                # 尝试自动识别
                # 添加市场前缀
                if stock_code.startswith("6"):
                    symbol = "sh" + stock_code
                elif stock_code.startswith(("0", "3")):
                    symbol = "sz" + stock_code
                else:
                    symbol = stock_code

                df = ak.stock_zh_a_daily(
                    symbol=symbol,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )

            # 标准化列名
            df = self._standardize_columns(df)

            # 数据预处理
            df = self._preprocess_data(df)

            # 保存到智能缓存
            if use_cache and settings.AKSHARE_CACHE_ENABLED:
                cache_key = self._get_cache_key(stock_code, start_date, end_date)
                self._save_to_cache(df, cache_key)

            return df

        except Exception as e:
            raise ValueError(f"获取股票 {stock_code} 数据失败: {e}")

    def _normalize_stock_code(self, code: str) -> str:
        """标准化股票代码格式"""
        code = code.strip().upper()
        if not code.endswith((".SH", ".SZ")):
            # 补齐 6 位前导零（如 "928" -> "000928"）
            if code.isdigit() and len(code) < 6:
                code = code.zfill(6)
            # 自动判断上海或深圳
            if code.startswith("6") or code.startswith("9"):
                return f"{code}.SH"
            elif code.startswith(("0", "3")):
                return f"{code}.SZ"
        return code

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化DataFrame列名"""
        # akshare 返回的列名映射
        column_mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }

        df = df.rename(columns=column_mapping)

        # 确保日期列是datetime类型
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        # 确保数值列是正确的类型
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.sort_index()

    def get_multiple_stocks_data(
        self,
        stock_codes: list[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        获取多个股票的数据

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存

        Returns:
            字典，key为股票代码，value为对应的DataFrame
        """
        result = {}
        for code in stock_codes:
            try:
                df = self.get_stock_data(code, start_date, end_date, use_cache)
                result[code] = df
            except Exception as e:
                print(f"Warning: 获取股票 {code} 数据失败: {e}")
        return result

    def get_multiple_stocks_data_with_report(
        self,
        stock_codes: list[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> DataFetchReport:
        """
        获取多个股票的数据，并返回结构化的拉取报告

        与 get_multiple_stocks_data 逻辑相同，但捕获每只股票的失败信息，
        供调用方进行覆盖率门禁判断。原有方法保持不变。

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存

        Returns:
            DataFetchReport，包含 requested_codes、success_codes、failed_codes、
            code_coverage、data 和 failures 字段
        """
        data: dict[str, pd.DataFrame] = {}
        success_codes: list[str] = []
        failed_codes: list[str] = []
        failures: list[dict] = []

        for code in stock_codes:
            try:
                df = self.get_stock_data(code, start_date, end_date, use_cache)
                data[code] = df
                success_codes.append(code)
            except Exception as e:
                failed_codes.append(code)
                failures.append(
                    {
                        "code": code,
                        "error_type": type(e).__name__,
                        "message": str(e),
                    }
                )

        return DataFetchReport(
            requested_codes=list(stock_codes),
            success_codes=success_codes,
            failed_codes=failed_codes,
            data=data,
            failures=failures,
        )

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据

        Args:
            df: 原始数据框

        Returns:
            预处理后的数据框
        """
        # 填充缺失值
        if settings.DATA_FILL_MISSING:
            df = self.preprocessing.fill_missing_values(
                df,
                method=settings.DATA_FILL_METHOD,
            )

        # 异常值检测和处理
        if settings.DATA_OUTLIER_DETECTION:
            df, _ = self.preprocessing.detect_and_handle_anomalies(
                df,
                price_columns=["open", "high", "low", "close"],
                n_sigma=settings.DATA_OUTLIER_N_SIGMA,
                handle_method=settings.DATA_OUTLIER_METHOD,
            )

        return df

    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        return self.cache_service.get_stats()

    def cleanup_cache(self) -> int:
        """清理过期缓存"""
        return self.cache_service.cleanup_expired()

    def clear_cache(self) -> int:
        """清空所有缓存"""
        return self.cache_service.clear_all()

    def incremental_update(
        self,
        stock_code: str,
        existing_df: pd.DataFrame,
        end_date: str,
    ) -> pd.DataFrame:
        """
        增量更新股票数据

        Args:
            stock_code: 股票代码
            existing_df: 现有的数据框
            end_date: 新的结束日期

        Returns:
            更新后的数据框
        """
        # 获取现有数据的最后日期
        last_date = existing_df.index.max()
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # 如果新日期在现有数据之前，直接返回现有数据
        if start_date > end_date:
            return existing_df

        # 获取新数据
        new_df = self.get_stock_data(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            use_cache=True,
        )

        # 增量合并
        combined_df = self.preprocessing.incremental_update(
            existing_df=existing_df,
            new_df=new_df,
        )

        return combined_df


# 全局数据服务实例
data_service = DataService()
