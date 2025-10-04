"""
AKShare数据获取模块
"""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time

from utils.logger import logger
from utils.config import DEFAULT_CONFIG, CACHE_DIR
from utils.helpers import format_date, validate_stock_code, normalize_stock_code, save_to_cache, load_from_cache

class AKShareDataProvider:
    """AKShare数据提供者"""

    def __init__(self):
        self.logger = logger
        self.cache_enabled = True
        self.default_cache_max_age_days = 7  # 默认缓存7天

    def clear_cache(self, older_than_days: int = None):
        """清理缓存文件"""
        from pathlib import Path

        if older_than_days is None:
            older_than_days = self.default_cache_max_age_days

        cache_dir = CACHE_DIR
        if not cache_dir.exists():
            return

        cache_files = list(cache_dir.glob("daily_*.pkl"))
        cleaned_count = 0

        for cache_file in cache_files:
            try:
                file_age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
                if file_age_days > older_than_days:
                    cache_file.unlink()
                    cleaned_count += 1
                    self.logger.debug(f"删除过期缓存文件: {cache_file.name}")
            except Exception as e:
                self.logger.warning(f"删除缓存文件失败 {cache_file.name}: {str(e)}")

        if cleaned_count > 0:
            self.logger.info(f"清理了 {cleaned_count} 个过期缓存文件")

    def get_cache_info(self) -> Dict:
        """获取缓存信息"""
        cache_dir = CACHE_DIR
        if not cache_dir.exists():
            return {"total_files": 0, "total_size_mb": 0}

        cache_files = list(cache_dir.glob("daily_*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files if f.exists())

        return {
            "total_files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(cache_dir)
        }

    def get_stock_list(self, market: str = "all") -> pd.DataFrame:
        """获取股票列表"""
        try:
            # 获取股票列表
            all_stocks = ak.stock_info_a_code_name()

            if market == "all":
                stock_list = all_stocks
            elif market == "SH":
                stock_list = all_stocks[all_stocks['code'].str.startswith('6')]
            elif market == "SZ":
                stock_list = all_stocks[all_stocks['code'].str.startswith(('0', '3'))]
            else:
                raise ValueError(f"不支持的market参数: {market}")

            # 标准化股票代码
            stock_list = stock_list.copy()
            stock_list['code'] = stock_list['code'].apply(normalize_stock_code)
            stock_list = stock_list.rename(columns={
                'code': 'symbol',
                'name': 'name'
            })

            self.logger.info(f"获取到{len(stock_list)}只{market}股票")
            return stock_list

        except Exception as e:
            self.logger.error(f"获取股票列表失败: {str(e)}")
            return pd.DataFrame()

    def get_stock_basic_info(self, symbol: str) -> Dict:
        """获取股票基本信息"""
        try:
            symbol = normalize_stock_code(symbol)

            # 获取基本信息
            info = ak.stock_individual_info_em(symbol=symbol)
            info_dict = {}
            if info is not None and len(info) > 0:
                info_dict = dict(zip(info['item'], info['value']))

            return info_dict

        except Exception as e:
            self.logger.error(f"获取股票{symbol}基本信息失败: {str(e)}")
            return {}

    def get_daily_data(self,
                      symbol: str,
                      start_date: str,
                      end_date: str,
                      adjust: str = "qfq") -> pd.DataFrame:
        """获取日线数据"""
        try:
            symbol = normalize_stock_code(symbol)

            # 缓存key
            cache_key = f"daily_{symbol}_{start_date}_{end_date}_{adjust}"
            if self.cache_enabled:
                cached_data = load_from_cache(cache_key)
                if cached_data is not None:
                    # 检查缓存数据的完整性
                    if not cached_data.empty and len(cached_data) > 0:
                        self.logger.info(f"从缓存加载股票{symbol}数据 ({len(cached_data)}条记录)")
                        return cached_data
                    else:
                        self.logger.warning(f"缓存数据为空，重新获取股票{symbol}数据")

            # 获取日线数据
            df = ak.stock_zh_a_hist(symbol=symbol,
                                  period="daily",
                                  start_date=start_date.replace('-', ''),
                                  end_date=end_date.replace('-', ''),
                                  adjust=adjust)

            if df is None or len(df) == 0:
                self.logger.warning(f"股票{symbol}没有数据")
                return pd.DataFrame()

            # 标准化列名
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '换手率': 'turnover'
            })

            # 确保日期格式正确
            df['date'] = pd.to_datetime(df['date'])
            df['symbol'] = symbol

            # 选择需要的列
            columns = ['date', 'symbol', 'open', 'high', 'low', 'close',
                      'volume', 'amount', 'pct_change', 'turnover']
            df = df[[col for col in columns if col in df.columns]]

            # 按日期排序
            df = df.sort_values('date').reset_index(drop=True)

            # 缓存数据
            if self.cache_enabled:
                save_to_cache(df, cache_key)

            self.logger.info(f"获取股票{symbol}数据: {len(df)}行")
            return df

        except Exception as e:
            self.logger.error(f"获取股票{symbol}日线数据失败: {str(e)}")
            return pd.DataFrame()

    def get_multiple_stocks_data(self,
                                symbols: List[str],
                                start_date: str,
                                end_date: str,
                                adjust: str = "qfq",
                                progress_callback=None) -> pd.DataFrame:
        """批量获取多只股票数据"""
        all_data = []
        cached_count = 0
        fetched_count = 0
        total = len(symbols)

        self.logger.info(f"开始获取{total}只股票数据，检查缓存...")

        for i, symbol in enumerate(symbols):
            try:
                # 先检查缓存
                cache_key = f"daily_{symbol}_{start_date}_{end_date}_{adjust}"
                cached_data = None

                if self.cache_enabled:
                    cached_data = load_from_cache(cache_key)
                    if cached_data is not None and not cached_data.empty and len(cached_data) > 0:
                        all_data.append(cached_data)
                        cached_count += 1
                        self.logger.debug(f"从缓存加载股票{symbol}数据 ({len(cached_data)}条记录)")
                    else:
                        # 缓存不存在或无效，从akshare获取
                        df = self.get_daily_data(symbol, start_date, end_date, adjust)
                        if not df.empty:
                            all_data.append(df)
                            fetched_count += 1
                else:
                    # 缓存禁用，直接从akshare获取
                    df = self.get_daily_data(symbol, start_date, end_date, adjust)
                    if not df.empty:
                        all_data.append(df)
                        fetched_count += 1

                # 进度回调
                if progress_callback:
                    progress = (i + 1) / total
                    cache_status = "缓存" if cached_data is not None and not cached_data.empty else "网络"
                    progress_callback(progress, f"{symbol} ({cache_status})")

                # 只有实际进行网络请求时才需要延迟
                if cached_data is None or cached_data.empty:
                    time.sleep(0.1)  # 避免请求过于频繁

            except Exception as e:
                self.logger.error(f"获取股票{symbol}数据失败: {str(e)}")
                continue

        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"成功获取{len(result_df)}条数据记录 (缓存:{cached_count}, 网络:{fetched_count})")
            return result_df
        else:
            self.logger.warning("没有获取到任何数据")
            return pd.DataFrame()

    def get_index_data(self,
                      index_code: str,
                      start_date: str,
                      end_date: str) -> pd.DataFrame:
        """获取指数数据"""
        try:
            # 缓存key
            cache_key = f"index_{index_code}_{start_date}_{end_date}"
            if self.cache_enabled:
                cached_data = load_from_cache(cache_key)
                if cached_data is not None:
                    return cached_data

            df = ak.stock_zh_index_daily(symbol=index_code)

            if df is None or len(df) == 0:
                return pd.DataFrame()

            # 过滤日期范围
            df['date'] = pd.to_datetime(df.index)
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

            # 标准化列名
            df = df.rename(columns={
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount'
            })
            df['symbol'] = index_code

            columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']
            df = df[[col for col in columns if col in df.columns]]

            df = df.sort_values('date').reset_index(drop=True)

            # 缓存数据
            if self.cache_enabled:
                save_to_cache(df, cache_key)

            return df

        except Exception as e:
            self.logger.error(f"获取指数{index_code}数据失败: {str(e)}")
            return pd.DataFrame()

    def get_stock_pool(self, pool_name: str = "hs300") -> List[str]:
        """获取股票池"""
        try:
            if pool_name == "hs300":
                df = ak.index_stock_cons(symbol="000300")
                return df['品种代码'].tolist()
            elif pool_name == "zz500":
                df = ak.index_stock_cons(symbol="000905")
                return df['品种代码'].tolist()
            elif pool_name == "cyb":
                df = ak.index_stock_cons(symbol="399006")
                return df['品种代码'].tolist()
            elif pool_name == "all":
                stock_list = self.get_stock_list("all")
                return stock_list['symbol'].tolist()
            else:
                raise ValueError(f"不支持的股票池: {pool_name}")

        except Exception as e:
            self.logger.error(f"获取股票池{pool_name}失败: {str(e)}")
            return []

    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日历"""
        try:
            # 使用A股交易日历
            dates = ak.tool_trade_date_hist_sina()
            if dates is None or len(dates) == 0:
                return []

            # 转换格式并过滤
            dates['trade_date'] = pd.to_datetime(dates['trade_date'])
            dates = dates[(dates['trade_date'] >= start_date) & (dates['trade_date'] <= end_date)]

            return dates['trade_date'].dt.strftime('%Y-%m-%d').tolist()

        except Exception as e:
            self.logger.error(f"获取交易日历失败: {str(e)}")
            return []