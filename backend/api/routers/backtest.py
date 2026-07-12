"""
策略回测API路由
"""
from datetime import date, datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.services.vectorbt_backtest_service import VectorBTBacktestService, check_vectorbt_available
from backend.repositories.backtest_repository import BacktestRepository
from backend.repositories.factor_repository import FactorRepository
from backend.core.database import get_db_session

router = APIRouter()


def _clean_json_value(value):
    """递归清理对象，确保可被 JSON 序列化。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, (datetime, date, pd.Timestamp)):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        return pd.Timestamp(value).isoformat()

    if isinstance(value, dict):
        return {str(key): _clean_json_value(item) for key, item in value.items()}

    if isinstance(value, pd.DataFrame):
        return _clean_json_value(value.to_dict("records"))

    if isinstance(value, (pd.Series, pd.Index)):
        return [_clean_json_value(item) for item in value.tolist()]

    if isinstance(value, np.ndarray):
        return [_clean_json_value(item) for item in value.tolist()]

    if isinstance(value, (list, tuple, set)):
        return [_clean_json_value(item) for item in value]

    if isinstance(value, np.generic):
        return _clean_json_value(value.item())

    if isinstance(value, float):
        return None if not math.isfinite(value) else value

    return value


# ========== 数据模型 ==========

class SingleBacktestRequest(BaseModel):
    """单策略回测请求"""
    data_mode: str = "single"  # single 或 pool
    stock_codes: List[str]
    factor_name: Optional[str] = None  # 单因子时的因子名称
    factor_names: Optional[List[str]] = None  # 多因子时的因子列表
    strategy_type: str = "single_factor"  # single_factor 或 multi_factor
    start_date: str
    end_date: str
    initial_capital: float = 1000000
    commission_rate: float = 0.0003
    slippage: float = 0.0
    percentile: int = 50
    direction: str = "long"
    n_quantiles: int = 5
    weight_method: str = "equal_weight"  # 多因子时的权重方法
    shares_per_trade: int = 100  # 每次交易手数，默认1手（100股）


class ComparisonRequest(BaseModel):
    """策略对比请求"""
    data_mode: str = "single"
    stock_codes: List[str]
    strategies: List[Dict]  # 策略配置列表
    start_date: str
    end_date: str
    initial_capital: float = 1000000
    commission_rate: float = 0.0003
    rebalance_freq: str = "monthly"


# ========== API端点 ==========

@router.post("/single")
async def run_single_backtest(request: SingleBacktestRequest):
    """运行单策略回测"""
    try:
        if not check_vectorbt_available():
            raise HTTPException(
                status_code=503,
                detail="VectorBT未安装，请先安装: pip install vectorbt"
            )

        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db_session

        # 确定要使用的因子列表
        if request.strategy_type == "multi_factor":
            # 多因子策略
            factor_names_to_use = request.factor_names or []
            if not factor_names_to_use:
                raise HTTPException(status_code=400, detail="多因子策略需要选择至少一个因子")
            primary_factor_name = factor_names_to_use[0]  # 使用第一个因子作为主因子
        else:
            # 单因子策略
            factor_names_to_use = [request.factor_name] if request.factor_name else []
            primary_factor_name = request.factor_name
            if not primary_factor_name:
                raise HTTPException(status_code=400, detail="请选择因子")

        # 从数据库获取所有因子定义
        db = get_db_session()
        repo = FactorRepository(db)
        factor_defs = {}
        for factor_name in factor_names_to_use:
            factor_def = repo.get_by_name(factor_name)
            if not factor_def:
                db.close()
                raise HTTPException(status_code=404, detail=f"因子 '{factor_name}' 不存在")
            factor_defs[factor_name] = factor_def
        db.close()

        # 获取数据并计算所有因子
        all_factor_data = {}
        for stock_code in request.stock_codes:
            stock_data = data_service.get_stock_data(
                stock_code,
                request.start_date,
                request.end_date
            )

            if stock_data is not None and len(stock_data) > 0:
                # 计算所有选中的因子
                factor_calculator = factor_service.calculator
                for factor_name in factor_names_to_use:
                    factor_def = factor_defs[factor_name]
                    factor_values = factor_calculator.calculate(
                        stock_data, factor_def.code, getattr(factor_def, "formula_type", None)
                    )
                    stock_data[factor_name] = factor_values

                all_factor_data[stock_code] = stock_data

        if not all_factor_data:
            raise HTTPException(status_code=404, detail="未获取到有效数据")

        # 创建回测服务
        backtest_service = VectorBTBacktestService(
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate,
            slippage=request.slippage,
        )

        is_single_stock = len(all_factor_data) == 1

        # 执行回测
        if request.strategy_type == "single_factor":
            if is_single_stock:
                df = list(all_factor_data.values())[0].copy()
                result = backtest_service.single_factor_backtest(
                    df=df,
                    factor_name=primary_factor_name,
                    percentile=request.percentile,
                    direction=request.direction,
                    n_quantiles=request.n_quantiles,
                    shares_per_trade=request.shares_per_trade,
                )
            else:
                # 横截面回测
                merged_list = []
                for code, data in all_factor_data.items():
                    data_copy = data.copy()
                    data_copy["stock_code"] = code
                    if "date" not in data_copy.columns:
                        data_copy = data_copy.reset_index()
                    merged_list.append(data_copy)

                df = pd.concat(merged_list, ignore_index=True)
                top_percentile = (100 - request.percentile) / 100.0
                result = backtest_service.cross_sectional_backtest(
                    df=df,
                    factor_name=primary_factor_name,
                    top_percentile=top_percentile,
                    direction=request.direction,
                )
        else:
            # 多因子回测
            if is_single_stock:
                df = list(all_factor_data.values())[0].copy()
                result = backtest_service.multi_factor_backtest(
                    df=df,
                    factor_names=factor_names_to_use,
                    method=request.weight_method,
                    percentile=request.percentile,
                    direction=request.direction,
                    shares_per_trade=request.shares_per_trade,
                )
            else:
                merged_list = []
                for code, data in all_factor_data.items():
                    prepared_data = backtest_service.prepare_multi_factor_data(
                        df=data.copy(),
                        factor_names=factor_names_to_use,
                        method=request.weight_method,
                    )
                    prepared_data["stock_code"] = code
                    if "date" not in prepared_data.columns:
                        prepared_data = prepared_data.reset_index()
                    merged_list.append(prepared_data)

                df = pd.concat(merged_list, ignore_index=True)
                top_percentile = (100 - request.percentile) / 100.0
                primary_factor_name = "composite_score"
                result = backtest_service.cross_sectional_backtest(
                    df=df,
                    factor_name=primary_factor_name,
                    top_percentile=top_percentile,
                    direction=request.direction,
                )

        # 提取指标
        metrics = {k: v for k, v in result.items() if k in [
            "total_return", "annual_return", "volatility", "sharpe_ratio",
            "max_drawdown", "calmar_ratio", "win_rate", "sortino_ratio",
            "var_95", "cvar_95"
        ]}

        result_serializable = _clean_json_value(result)

        # 添加详细的图表数据
        chart_data = {}

        # 为每只股票生成图表数据（单股票和多股票模式都支持）
        for stock_code, df in all_factor_data.items():
            # 确保索引是DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                if "date" in df.columns:
                    df = df.set_index("date")
                df.index = pd.to_datetime(df.index)

            # 清理因子数据中的NaN和Inf值
            def clean_factor_values(series):
                """清理因子值：将NaN和Inf替换为None"""
                cleaned = []
                for val in series:
                    # 使用numpy的isinf函数检查无穷大值
                    if pd.isna(val) or np.isinf(val):
                        cleaned.append(None)
                    else:
                        cleaned.append(float(val))
                return cleaned

            # K线数据
            stock_chart_data = {
                "kline": {
                    "dates": df.index.strftime('%Y-%m-%d').tolist(),
                    "open": df["open"].tolist() if "open" in df.columns else df["close"].tolist(),
                    "high": df["high"].tolist() if "high" in df.columns else df["close"].tolist(),
                    "low": df["low"].tolist() if "low" in df.columns else df["close"].tolist(),
                    "close": df["close"].tolist(),
                },
                "factor": {
                    "dates": df.index.strftime('%Y-%m-%d').tolist(),
                    # 单因子模式：保持原有格式（向后兼容）
                    # 多因子模式：返回所有因子数据
                    "factors": [
                        {
                            "name": factor_name,
                            "values": clean_factor_values(df[factor_name])
                        }
                        for factor_name in factor_names_to_use
                        if factor_name in df.columns
                    ]
                }
            }

            # 买卖信号 - 两种类型
            if is_single_stock:
                factor_rank = df[primary_factor_name].rolling(252, min_periods=1).rank(pct=True)
                percentile_threshold = request.percentile / 100.0

                if request.direction == "long":
                    raw_entries = factor_rank >= percentile_threshold
                    raw_exits = factor_rank < percentile_threshold
                else:
                    raw_entries = factor_rank <= percentile_threshold
                    raw_exits = factor_rank > percentile_threshold

                entries = raw_entries.shift(1, fill_value=False)
                exits = raw_exits.shift(1, fill_value=False)
            else:
                selection_by_date = result.get("selection_by_date", {})
                entry_flags = []
                exit_flags = []
                previous_selected = False
                for date in df.index.strftime('%Y-%m-%d'):
                    current_selected = stock_code in selection_by_date.get(date, [])
                    entry_flags.append(current_selected)
                    exit_flags.append(previous_selected and not current_selected)
                    previous_selected = current_selected
                entries = pd.Series(entry_flags, index=df.index).astype(bool)
                exits = pd.Series(exit_flags, index=df.index).astype(bool)

            # 1. 策略信号（所有满足条件的信号，不考虑持仓状态）
            strategy_buy_dates = df.index[entries].strftime('%Y-%m-%d').tolist()
            strategy_buy_prices = df.loc[entries, "close"].tolist()
            strategy_sell_dates = df.index[exits].strftime('%Y-%m-%d').tolist()
            strategy_sell_prices = df.loc[exits, "close"].tolist()

            # 2. 实际交易信号（从VectorBT交易记录中提取）
            actual_buy_dates = []
            actual_buy_prices = []
            actual_sell_dates = []
            actual_sell_prices = []

            if result_serializable.get("trades"):
                trades_df = result["trades"]
                if trades_df is not None and len(trades_df) > 0:
                    # 在多股票模式下，只提取当前股票的交易记录
                    stock_trades_df = trades_df

                    # 如果有'股票代码'列，筛选出当前股票的交易
                    if '股票代码' in trades_df.columns:
                        stock_trades_df = trades_df[trades_df['股票代码'] == stock_code]

                    # trades_df的索引是入场时间（DatetimeIndex，name='入场时间'）
                    # 直接遍历索引和行
                    for entry_time, row in stock_trades_df.iterrows():
                        # entry_time 是买入日期（Timestamp）
                        if pd.notna(entry_time):
                            buy_date = pd.Timestamp(entry_time).strftime('%Y-%m-%d')
                            if '入场价格' in row and pd.notna(row['入场价格']):
                                actual_buy_dates.append(buy_date)
                                actual_buy_prices.append(float(row['入场价格']))

                        # 提取出场时间（卖出）
                        if '出场时间' in row and pd.notna(row['出场时间']):
                            exit_time = row['出场时间']
                            #出场时间可能是字符串或Timestamp
                            if isinstance(exit_time, str):
                                sell_date = exit_time  # 已经是格式化的字符串
                            else:
                                sell_date = pd.Timestamp(exit_time).strftime('%Y-%m-%d')

                            actual_sell_dates.append(sell_date)

                            if '出场价格' in row and pd.notna(row['出场价格']):
                                actual_sell_prices.append(float(row['出场价格']))

            stock_chart_data["signals"] = {
                "strategy": {
                    "buy": {
                        "dates": strategy_buy_dates,
                        "prices": strategy_buy_prices
                    },
                    "sell": {
                        "dates": strategy_sell_dates,
                        "prices": strategy_sell_prices
                    }
                },
                "actual": {
                    "buy": {
                        "dates": actual_buy_dates,
                        "prices": actual_buy_prices
                    },
                    "sell": {
                        "dates": actual_sell_dates,
                        "prices": actual_sell_prices
                    }
                }
            }

            # 净值曲线
            if "equity_curve" in result_serializable:
                # 单股票模式使用回测结果的净值曲线
                equity_dates = result["equity_curve"].index.strftime('%Y-%m-%d').tolist()
                stock_chart_data["equity"] = {
                    "dates": equity_dates,
                    "values": result_serializable["equity_curve"]
                }

            # 保存当前股票的图表数据
            chart_data[stock_code] = stock_chart_data

        # 如果是单股票模式，保持向后兼容的格式，只返回该股票的数据
        if is_single_stock and len(all_factor_data) > 0:
            first_stock = list(all_factor_data.keys())[0]
            chart_data = chart_data[first_stock]

        # 清洗数据以确保JSON序列化有效
        cleaned_metrics = _clean_json_value(metrics)
        cleaned_result = _clean_json_value(result_serializable)
        cleaned_chart_data = _clean_json_value(chart_data)

        return {
            "success": True,
            "data": {
                "metrics": cleaned_metrics,
                "result": cleaned_result,
                "chart_data": cleaned_chart_data
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/comparison")
async def run_strategy_comparison(request: ComparisonRequest):
    """运行策略对比"""
    try:
        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service

        # 获取数据
        all_data = {}
        for stock_code in request.stock_codes:
            data = data_service.get_stock_data(
                stock_code,
                request.start_date,
                request.end_date
            )
            if not data.empty:
                all_data[stock_code] = data

        if not all_data:
            raise HTTPException(status_code=404, detail="未获取到任何数据")

        # 计算每个策略的收益
        results = {}
        for config in request.strategies:
            strategy_name = config.get("name", "未命名策略")

            # 从数据库获取因子定义
            db = get_db_session()
            repo = FactorRepository(db)
            factor_def = repo.get_by_name(config["factor"])
            db.close()

            if not factor_def:
                continue

            strategy_frames = []
            for stock_code, stock_data in all_data.items():
                stock_df = stock_data.copy()
                factor_values = factor_service.calculator.calculate(
                    stock_df, factor_def.code, getattr(factor_def, "formula_type", None)
                )
                if factor_values is None:
                    continue

                stock_df["factor_value"] = factor_values
                stock_df["return"] = stock_df["close"].pct_change().shift(-1)
                stock_df["stock_code"] = stock_code
                strategy_frames.append(stock_df.rename_axis("date").reset_index())

            if not strategy_frames:
                continue

            factor_df = pd.concat(strategy_frames, ignore_index=True)
            factor_df = factor_df.sort_values(["date", "stock_code"])

            selected_returns = []
            selected_dates = []
            for date, group in factor_df.groupby("date"):
                group = group.dropna(subset=["factor_value", "return"])
                if group.empty:
                    continue

                top_pct = config.get("top_pct", 20) / 100
                n_select = max(1, int(len(group) * top_pct))

                if config.get("direction", "long") == "long":
                    top_stocks = group.nlargest(n_select, "factor_value")
                else:
                    top_stocks = group.nsmallest(n_select, "factor_value")

                if len(top_stocks) == 0:
                    continue

                avg_return = top_stocks["return"].mean()
                if pd.notna(avg_return):
                    selected_returns.append(avg_return)
                    selected_dates.append(date)

            if selected_returns:
                returns_series = pd.Series(selected_returns, index=selected_dates)
                total_return = float((1 + returns_series).prod() - 1)
                n_days = len(returns_series)

                if n_days > 0:
                    annual_return = float((1 + total_return) ** (252 / n_days) - 1)
                else:
                    annual_return = 0.0

                results[strategy_name] = {
                    "dates": [pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in returns_series.index],
                    "returns": returns_series.tolist(),
                    "total_return": total_return,
                    "annual_return": annual_return,
                    "volatility": float(returns_series.std() * np.sqrt(252)),
                }

                if results[strategy_name]["volatility"] > 0:
                    results[strategy_name]["sharpe_ratio"] = (
                        results[strategy_name]["annual_return"] /
                        results[strategy_name]["volatility"]
                    )
                else:
                    results[strategy_name]["sharpe_ratio"] = 0.0

        results_clean = _clean_json_value(results)

        return {
            "success": True,
            "data": {
                "results": results_clean
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_backtest_history(limit: int = 10):
    """获取回测历史"""
    try:
        repo = BacktestRepository()
        history = repo.get_history(limit=limit)

        # 转换为字典列表
        history_list = []
        for record in history:
            history_list.append({
                "id": record.id,
                "strategy_name": record.strategy_name,
                "factor_combination": record.factor_combination,
                "start_date": record.start_date,
                "end_date": record.end_date,
                "total_return": record.total_return,
                "sharpe_ratio": record.sharpe_ratio,
                "max_drawdown": record.max_drawdown,
                "created_at": record.created_at.isoformat()
            })

        return {
            "success": True,
            "data": history_list,
            "total": len(history_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{record_id}")
async def delete_backtest_history(record_id: int):
    """删除回测历史"""
    try:
        repo = BacktestRepository()
        success = repo.delete_by_id(record_id)

        if not success:
            raise HTTPException(status_code=404, detail="记录不存在或删除失败")

        return {
            "success": True,
            "message": "删除成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
