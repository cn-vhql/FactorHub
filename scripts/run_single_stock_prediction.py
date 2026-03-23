"""Stage-5: 单票当日预测。

基于 Stage-4 冠军策略 + Stage-3 近期因子 + Stage-2 历史因子，
对指定股票在当前交易日做多维度预测，输出明日方向判断与置信度。
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pipeline_common import (
    bootstrap_runtime,
    ensure_output_dir,
    normalize_stock_code,
    write_json,
    write_text,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-5: 单票当日预测")
    parser.add_argument("--stock", default="600089", help="股票代码")
    parser.add_argument(
        "--trade-date",
        default="",
        help="预测基准日期 YYYY-MM-DD；为空时使用今天",
    )
    parser.add_argument(
        "--stage2-result",
        default="",
        help="Stage-2 result.json 路径；为空时自动推断",
    )
    parser.add_argument(
        "--stage3-result",
        default="",
        help="Stage-3 result.json 路径；为空时自动推断",
    )
    parser.add_argument(
        "--stage4-result",
        default="",
        help="Stage-4 result.json 路径；为空时自动推断",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=120,
        help="历史评分回望窗口（交易日），用于趋势分析",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="输出目录；为空时按 stock/trade_date 自动生成",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 路径推断
# ---------------------------------------------------------------------------

def _resolve_stage_path(explicit: str, stock_code: str, stage: str, trade_date: str) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    if stage == "stage2":
        base = Path("data/pipeline_runs/stage2_single_stock_factor_research")
        candidates = sorted(base.glob(f"{stock_code}_*/result.json"), reverse=True)
        return candidates[0] if candidates else None

    if stage == "stage3":
        p = Path(
            f"data/pipeline_runs/stage3_single_stock_recent_factor_mining/"
            f"{stock_code}_{trade_date}/result.json"
        )
        if p.exists():
            return p
        # 回退：取最新一个
        base = Path("data/pipeline_runs/stage3_single_stock_recent_factor_mining")
        candidates = sorted(base.glob(f"{stock_code}_*/result.json"), reverse=True)
        return candidates[0] if candidates else None

    if stage == "stage4":
        p = Path(
            f"data/pipeline_runs/stage4_single_stock_strategy_search/"
            f"{stock_code}_{trade_date}/result.json"
        )
        if p.exists():
            return p
        base = Path("data/pipeline_runs/stage4_single_stock_strategy_search")
        candidates = sorted(base.glob(f"{stock_code}_*/result.json"), reverse=True)
        return candidates[0] if candidates else None

    return None


def _load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"读取 {path} 失败: {exc}")
        return None


# ---------------------------------------------------------------------------
# 因子信号提取
# ---------------------------------------------------------------------------

def _extract_stage2_signals(data: Optional[dict], stock_code: str) -> dict:
    """从 Stage-2 提取历史有效因子信号。"""
    if not data:
        return {"available": False, "top_factors": [], "current_regime": "unknown"}

    top_factors = data.get("top_factors", [])[:10]
    meta = data.get("metadata", {})
    return {
        "available": True,
        "current_regime": meta.get("current_regime", "unknown"),
        "analysis_period": f"{meta.get('analysis_start_date', '')} ~ {meta.get('actual_end_date', '')}",
        "top_factors": [
            {
                "name": f["factor_name"],
                "direction": f["recommended_direction"],
                "abs_ic": round(f.get("abs_ic_mean", 0), 4),
                "abs_ir": round(f.get("abs_ir", 0), 4),
                "score": round(f.get("historical_score", 0), 2),
            }
            for f in top_factors
        ],
    }


def _extract_stage3_signals(data: Optional[dict], stock_code: str) -> dict:
    """从 Stage-3 提取近期有效因子信号。"""
    if not data:
        return {"available": False, "top_factors": [], "current_regime": "unknown"}

    per_stock = data.get("per_stock_top_factors", {}).get(stock_code, {})
    meta = data.get("metadata", {})
    top_factors = per_stock.get("top_factors", [])[:10]
    return {
        "available": True,
        "current_regime": per_stock.get("current_regime", "unknown"),
        "eval_window": f"{meta.get('eval_start_date', '')} ~ {meta.get('actual_end_date', '')}",
        "top_factors": [
            {
                "name": f["factor_name"],
                "direction": f["recommended_direction"],
                "abs_ic": round(f.get("abs_ic_mean", 0), 4),
                "abs_ir": round(f.get("abs_ir", 0), 4),
                "score": round(f.get("stock_score", 0), 2),
            }
            for f in top_factors
        ],
    }


def _extract_stage4_champion(data: Optional[dict]) -> dict:
    """从 Stage-4 提取冠军策略信息。"""
    if not data:
        return {"available": False}

    champion = data.get("champion", {})
    metrics = champion.get("metrics", {})
    return {
        "available": True,
        "profile_id": champion.get("profile_id", ""),
        "stability_score": round(champion.get("stability_score", 0), 2),
        "effective_from": champion.get("effective_from", ""),
        "test_sharpe": round(float(metrics.get("sharpe") or 0), 4),
        "test_annual_return": round(float(metrics.get("annual_return") or 0), 4),
        "test_max_drawdown": round(float(metrics.get("max_drawdown") or 0), 4),
        "test_active_ratio": round(float(metrics.get("active_ratio") or 0), 4),
        "inline_profile": champion.get("inline_profile"),
    }


# ---------------------------------------------------------------------------
# 当日评分（temporal scoring）
# ---------------------------------------------------------------------------

def _compute_today_score(stock_code: str, trade_date: str, profile_id: Optional[str]) -> dict:
    """调用 temporal_scoring_service 计算当日评分。"""
    from backend.services.temporal_scoring_service import temporal_scoring_service

    result = {"default_score": None, "profile_score": None, "factors": {}, "pcts": {}}

    # 默认 8 因子评分
    try:
        default_res = temporal_scoring_service.score_one_stock(stock_code, trade_date)
        result["default_score"] = default_res["score"]
        result["factors"] = default_res.get("factors", {})
        result["pcts"] = default_res.get("pcts", {})
        result["actual_date"] = default_res.get("date", trade_date)
    except Exception as exc:
        logger.warning(f"默认评分失败: {exc}")
        result["actual_date"] = trade_date

    # Champion profile 评分
    if profile_id:
        try:
            profile_res = temporal_scoring_service.score_one_stock_with_profile(
                stock_code, trade_date, profile_id
            )
            result["profile_score"] = profile_res["score"]
            result["profile_factors"] = profile_res.get("factors", {})
            result["profile_pcts"] = profile_res.get("pcts", {})
        except Exception as exc:
            logger.warning(f"Profile 评分失败 (profile_id={profile_id}): {exc}")

    return result


# ---------------------------------------------------------------------------
# 历史评分趋势
# ---------------------------------------------------------------------------

def _compute_score_trend(
    stock_code: str,
    trade_date: str,
    lookback_days: int,
    profile_id: Optional[str],
) -> dict:
    """计算近 lookback_days 交易日的评分趋势统计。"""
    from backend.services.temporal_scoring_service import temporal_scoring_service
    from backend.services.data_service import data_service

    trade_date_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    # 自然日换算（交易日约 0.7 倍）
    start_dt = trade_date_dt - timedelta(days=int(lookback_days * 1.5))
    start_date = start_dt.strftime("%Y-%m-%d")

    trend = {"available": False}

    try:
        if profile_id:
            hist_df = temporal_scoring_service.score_stock_history_with_profile(
                stock_code, start_date, trade_date, profile_id
            )
        else:
            hist_df = temporal_scoring_service.score_stock_history(
                stock_code, start_date, trade_date
            )

        if hist_df.empty:
            return trend

        scores = hist_df["score"].dropna()
        if len(scores) < 5:
            return trend

        recent_5 = scores.iloc[-5:].mean()
        recent_20 = scores.iloc[-20:].mean() if len(scores) >= 20 else scores.mean()
        recent_60 = scores.iloc[-60:].mean() if len(scores) >= 60 else scores.mean()
        current = float(scores.iloc[-1])

        # 趋势方向：近5日均分 vs 近20日均分
        short_trend = "上升" if recent_5 > recent_20 + 2 else ("下降" if recent_5 < recent_20 - 2 else "横盘")

        # 历史分位
        pct_rank = float(np.sum(scores <= current) / len(scores))

        trend = {
            "available": True,
            "current_score": round(current, 1),
            "recent_5d_avg": round(float(recent_5), 1),
            "recent_20d_avg": round(float(recent_20), 1),
            "recent_60d_avg": round(float(recent_60), 1),
            "score_max": round(float(scores.max()), 1),
            "score_min": round(float(scores.min()), 1),
            "score_std": round(float(scores.std()), 2),
            "short_trend": short_trend,
            "historical_percentile": round(pct_rank, 3),
            "sample_count": len(scores),
        }

        # 附上最近 20 条评分序列（供报告展示）
        recent_records = hist_df.tail(20)[["date", "score"]].copy()
        recent_records["date"] = recent_records["date"].astype(str)
        trend["recent_scores"] = recent_records.to_dict(orient="records")

    except Exception as exc:
        logger.warning(f"历史评分趋势计算失败: {exc}")

    return trend


# ---------------------------------------------------------------------------
# 技术面快照
# ---------------------------------------------------------------------------

def _compute_technical_snapshot(stock_code: str, trade_date: str) -> dict:
    """计算当日技术面快照：价格、均线、MACD、RSI、布林带、成交量等。"""
    from backend.services.data_service import data_service
    from backend.services.factor_service import factor_service

    trade_date_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    start_dt = trade_date_dt - timedelta(days=400)
    start_date = start_dt.strftime("%Y-%m-%d")

    snap = {"available": False}
    try:
        df = data_service.get_stock_data(stock_code, start_date, trade_date)
        if df.empty:
            return snap

        calc = factor_service.calculator
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        def last(series: pd.Series) -> Optional[float]:
            s = series.dropna()
            return round(float(s.iloc[-1]), 4) if not s.empty else None

        # 均线
        sma5 = last(calc.calculate(df, "SMA(close, timeperiod=5)"))
        sma10 = last(calc.calculate(df, "SMA(close, timeperiod=10)"))
        sma20 = last(calc.calculate(df, "SMA(close, timeperiod=20)"))
        sma60 = last(calc.calculate(df, "SMA(close, timeperiod=60)"))
        sma120 = last(calc.calculate(df, "SMA(close, timeperiod=120)"))

        # 当日价格
        current_close = last(close)
        current_open = last(df["open"])
        current_high = last(high)
        current_low = last(low)
        prev_close = round(float(close.dropna().iloc[-2]), 4) if len(close.dropna()) >= 2 else None
        pct_change = round((current_close / prev_close - 1) * 100, 2) if (current_close and prev_close) else None

        # RSI
        rsi14 = last(calc.calculate(df, "RSI(close, timeperiod=14)"))

        # MACD
        macd_line = last(calc.calculate(df, "MACD(close, fastperiod=12, slowperiod=26)[0]"))
        macd_signal = last(calc.calculate(df, "MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[1]"))
        macd_hist = last(calc.calculate(df, "MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[2]"))

        # 布林带
        bb_upper = last(calc.calculate(df, "BBANDS(close, timeperiod=20)[0]"))
        bb_mid = last(calc.calculate(df, "BBANDS(close, timeperiod=20)[1]"))
        bb_lower = last(calc.calculate(df, "BBANDS(close, timeperiod=20)[2]"))
        bb_pos = last(calc.calculate(
            df,
            "(close - BBANDS(close, timeperiod=20)[2]) / "
            "(BBANDS(close, timeperiod=20)[0] - BBANDS(close, timeperiod=20)[2])"
        ))

        # ATR
        atr14 = last(calc.calculate(df, "ATR(high, low, close, timeperiod=14)"))
        atr_norm = round(atr14 / current_close, 4) if (atr14 and current_close) else None

        # 成交量
        vol_ma10 = last(calc.calculate(df, "SMA(volume, timeperiod=10)"))
        vol_ratio = round(float(volume.dropna().iloc[-1]) / float(vol_ma10), 2) if vol_ma10 else None

        # KDJ
        stoch_k = last(calc.calculate(
            df,
            "(close - LLV(low, 14)) / (HHV(high, 14) - LLV(low, 14)) * 100"
        ))
        stoch_d = last(calc.calculate(
            df,
            "SMA((close - LLV(low, 14)) / (HHV(high, 14) - LLV(low, 14)) * 100, timeperiod=3)"
        ))

        # 20日新高/新低
        hhv20 = last(calc.calculate(df, "HHV(high, 20)"))
        llv20 = last(calc.calculate(df, "LLV(low, 20)"))
        is_new_high_20 = bool(current_high and hhv20 and current_high >= hhv20)
        is_new_low_20 = bool(current_low and llv20 and current_low <= llv20)

        # 均线多头排列判断
        ma_bullish = bool(sma5 and sma10 and sma20 and sma60 and
                          sma5 > sma10 > sma20 > sma60)
        ma_bearish = bool(sma5 and sma10 and sma20 and sma60 and
                          sma5 < sma10 < sma20 < sma60)

        snap = {
            "available": True,
            "price": {
                "open": current_open,
                "high": current_high,
                "low": current_low,
                "close": current_close,
                "prev_close": prev_close,
                "pct_change": pct_change,
            },
            "moving_averages": {
                "sma5": sma5, "sma10": sma10, "sma20": sma20,
                "sma60": sma60, "sma120": sma120,
                "ma_bullish_alignment": ma_bullish,
                "ma_bearish_alignment": ma_bearish,
            },
            "macd": {
                "macd_line": macd_line,
                "signal_line": macd_signal,
                "histogram": macd_hist,
                "golden_cross": bool(macd_hist and macd_hist > 0),
            },
            "rsi": {"rsi14": rsi14},
            "bollinger": {
                "upper": bb_upper, "mid": bb_mid, "lower": bb_lower,
                "position": bb_pos,
            },
            "kdj": {"k": stoch_k, "d": stoch_d},
            "volatility": {"atr14": atr14, "atr_norm": atr_norm},
            "volume": {"vol_ma10": vol_ma10, "vol_ratio": vol_ratio},
            "pattern": {
                "is_new_high_20d": is_new_high_20,
                "is_new_low_20d": is_new_low_20,
            },
        }
    except Exception as exc:
        logger.warning(f"技术面快照计算失败: {exc}")

    return snap


# ---------------------------------------------------------------------------
# 综合预测引擎
# ---------------------------------------------------------------------------

def _synthesize_prediction(
    today_score: dict,
    score_trend: dict,
    technical: dict,
    stage2: dict,
    stage3: dict,
    stage4: dict,
) -> dict:
    """
    综合多维度信号，输出明日方向预测与置信度。

    评分维度（满分 100）：
      - 当日 temporal score（默认 + profile）：30 分
      - 评分趋势（短期动量）：20 分
      - 技术面信号：30 分
      - Stage-2/3 因子共识：20 分
    """
    signals: list[dict] = []
    score_total = 0.0
    weight_total = 0.0

    # ---- 1. 当日 temporal score ----
    default_score = today_score.get("default_score")
    profile_score = today_score.get("profile_score")

    if default_score is not None:
        # 归一化到 [-1, 1]：score 50 为中性
        norm = (default_score - 50) / 50.0
        score_total += norm * 20
        weight_total += 20
        signals.append({
            "source": "temporal_score_default",
            "value": default_score,
            "signal": "看多" if norm > 0.1 else ("看空" if norm < -0.1 else "中性"),
            "weight": 20,
        })

    if profile_score is not None:
        norm = (profile_score - 50) / 50.0
        score_total += norm * 10
        weight_total += 10
        signals.append({
            "source": "temporal_score_champion_profile",
            "value": profile_score,
            "signal": "看多" if norm > 0.1 else ("看空" if norm < -0.1 else "中性"),
            "weight": 10,
        })

    # ---- 2. 评分趋势 ----
    if score_trend.get("available"):
        short_trend = score_trend.get("short_trend", "横盘")
        hist_pct = score_trend.get("historical_percentile", 0.5)
        trend_norm = (hist_pct - 0.5) * 2  # [-1, 1]
        if short_trend == "上升":
            trend_norm = min(trend_norm + 0.2, 1.0)
        elif short_trend == "下降":
            trend_norm = max(trend_norm - 0.2, -1.0)
        score_total += trend_norm * 20
        weight_total += 20
        signals.append({
            "source": "score_trend",
            "value": f"{short_trend}（历史分位 {hist_pct:.1%}）",
            "signal": "看多" if trend_norm > 0.1 else ("看空" if trend_norm < -0.1 else "中性"),
            "weight": 20,
        })

    # ---- 3. 技术面信号 ----
    if technical.get("available"):
        tech_score = 0.0
        tech_count = 0

        macd = technical.get("macd", {})
        if macd.get("histogram") is not None:
            tech_score += 1 if macd["histogram"] > 0 else -1
            tech_count += 1

        rsi = technical.get("rsi", {}).get("rsi14")
        if rsi is not None:
            if rsi > 70:
                tech_score -= 0.5  # 超买
            elif rsi < 30:
                tech_score += 0.5  # 超卖
            elif rsi > 50:
                tech_score += 0.3
            else:
                tech_score -= 0.3
            tech_count += 1

        ma = technical.get("moving_averages", {})
        if ma.get("ma_bullish_alignment"):
            tech_score += 1.5
            tech_count += 1
        elif ma.get("ma_bearish_alignment"):
            tech_score -= 1.5
            tech_count += 1

        bb_pos = technical.get("bollinger", {}).get("position")
        if bb_pos is not None:
            if bb_pos > 0.8:
                tech_score -= 0.5  # 接近上轨，超买
            elif bb_pos < 0.2:
                tech_score += 0.5  # 接近下轨，超卖
            elif bb_pos > 0.5:
                tech_score += 0.3
            else:
                tech_score -= 0.3
            tech_count += 1

        kdj = technical.get("kdj", {})
        k, d = kdj.get("k"), kdj.get("d")
        if k is not None and d is not None:
            if k > d and k < 80:
                tech_score += 0.5
            elif k < d and k > 20:
                tech_score -= 0.5
            tech_count += 1

        vol_ratio = technical.get("volume", {}).get("vol_ratio")
        pct_change = technical.get("price", {}).get("pct_change")
        if vol_ratio and pct_change is not None:
            if vol_ratio > 1.5 and pct_change > 0:
                tech_score += 0.5  # 放量上涨
            elif vol_ratio > 1.5 and pct_change < 0:
                tech_score -= 0.5  # 放量下跌
            tech_count += 1

        if tech_count > 0:
            tech_norm = np.clip(tech_score / (tech_count * 1.5), -1, 1)
            score_total += float(tech_norm) * 30
            weight_total += 30
            signals.append({
                "source": "technical_indicators",
                "value": round(tech_score, 2),
                "signal": "看多" if tech_norm > 0.1 else ("看空" if tech_norm < -0.1 else "中性"),
                "weight": 30,
            })

    # ---- 4. Stage-2/3 因子共识 ----
    factor_consensus = _compute_factor_consensus(stage2, stage3)
    if factor_consensus["available"]:
        fc_norm = factor_consensus["consensus_score"]  # [-1, 1]
        score_total += fc_norm * 20
        weight_total += 20
        signals.append({
            "source": "factor_consensus_stage2_stage3",
            "value": round(fc_norm, 3),
            "signal": "看多" if fc_norm > 0.1 else ("看空" if fc_norm < -0.1 else "中性"),
            "weight": 20,
        })

    # ---- 综合 ----
    if weight_total == 0:
        return {"direction": "无法判断", "confidence": 0.0, "signals": signals}

    composite = score_total / weight_total  # [-1, 1]
    confidence = round(abs(composite), 3)

    if composite > 0.15:
        direction = "看多"
    elif composite < -0.15:
        direction = "看空"
    else:
        direction = "中性"

    # 置信度分级
    if confidence >= 0.6:
        confidence_level = "高"
    elif confidence >= 0.35:
        confidence_level = "中"
    else:
        confidence_level = "低"

    return {
        "direction": direction,
        "composite_score": round(composite, 4),
        "confidence": confidence,
        "confidence_level": confidence_level,
        "signals": signals,
    }


def _compute_factor_consensus(stage2: dict, stage3: dict) -> dict:
    """
    计算 Stage-2（历史）与 Stage-3（近期）因子方向共识。
    对两个来源的 top_factors 按 direction 加权投票，返回 [-1, 1] 共识分。
    """
    all_factors: list[dict] = []

    # Stage-3 权重更高（近期有效性更强）
    for f in stage3.get("top_factors", []):
        all_factors.append({**f, "source_weight": 1.5})
    for f in stage2.get("top_factors", []):
        all_factors.append({**f, "source_weight": 1.0})

    if not all_factors:
        return {"available": False, "consensus_score": 0.0}

    weighted_sum = 0.0
    weight_total = 0.0
    for f in all_factors:
        direction = f.get("direction", 0)
        abs_ir = f.get("abs_ir", 0.01)
        w = f["source_weight"] * max(abs_ir, 0.01)
        weighted_sum += direction * w
        weight_total += w

    consensus = float(np.clip(weighted_sum / weight_total, -1, 1)) if weight_total > 0 else 0.0

    # 统计多空因子数
    bullish = sum(1 for f in all_factors if f.get("direction", 0) > 0)
    bearish = sum(1 for f in all_factors if f.get("direction", 0) < 0)

    return {
        "available": True,
        "consensus_score": round(consensus, 4),
        "bullish_factor_count": bullish,
        "bearish_factor_count": bearish,
        "total_factor_count": len(all_factors),
    }


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def _build_summary(
    stock_code: str,
    trade_date: str,
    prediction: dict,
    today_score: dict,
    score_trend: dict,
    technical: dict,
    stage2: dict,
    stage3: dict,
    stage4: dict,
    factor_consensus: dict,
) -> str:
    lines = [
        f"# Stage-5 单票当日预测报告",
        f"",
        f"- 股票: {stock_code}",
        f"- 预测基准日: {today_score.get('actual_date', trade_date)}",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"---",
        f"",
        f"## 🎯 明日方向预测",
        f"",
        f"| 维度 | 结果 |",
        f"|------|------|",
        f"| 预测方向 | **{prediction['direction']}** |",
        f"| 综合得分 | {prediction.get('composite_score', 0):.4f}（范围 -1 ~ +1） |",
        f"| 置信度 | {prediction['confidence']:.1%}（{prediction.get('confidence_level', '')}） |",
        f"",
        f"### 信号明细",
        f"",
        f"| 信号来源 | 值 | 方向 | 权重 |",
        f"|---------|-----|------|------|",
    ]
    for sig in prediction.get("signals", []):
        lines.append(
            f"| {sig['source']} | {sig['value']} | {sig['signal']} | {sig['weight']} |"
        )

    # ---- 当日评分 ----
    lines += [
        f"",
        f"---",
        f"",
        f"## 📊 当日 Temporal Score",
        f"",
        f"| 评分类型 | 分值 |",
        f"|---------|------|",
        f"| 默认 8 因子评分 | {today_score.get('default_score', 'N/A')} |",
        f"| Champion Profile 评分 | {today_score.get('profile_score', 'N/A')} |",
        f"",
    ]

    # 因子详情
    factors = today_score.get("factors", {})
    pcts = today_score.get("pcts", {})
    if factors:
        lines += [
            f"### 因子值与百分位",
            f"",
            f"| 因子 | 当日值 | 历史百分位 | 信号 |",
            f"|------|--------|-----------|------|",
        ]
        for name, val in factors.items():
            pct = pcts.get(name, 0)
            signal = "强多" if pct > 0.75 else ("弱多" if pct > 0.5 else ("弱空" if pct > 0.25 else "强空"))
            lines.append(f"| {name} | {val:.4f} | {pct:.1%} | {signal} |")

    # ---- 评分趋势 ----
    if score_trend.get("available"):
        lines += [
            f"",
            f"---",
            f"",
            f"## 📈 评分趋势（近期）",
            f"",
            f"| 统计项 | 值 |",
            f"|--------|-----|",
            f"| 当日评分 | {score_trend.get('current_score')} |",
            f"| 近5日均分 | {score_trend.get('recent_5d_avg')} |",
            f"| 近20日均分 | {score_trend.get('recent_20d_avg')} |",
            f"| 近60日均分 | {score_trend.get('recent_60d_avg')} |",
            f"| 历史最高 | {score_trend.get('score_max')} |",
            f"| 历史最低 | {score_trend.get('score_min')} |",
            f"| 历史分位 | {score_trend.get('historical_percentile', 0):.1%} |",
            f"| 短期趋势 | {score_trend.get('short_trend')} |",
            f"| 样本数 | {score_trend.get('sample_count')} |",
        ]

    # ---- 技术面快照 ----
    if technical.get("available"):
        price = technical.get("price", {})
        ma = technical.get("moving_averages", {})
        macd = technical.get("macd", {})
        rsi = technical.get("rsi", {})
        bb = technical.get("bollinger", {})
        kdj = technical.get("kdj", {})
        vol = technical.get("volume", {})
        pat = technical.get("pattern", {})

        lines += [
            f"",
            f"---",
            f"",
            f"## 🕯️ 技术面快照",
            f"",
            f"### 价格",
            f"",
            f"| 项目 | 值 |",
            f"|------|-----|",
            f"| 收盘价 | {price.get('close')} |",
            f"| 涨跌幅 | {price.get('pct_change')}% |",
            f"| 开盘 | {price.get('open')} |",
            f"| 最高 | {price.get('high')} |",
            f"| 最低 | {price.get('low')} |",
            f"",
            f"### 均线",
            f"",
            f"| 均线 | 值 | 价格位置 |",
            f"|------|-----|---------|",
        ]
        close_price = price.get("close")
        for ma_name, ma_val in [("SMA5", ma.get("sma5")), ("SMA10", ma.get("sma10")),
                                  ("SMA20", ma.get("sma20")), ("SMA60", ma.get("sma60")),
                                  ("SMA120", ma.get("sma120"))]:
            if ma_val and close_price:
                pos = "上方 ✅" if close_price > ma_val else "下方 ❌"
            else:
                pos = "N/A"
            lines.append(f"| {ma_name} | {ma_val} | {pos} |")

        ma_align = "多头排列 ✅" if ma.get("ma_bullish_alignment") else ("空头排列 ❌" if ma.get("ma_bearish_alignment") else "无明显排列")
        lines.append(f"| 均线排列 | {ma_align} | - |")

        lines += [
            f"",
            f"### 技术指标",
            f"",
            f"| 指标 | 值 | 信号 |",
            f"|------|-----|------|",
            f"| RSI(14) | {rsi.get('rsi14')} | {'超买 ⚠️' if (rsi.get('rsi14') or 0) > 70 else ('超卖 💡' if (rsi.get('rsi14') or 100) < 30 else '正常')} |",
            f"| MACD 柱 | {macd.get('histogram')} | {'金叉 ✅' if macd.get('golden_cross') else '死叉 ❌'} |",
            f"| MACD 线 | {macd.get('macd_line')} | - |",
            f"| 布林位置 | {bb.get('position')} | {'接近上轨' if (bb.get('position') or 0) > 0.8 else ('接近下轨' if (bb.get('position') or 1) < 0.2 else '中间区域')} |",
            f"| KDJ K | {kdj.get('k')} | - |",
            f"| KDJ D | {kdj.get('d')} | {'K>D 多头' if (kdj.get('k') or 0) > (kdj.get('d') or 0) else 'K<D 空头'} |",
            f"| 量比 | {vol.get('vol_ratio')} | {'放量' if (vol.get('vol_ratio') or 0) > 1.5 else ('缩量' if (vol.get('vol_ratio') or 1) < 0.7 else '正常')} |",
            f"| 20日新高 | {'是 🚀' if pat.get('is_new_high_20d') else '否'} | - |",
            f"| 20日新低 | {'是 ⚠️' if pat.get('is_new_low_20d') else '否'} | - |",
        ]

    # ---- Stage-2/3 因子共识 ----
    lines += [
        f"",
        f"---",
        f"",
        f"## 🔬 因子共识（Stage-2 历史 + Stage-3 近期）",
        f"",
    ]

    if factor_consensus.get("available"):
        cs = factor_consensus.get("consensus_score", 0)
        lines += [
            f"| 项目 | 值 |",
            f"|------|-----|",
            f"| 共识得分 | {cs:.4f}（+1=全多，-1=全空） |",
            f"| 看多因子数 | {factor_consensus.get('bullish_factor_count')} |",
            f"| 看空因子数 | {factor_consensus.get('bearish_factor_count')} |",
            f"| 总因子数 | {factor_consensus.get('total_factor_count')} |",
        ]
    else:
        lines.append("*无可用因子共识数据*")

    # Stage-2 top factors
    if stage2.get("available"):
        lines += [
            f"",
            f"### Stage-2 历史 Top 因子（{stage2.get('analysis_period', '')}）",
            f"",
            f"当前市场状态: **{stage2.get('current_regime', 'unknown')}**",
            f"",
            f"| # | 因子 | 方向 | absIC | absIR | 历史分 |",
            f"|---|------|------|-------|-------|--------|",
        ]
        for i, f in enumerate(stage2.get("top_factors", [])[:8], 1):
            dir_str = "正向 ↑" if f["direction"] >= 0 else "反向 ↓"
            lines.append(
                f"| {i} | {f['name']} | {dir_str} | {f['abs_ic']} | {f['abs_ir']} | {f['score']} |"
            )

    # Stage-3 top factors
    if stage3.get("available"):
        lines += [
            f"",
            f"### Stage-3 近期 Top 因子（{stage3.get('eval_window', '')}）",
            f"",
            f"当前市场状态: **{stage3.get('current_regime', 'unknown')}**",
            f"",
            f"| # | 因子 | 方向 | absIC | absIR | 近期分 |",
            f"|---|------|------|-------|-------|--------|",
        ]
        for i, f in enumerate(stage3.get("top_factors", [])[:8], 1):
            dir_str = "正向 ↑" if f["direction"] >= 0 else "反向 ↓"
            lines.append(
                f"| {i} | {f['name']} | {dir_str} | {f['abs_ic']} | {f['abs_ir']} | {f['score']} |"
            )

    # ---- Stage-4 冠军策略 ----
    if stage4.get("available"):
        lines += [
            f"",
            f"---",
            f"",
            f"## 🏆 Stage-4 冠军策略参考",
            f"",
            f"| 项目 | 值 |",
            f"|------|-----|",
            f"| Profile ID | `{stage4.get('profile_id')}` |",
            f"| 稳定性分 | {stage4.get('stability_score')} |",
            f"| 生效日期 | {stage4.get('effective_from')} |",
            f"| 测试期 Sharpe | {stage4.get('test_sharpe')} |",
            f"| 测试期年化收益 | {stage4.get('test_annual_return'):.1%} |",
            f"| 测试期最大回撤 | {stage4.get('test_max_drawdown'):.1%} |",
            f"| 测试期活跃比例 | {stage4.get('test_active_ratio'):.1%} |",
        ]

    lines += [
        f"",
        f"---",
        f"",
        f"*报告由 Stage-5 单票预测脚本自动生成*",
        f"",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    bootstrap_runtime()

    stock_code = normalize_stock_code(args.stock)
    trade_date = args.trade_date or datetime.now().strftime("%Y-%m-%d")

    output_dir = (
        ensure_output_dir(args.output_dir)
        if args.output_dir
        else ensure_output_dir(
            f"data/pipeline_runs/stage5_single_stock_prediction/"
            f"{stock_code}_{trade_date}"
        )
    )

    print(f"[Stage-5] 股票: {stock_code} | 预测基准日: {trade_date}")

    # 1. 加载上游结果
    stage2_path = _resolve_stage_path(args.stage2_result, stock_code, "stage2", trade_date)
    stage3_path = _resolve_stage_path(args.stage3_result, stock_code, "stage3", trade_date)
    stage4_path = _resolve_stage_path(args.stage4_result, stock_code, "stage4", trade_date)

    stage2_data = _load_json(stage2_path)
    stage3_data = _load_json(stage3_path)
    stage4_data = _load_json(stage4_path)

    print(f"[Stage-5] Stage-2: {'✓' if stage2_data else '✗'}  "
          f"Stage-3: {'✓' if stage3_data else '✗'}  "
          f"Stage-4: {'✓' if stage4_data else '✗'}")

    # 2. 提取各阶段信号
    stage2 = _extract_stage2_signals(stage2_data, stock_code)
    stage3 = _extract_stage3_signals(stage3_data, stock_code)
    stage4 = _extract_stage4_champion(stage4_data)

    profile_id: Optional[str] = stage4.get("profile_id") if stage4.get("available") else None

    # 若 champion 有 inline_profile，先持久化确保 scoring service 能找到
    if stage4.get("available") and stage4.get("inline_profile"):
        try:
            from backend.services.champion_strategy_service import _persist_inline_profile
            _persist_inline_profile(stage4["inline_profile"])
        except Exception as exc:
            logger.warning(f"持久化 inline_profile 失败: {exc}")

    # 3. 当日评分
    print("[Stage-5] 计算当日 Temporal Score...")
    today_score = _compute_today_score(stock_code, trade_date, profile_id)

    # 4. 历史评分趋势
    print("[Stage-5] 计算历史评分趋势...")
    score_trend = _compute_score_trend(stock_code, trade_date, args.lookback_days, profile_id)

    # 5. 技术面快照
    print("[Stage-5] 计算技术面快照...")
    technical = _compute_technical_snapshot(stock_code, trade_date)

    # 6. 因子共识
    factor_consensus = _compute_factor_consensus(stage2, stage3)

    # 7. 综合预测
    prediction = _synthesize_prediction(
        today_score, score_trend, technical, stage2, stage3, stage4
    )

    # 8. 组装输出
    payload = {
        "metadata": {
            "stage": "stage5_single_stock_prediction",
            "stock_code": stock_code,
            "trade_date": trade_date,
            "actual_date": today_score.get("actual_date", trade_date),
            "stage2_path": str(stage2_path) if stage2_path else None,
            "stage3_path": str(stage3_path) if stage3_path else None,
            "stage4_path": str(stage4_path) if stage4_path else None,
            "champion_profile_id": profile_id,
        },
        "prediction": prediction,
        "today_score": today_score,
        "score_trend": score_trend,
        "technical_snapshot": technical,
        "stage2_signals": stage2,
        "stage3_signals": stage3,
        "stage4_champion": stage4,
        "factor_consensus": factor_consensus,
    }

    write_json(output_dir / "result.json", payload)
    summary = _build_summary(
        stock_code, trade_date, prediction, today_score,
        score_trend, technical, stage2, stage3, stage4, factor_consensus,
    )
    write_text(output_dir / "summary.md", summary)

    # 9. 控制台输出
    print(f"\n{'='*60}")
    print(f"  {stock_code}  {today_score.get('actual_date', trade_date)}  明日预测")
    print(f"{'='*60}")
    print(f"  方向:    {prediction['direction']}")
    print(f"  置信度:  {prediction['confidence']:.1%}（{prediction.get('confidence_level', '')}）")
    print(f"  综合分:  {prediction.get('composite_score', 0):.4f}")
    print(f"  默认评分: {today_score.get('default_score', 'N/A')}")
    if today_score.get("profile_score") is not None:
        print(f"  Profile评分: {today_score['profile_score']}")
    if score_trend.get("available"):
        print(f"  评分趋势: {score_trend['short_trend']}（历史分位 {score_trend['historical_percentile']:.1%}）")
    if factor_consensus.get("available"):
        print(f"  因子共识: {factor_consensus['consensus_score']:.4f}"
              f"（多{factor_consensus['bullish_factor_count']} 空{factor_consensus['bearish_factor_count']}）")
    print(f"{'='*60}")
    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":
    main()
