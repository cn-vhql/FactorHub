"""
因子分析服务模块 - IC/IR统计、SHAP分析
"""
import hashlib
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import json

# 配置日志
logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.preprocessing import StandardScaler

from backend.core.settings import settings
from backend.core.database import get_db_session
from backend.repositories.factor_repository import AnalysisCacheRepository
from backend.models.factor import AnalysisCacheModel
from backend.services.factor_service import factor_service


class AnalysisService:
    """因子分析服务类"""

    def __init__(self):
        self.results_cache = {}

    def _serialize_for_cache(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """将结果序列化为可JSON序列化的格式"""
        serialized = {
            "metadata": results["metadata"],
            "ic_ir": {},
            "shap": results.get("shap", {}),
        }

        # 序列化 IC/IR 结果
        if "ic_ir" in results:
            ic_ir = results["ic_ir"]
            if "ic_stats" in ic_ir:
                # 需要将 IC序列 中的 Timestamp 键转换为字符串
                serialized_stats = {}
                for factor_name, stats in ic_ir["ic_stats"].items():
                    serialized_stats[factor_name] = {}
                    for key, value in stats.items():
                        if key == "IC序列" and isinstance(value, dict):
                            # 将 Timestamp 键转换为字符串
                            serialized_stats[factor_name][key] = {
                                str(k) if hasattr(k, 'isoformat') else k: v
                                for k, v in value.items()
                            }
                        else:
                            serialized_stats[factor_name][key] = value
                serialized["ic_ir"]["ic_stats"] = serialized_stats

            if "monthly_ic" in ic_ir:
                serialized["ic_ir"]["monthly_ic"] = {
                    k: v.to_dict() if hasattr(v, 'to_dict') else v
                    for k, v in ic_ir["monthly_ic"].items()
                }

            if "rolling_ir" in ic_ir:
                serialized["ic_ir"]["rolling_ir"] = {
                    k: (
                        {str(idx): val for idx, val in v.to_dict().items()} if hasattr(v, 'to_dict')
                        else {str(idx): val for idx, val in dict(enumerate(v.tolist())).items()} if hasattr(v, 'tolist')
                        else v
                    )
                    for k, v in ic_ir["rolling_ir"].items()
                }

        return serialized

    def _deserialize_from_cache(self, cached_data: Dict[str, Any], factor_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """从缓存数据反序列化"""
        # 将 factor_data 添加到结果中
        result = {
            "metadata": cached_data["metadata"],
            "factor_data": factor_data,
            "ic_ir": cached_data.get("ic_ir", {}),
            "shap": cached_data.get("shap", {}),
        }

        # 将 monthly_ic 转回 DataFrame
        if "ic_ir" in result and "monthly_ic" in result["ic_ir"]:
            result["ic_ir"]["monthly_ic"] = {
                k: pd.DataFrame(v) if isinstance(v, dict) else v
                for k, v in result["ic_ir"]["monthly_ic"].items()
            }

        # 将 rolling_ir 转回 Series
        if "ic_ir" in result and "rolling_ir" in result["ic_ir"]:
            result["ic_ir"]["rolling_ir"] = {}
            for k, v in result["ic_ir"]["rolling_ir"].items():
                if isinstance(v, dict):
                    # 尝试解析字符串索引为 datetime
                    try:
                        result["ic_ir"]["rolling_ir"][k] = pd.Series(
                            v.values(),
                            index=pd.to_datetime(list(v.keys()))
                        )
                    except Exception as e:
                        # 如果失败，直接使用字符串索引
                        import logging
                        logging.getLogger(__name__).debug(f"日期解析失败，使用字符串索引: {e}")
                        result["ic_ir"]["rolling_ir"][k] = pd.Series(v)
                else:
                    result["ic_ir"]["rolling_ir"][k] = v

        return result

    def _get_factor_cache_signature(self, factor_names: List[str]) -> str:
        """生成因子定义签名，避免同名因子改版后复用旧缓存。"""
        db = get_db_session()
        try:
            from backend.repositories.factor_repository import FactorRepository

            repo = FactorRepository(db)
            signature_parts = []
            for factor_name in sorted(factor_names):
                factor = repo.get_by_name(factor_name)
                if factor is None:
                    signature_parts.append(f"{factor_name}:missing")
                    continue

                code_hash = hashlib.md5(factor.code.encode("utf-8")).hexdigest()[:12]
                updated_at = factor.updated_at.isoformat() if factor.updated_at else "unknown"
                signature_parts.append(f"{factor.name}:{code_hash}:{updated_at}")

            return "|".join(signature_parts)
        finally:
            db.close()

    def _generate_cache_key(
        self,
        stock_codes: List[str],
        factor_names: List[str],
        start_date: str,
        end_date: str,
        rolling_window: int,
        factor_signature: str,
    ) -> str:
        """生成缓存键"""
        key_str = (
            f"{','.join(sorted(stock_codes))}_"
            f"{','.join(sorted(factor_names))}_"
            f"{start_date}_{end_date}_{rolling_window}_{factor_signature}"
        )
        return hashlib.md5(key_str.encode()).hexdigest()[:32]

    def analyze(
        self,
        stock_codes: List[str],
        factor_names: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        rolling_window: int = 252,
    ) -> Dict[str, Any]:
        """
        执行完整的因子分析

        Args:
            stock_codes: 股票代码列表
            factor_names: 因子名称列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            rolling_window: 滚动窗口大小

        Returns:
            包含所有分析结果的字典
        """
        factor_signature = self._get_factor_cache_signature(factor_names)
        cache_key = self._generate_cache_key(
            stock_codes,
            factor_names,
            start_date,
            end_date,
            rolling_window,
            factor_signature,
        )

        # 计算因子数据（始终需要，因为缓存不包含原始数据）
        factor_data = factor_service.calculate_factors_for_stocks(
            stock_codes, factor_names, start_date, end_date, rolling_window
        )

        if not factor_data:
            raise ValueError("未能获取任何有效的因子数据")

        # 检查缓存（仅获取分析结果，factor_data 已重新计算）
        if use_cache:
            db = get_db_session()
            repo = AnalysisCacheRepository(db)
            cached = repo.get_by_key(cache_key)
            if cached:
                db.close()
                # 从缓存反序列化并合并当前的 factor_data
                return self._deserialize_from_cache(cached.result_data, factor_data)
            db.close()

        # 执行各种分析
        results = {
            "metadata": {
                "stock_codes": stock_codes,
                "factor_names": factor_names,
                "start_date": start_date,
                "end_date": end_date,
                "rolling_window": rolling_window,
                "analysis_time": datetime.now().isoformat(),
            },
            "factor_data": factor_data,
        }

        # IC/IR 分析
        ic_ir_results = self.calculate_ic_ir(factor_data, factor_names)
        results["ic_ir"] = ic_ir_results

        # SHAP 分析
        if SHAP_AVAILABLE:
            shap_results = self.calculate_shap(factor_data, factor_names)
            results["shap"] = shap_results
        else:
            results["shap"] = {"error": "SHAP not available"}

        # 保存到缓存（不包含原始 factor_data，因为数据量太大）
        if use_cache:
            db = get_db_session()
            repo = AnalysisCacheRepository(db)
            # 序列化结果（移除 factor_data）
            serialized_results = self._serialize_for_cache(results)
            existing_cache = repo.get_by_key(cache_key)
            if existing_cache:
                existing_cache.stock_codes = ",".join(stock_codes)
                existing_cache.factor_names = ",".join(factor_names)
                existing_cache.start_date = start_date
                existing_cache.end_date = end_date
                existing_cache.result_data = serialized_results
                repo.update(existing_cache)
            else:
                cache = AnalysisCacheModel(
                    cache_key=cache_key,
                    stock_codes=",".join(stock_codes),
                    factor_names=",".join(factor_names),
                    start_date=start_date,
                    end_date=end_date,
                    result_data=serialized_results,
                )
                repo.create(cache)
            db.close()

        return results

    def calculate_ic_ir(
        self, factor_data: Dict[str, pd.DataFrame], factor_names: List[str]
    ) -> Dict[str, Any]:
        """
        计算IC和IR统计

        Args:
            factor_data: 股票代码到因子数据的映射
            factor_names: 因子名称列表

        Returns:
            IC/IR统计结果
        """
        # 计算未来收益率（用于计算IC）
        for stock_code in factor_data.keys():
            factor_data[stock_code]["future_return_1"] = factor_data[stock_code]["close"].pct_change(1).shift(-1)
            factor_data[stock_code]["future_return_5"] = factor_data[stock_code]["close"].pct_change(5).shift(-5)

            # 清理无穷大值和NaN值
            for col in factor_data[stock_code].columns:
                factor_data[stock_code][col] = factor_data[stock_code][col].replace([np.inf, -np.inf], np.nan)

        # 判断是单股票还是多股票
        is_single_stock = len(factor_data) == 1

        if is_single_stock:
            # 单股票：计算时序IC（因子值与自身未来收益率的滚动相关性）
            return self._calculate_single_stock_ic(factor_data, factor_names)
        else:
            # 多股票：计算横截面IC
            return self._calculate_cross_sectional_ic(factor_data, factor_names)

    def _calculate_single_stock_ic(
        self, factor_data: Dict[str, pd.DataFrame], factor_names: List[str]
    ) -> Dict[str, Any]:
        """单股票时序IC计算"""
        stock_code = list(factor_data.keys())[0]
        df = factor_data[stock_code]

        ic_series = {}
        for factor_name in factor_names:
            if factor_name not in df.columns:
                continue

            # 计算滚动窗口的时序IC（因子值与未来收益率的相关性）
            factor_values = df[factor_name]
            return_values = df["future_return_1"]

            # 移除NaN和无穷大值
            valid_mask = (
                factor_values.notna() &
                return_values.notna() &
                ~np.isinf(factor_values) &
                ~np.isinf(return_values)
            )
            factor_clean = factor_values[valid_mask]
            return_clean = return_values[valid_mask]

            if len(factor_clean) < 20:  # 至少需要20个数据点
                continue

            # 计算滚动相关性（窗口大小20）
            rolling_ic = factor_clean.rolling(window=20).corr(return_clean)

            # 移除NaN和无穷大值
            rolling_ic = rolling_ic.replace([np.inf, -np.inf], np.nan).dropna()

            if len(rolling_ic) > 0:
                ic_series[factor_name] = rolling_ic

        # 计算IC统计指标
        ic_stats = {}
        for factor_name, ic_s in ic_series.items():
            ic_mean = ic_s.mean()
            ic_std = ic_s.std()
            ir = ic_mean / ic_std if ic_std != 0 else 0

            ic_stats[factor_name] = {
                "IC均值": ic_mean,
                "IC标准差": ic_std,
                "IR": ir,
                "IC>0占比": (ic_s > 0).mean(),
                "IC绝对值均值": abs(ic_s).mean(),
                "IC序列": ic_s.to_dict(),
                "IC类型": "时序IC（单股票）",
            }

        # 月度IC热力图数据
        monthly_ic = self._calculate_monthly_ic(ic_series)

        # 滚动窗口IR（使用更小的窗口）
        rolling_ir = self._calculate_rolling_ir(ic_series, window=20)

        return {
            "ic_stats": ic_stats,
            "monthly_ic": monthly_ic,
            "rolling_ir": rolling_ir,
        }

    def _calculate_cross_sectional_ic(
        self, factor_data: Dict[str, pd.DataFrame], factor_names: List[str]
    ) -> Dict[str, Any]:
        """多股票横截面IC计算"""
        # 合并所有股票数据用于横截面IC计算
        all_data = []
        for stock_code, df in factor_data.items():
            stock_df = df.copy()
            stock_df["stock_code"] = stock_code
            all_data.append(stock_df)

        merged_df = pd.concat(all_data, ignore_index=False)

        # 计算IC时间序列
        ic_series = {}
        for factor_name in factor_names:
            ics = []
            dates = []

            # 按日期横截面计算IC
            for date in merged_df.index.unique():
                date_data = merged_df.loc[[date]]
                if len(date_data) < 2 or factor_name not in date_data.columns:
                    continue

                # 计算未来收益率（需要与原数据对应）
                factor_values = []
                return_values = []
                for stock_code in date_data["stock_code"]:
                    if stock_code in factor_data and date in factor_data[stock_code].index:
                        factor_val = factor_data[stock_code].loc[date, factor_name]
                        return_val = factor_data[stock_code].loc[date, "future_return_1"]
                        # 检查非NaN且非无穷大
                        if pd.notna(factor_val) and pd.notna(return_val) and not np.isinf(factor_val) and not np.isinf(return_val):
                            factor_values.append(factor_val)
                            return_values.append(return_val)

                if len(factor_values) >= 2:
                    ic = pd.Series(factor_values).corr(pd.Series(return_values))
                    if pd.notna(ic) and not np.isinf(ic):
                        ics.append(ic)
                        dates.append(date)

            if ics:
                ic_series[factor_name] = pd.Series(ics, index=dates)

        # 计算IC统计指标
        ic_stats = {}
        for factor_name, ic_s in ic_series.items():
            ic_mean = ic_s.mean()
            ic_std = ic_s.std()
            ir = ic_mean / ic_std if ic_std != 0 else 0

            ic_stats[factor_name] = {
                "IC均值": ic_mean,
                "IC标准差": ic_std,
                "IR": ir,
                "IC>0占比": (ic_s > 0).mean(),
                "IC绝对值均值": abs(ic_s).mean(),
                "IC序列": ic_s.to_dict(),
                "IC类型": "横截面IC（多股票）",
            }

        # 月度IC热力图数据
        monthly_ic = self._calculate_monthly_ic(ic_series)

        # 滚动窗口IR
        rolling_ir = self._calculate_rolling_ir(ic_series, window=60)

        return {
            "ic_stats": ic_stats,
            "monthly_ic": monthly_ic,
            "rolling_ir": rolling_ir,
        }

    def _calculate_monthly_ic(
        self, ic_series: Dict[str, pd.Series]
    ) -> Dict[str, pd.DataFrame]:
        """计算月度IC热力图数据"""
        monthly_ic = {}
        for factor_name, ic_s in ic_series.items():
            # 按年月分组计算平均IC
            ic_df = pd.DataFrame({"ic": ic_s})
            ic_df["year"] = ic_df.index.year
            ic_df["month"] = ic_df.index.month

            pivot = ic_df.pivot_table(values="ic", index="year", columns="month", aggfunc="mean")
            monthly_ic[factor_name] = pivot
        return monthly_ic

    def _calculate_rolling_ir(
        self, ic_series: Dict[str, pd.Series], window: int = 60
    ) -> Dict[str, pd.Series]:
        """计算滚动窗口IR"""
        rolling_ir = {}
        for factor_name, ic_s in ic_series.items():
            # 使用较小的 min_periods 以便在数据不足时也能产生一些结果
            min_periods = max(1, window // 4)
            rolling_mean = ic_s.rolling(window=window, min_periods=min_periods).mean()
            rolling_std = ic_s.rolling(window=window, min_periods=min_periods).std()
            ir = rolling_mean / rolling_std
            rolling_ir[factor_name] = ir
        return rolling_ir

    def calculate_shap(
        self, factor_data: Dict[str, pd.DataFrame], factor_names: List[str]
    ) -> Dict[str, Any]:
        """
        计算SHAP值分析

        Args:
            factor_data: 因子数据
            factor_names: 因子名称列表

        Returns:
            SHAP分析结果
        """
        if not SHAP_AVAILABLE:
            return {"error": "SHAP library not installed"}

        logger.debug(f"[SHAP] Starting SHAP analysis")
        logger.debug(f"[SHAP] factor_names: {factor_names}")
        logger.debug(f"[SHAP] Number of stocks in factor_data: {len(factor_data)}")

        # 准备训练数据
        X_list = []
        y_list = []

        for stock_code, df in factor_data.items():
            logger.debug(f"[SHAP] Processing stock: {stock_code}")
            logger.debug(f"[SHAP]   DataFrame columns: {df.columns.tolist()}")
            logger.debug(f"[SHAP]   DataFrame shape: {df.shape}")

            if "future_return_5" not in df.columns:
                df["future_return_5"] = df["close"].pct_change(5).shift(-5)

            # 提取特征列
            feature_cols = [col for col in factor_names if col in df.columns]
            logger.debug(f"[SHAP]   Available feature_cols: {feature_cols}")

            if not feature_cols:
                logger.debug(f"[SHAP]   No feature columns found, skipping")
                continue

            X = df[feature_cols].dropna()
            y = df.loc[X.index, "future_return_5"]

            logger.debug(f"[SHAP]   X shape before NaN removal: {X.shape}")
            logger.debug(f"[SHAP]   X NaN count: {X.isna().sum().sum()}")

            # 移除NaN
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]

            logger.debug(f"[SHAP]   X_valid shape: {X_valid.shape}")

            if len(X_valid) > 0:
                X_list.append(X_valid)
                y_list.append(y_valid)
                logger.debug(f"[SHAP]   Added {len(X_valid)} valid samples")
            else:
                logger.debug(f"[SHAP]   No valid samples after NaN removal")

        logger.debug(f"[SHAP] Total X_list length: {len(X_list)}")

        if not X_list:
            logger.error("[SHAP] No valid data for SHAP analysis")
            return {"error": "No valid data for SHAP analysis"}

        # 合并所有数据
        X_combined = pd.concat(X_list, ignore_index=True)
        y_combined = pd.concat(y_list, ignore_index=True)

        logger.debug(f"[SHAP] X_combined shape: {X_combined.shape}")
        logger.debug(f"[SHAP] X_combined columns: {X_combined.columns.tolist()}")

        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        X_scaled = pd.DataFrame(X_scaled, columns=X_combined.columns)

        # 训练XGBoost模型
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="reg:squarederror",
            random_state=42,
        )

        # 分割训练集和测试集
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_combined[:split_idx], y_combined[split_idx:]

        logger.debug(f"[SHAP] Training with {len(X_train)} samples, testing with {len(X_test)} samples")

        model.fit(X_train, y_train)

        # 计算SHAP值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # 特征重要性
        feature_importance = pd.DataFrame({
            "feature": X_test.columns,
            "importance": np.abs(shap_values).mean(axis=0),
        }).sort_values("importance", ascending=False)

        logger.debug(f"[SHAP] SHAP analysis completed successfully")

        return {
            "feature_importance": feature_importance.to_dict("records"),
            "shap_values": shap_values.tolist(),
            "feature_names": X_test.columns.tolist(),
            "model_score": model.score(X_test, y_test),
        }

    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        生成分析报告（Markdown格式）

        Args:
            analysis_results: 分析结果字典

        Returns:
            Markdown格式的报告文本
        """
        metadata = analysis_results["metadata"]
        ic_ir = analysis_results.get("ic_ir", {})
        shap_data = analysis_results.get("shap", {})

        report = f"""# 因子分析报告

## 分析参数

- **股票代码**: {', '.join(metadata['stock_codes'])}
- **因子列表**: {', '.join(metadata['factor_names'])}
- **时间区间**: {metadata['start_date']} 至 {metadata['end_date']}
- **滚动窗口**: {metadata['rolling_window']}天
- **分析时间**: {metadata['analysis_time']}

---

## IC/IR 统计分析

"""

        if "ic_stats" in ic_ir:
            ic_stats = ic_ir["ic_stats"]
            report += "### 因子IC统计\n\n"
            report += "| 因子名称 | IC均值 | IC标准差 | IR | IC>0占比 | IC绝对值均值 |\n"
            report += "|---------|--------|----------|-----|---------|-------------|\n"

            for factor_name, stats in ic_stats.items():
                report += f"| {factor_name} | {stats['IC均值']:.4f} | {stats['IC标准差']:.4f} | {stats['IR']:.4f} | {stats['IC>0占比']:.2%} | {stats['IC绝对值均值']:.4f} |\n"

            report += "\n"

        if shap_data and "feature_importance" in shap_data:
            report += "---\n\n## SHAP 特征重要性分析\n\n"
            report += "### 全局特征重要性排序\n\n"
            report += "| 排名 | 特征名称 | 重要性 |\n"
            report += "|------|---------|--------|\n"

            for i, feat in enumerate(shap_data["feature_importance"], 1):
                report += f"| {i} | {feat['feature']} | {feat['importance']:.6f} |\n"

            report += f"\n**模型R²得分**: {shap_data.get('model_score', 0):.4f}\n\n"

        report += "---\n\n*报告由 FactorFlow 自动生成*"

        return report

    def export_report(self, analysis_results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        导出分析报告到文件

        Args:
            analysis_results: 分析结果
            output_path: 输出路径，默认保存到reports目录

        Returns:
            保存的文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = settings.REPORTS_DIR / f"factor_analysis_report_{timestamp}.md"

        report = self.generate_report(analysis_results)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        return str(output_path)


# 全局分析服务实例
analysis_service = AnalysisService()
