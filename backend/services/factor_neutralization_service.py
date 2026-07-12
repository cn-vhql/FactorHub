"""
因子中性化服务 - 市值中性化和行业中性化
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.linear_model import LinearRegression


class FactorNeutralizationService:
    """因子中性化服务类"""

    def __init__(self):
        self._industry_cache: Dict[str, str] = {}

    def neutralize_market_cap(
        self,
        df: pd.DataFrame,
        factor_name: str,
        market_cap_column: str = "market_cap"
    ) -> pd.Series:
        """
        市值中性化 - 使用线性回归去除市值影响

        Args:
            df: 包含因子值和市值的数据框
            factor_name: 因子列名
            market_cap_column: 市值列名

        Returns:
            中性化后的因子值（回归残差）
        """
        if market_cap_column not in df.columns:
            raise ValueError(f"数据框中缺少市值列: {market_cap_column}")

        if factor_name not in df.columns:
            raise ValueError(f"数据框中缺少因子列: {factor_name}")

        # 移除缺失值
        valid_data = df[[factor_name, market_cap_column]].dropna()

        if len(valid_data) < 10:
            raise ValueError("有效数据不足，无法进行中性化")

        # 对市值取对数
        log_market_cap = np.log(valid_data[market_cap_column].replace(0, np.nan))
        log_market_cap = log_market_cap.fillna(log_market_cap.mean())

        # 线性回归
        model = LinearRegression()
        X = log_market_cap.values.reshape(-1, 1)
        y = valid_data[factor_name].values

        model.fit(X, y)

        # 计算残差（中性化后的因子值）
        residual = y - model.predict(X)

        # 创建返回的Series，保持原索引
        result = pd.Series(index=df.index, dtype=float)
        result.loc[valid_data.index] = residual

        return result

    def neutralize_industry(
        self,
        df: pd.DataFrame,
        factor_name: str,
        industry_column: str = "industry"
    ) -> pd.Series:
        """
        行业中性化 - 在行业内标准化因子

        Args:
            df: 包含因子值和行业分类的数据框
            factor_name: 因子列名
            industry_column: 行业分类列名

        Returns:
            行业中性化后的因子值
        """
        if industry_column not in df.columns:
            raise ValueError(f"数据框中缺少行业列: {industry_column}")

        if factor_name not in df.columns:
            raise ValueError(f"数据框中缺少因子列: {factor_name}")

        # 创建结果Series
        result = pd.Series(index=df.index, dtype=float)

        # 按行业分组，每组内标准化
        for industry, group in df.groupby(industry_column):
            factor_values = group[factor_name]

            # 计算行业内的均值和标准差
            industry_mean = factor_values.mean()
            industry_std = factor_values.std()

            if industry_std > 0:
                # 标准化
                normalized = (factor_values - industry_mean) / industry_std
            else:
                # 如果标准差为0，直接使用原值
                normalized = factor_values - industry_mean

            result.loc[group.index] = normalized

        return result

    def neutralize_both(
        self,
        df: pd.DataFrame,
        factor_name: str,
        market_cap_column: str = "market_cap",
        industry_column: str = "industry"
    ) -> pd.Series:
        """
        同时进行市值和行业中性化

        Args:
            df: 数据框
            factor_name: 因子列名
            market_cap_column: 市值列名
            industry_column: 行业列名

        Returns:
            双重中性化后的因子值
        """
        # 先进行市值中性化
        market_cap_neutralized = self.neutralize_market_cap(
            df, factor_name, market_cap_column
        )

        # 创建临时数据框用于行业中性化
        temp_df = df.copy()
        temp_df[f"{factor_name}_mc_neutral"] = market_cap_neutralized

        # 再进行行业中性化
        industry_neutralized = self.neutralize_industry(
            temp_df, f"{factor_name}_mc_neutral", industry_column
        )

        return industry_neutralized

    def get_industry_classification(self, stock_codes: List[str]) -> Dict[str, str]:
        """
        获取股票的真实行业分类

        Args:
            stock_codes: 股票代码列表

        Returns:
            股票代码到行业的映射字典
        """
        normalized_pairs = [
            (code, self._normalize_stock_code(code))
            for code in stock_codes
        ]
        pending_codes = {
            normalized
            for _, normalized in normalized_pairs
            if normalized not in self._industry_cache
        }

        if pending_codes:
            self._hydrate_industry_cache(pending_codes)

        missing_codes = sorted(
            normalized
            for _, normalized in normalized_pairs
            if normalized not in self._industry_cache
        )
        if missing_codes:
            missing_display = ", ".join(missing_codes[:10])
            raise ValueError(
                f"无法从真实数据源获取行业分类: {missing_display}"
            )

        return {
            original_code: self._industry_cache[normalized_code]
            for original_code, normalized_code in normalized_pairs
        }

    def add_industry_classification(
        self,
        df: pd.DataFrame,
        stock_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        为数据框添加行业分类列

        Args:
            df: 数据框
            stock_codes: 股票代码列表

        Returns:
            添加了industry列的数据框
        """
        result = df.copy()
        if "stock_code" in df.columns:
            codes = result["stock_code"].astype(str).tolist()
            industry_map = self.get_industry_classification(codes)
            result["industry"] = result["stock_code"].astype(str).map(industry_map)
        elif isinstance(df.index, pd.MultiIndex) and "stock_code" in df.index.names:
            codes = df.index.get_level_values("stock_code").astype(str).tolist()
            industry_map = self.get_industry_classification(codes)
            result["industry"] = df.index.get_level_values("stock_code").astype(str).map(industry_map)
        elif stock_codes and len({self._normalize_stock_code(code) for code in stock_codes}) == 1:
            industry_map = self.get_industry_classification(stock_codes)
            result["industry"] = industry_map[stock_codes[0]]
        else:
            raise ValueError("缺少股票代码信息，无法添加真实行业分类")

        if result["industry"].isna().any():
            missing_codes = result.loc[result["industry"].isna(), "stock_code"].astype(str).unique().tolist() if "stock_code" in result.columns else []
            raise ValueError(f"以下股票缺少行业分类: {missing_codes}")

        return result

    def _normalize_stock_code(self, code: str) -> str:
        """标准化股票代码为 6 位数字格式。"""
        code = str(code).strip().upper()
        if "." in code:
            code = code.split(".")[0]
        return code.zfill(6) if code.isdigit() else code

    def _hydrate_industry_cache(self, pending_codes: set[str]) -> None:
        """使用真实数据源批量或逐只补齐行业映射缓存。"""
        for code in pending_codes:
            industry = self._load_industry_from_cninfo(code)
            if industry:
                self._industry_cache[code] = industry

        unresolved = [code for code in pending_codes if code not in self._industry_cache]
        if unresolved:
            self._load_industry_map_from_spot(set(unresolved))

        unresolved = [code for code in pending_codes if code not in self._industry_cache]
        for code in unresolved:
            industry = self._load_industry_from_individual_info(code)
            if industry:
                self._industry_cache[code] = industry

    def _load_industry_from_cninfo(self, stock_code: str) -> Optional[str]:
        """从巨潮行业变更接口获取最新有效行业分类。"""
        try:
            import akshare as ak
        except ImportError as exc:
            raise ValueError("AkShare 未安装，无法获取真实行业分类") from exc

        try:
            info_df = ak.stock_industry_change_cninfo(
                symbol=stock_code,
                start_date="20000101",
                end_date=pd.Timestamp.today().strftime("%Y%m%d"),
            )
        except Exception:
            return None

        if info_df is None or info_df.empty:
            return None

        info_df = info_df.copy()
        if "变更日期" in info_df.columns:
            info_df["变更日期"] = pd.to_datetime(info_df["变更日期"], errors="coerce")

        preferred_standards = [
            "巨潮行业分类标准",
            "申银万国行业分类标准",
            "中证行业分类标准",
            "中国上市公司协会上市公司行业分类标准",
        ]

        for standard in preferred_standards:
            if "分类标准" not in info_df.columns:
                break
            filtered = info_df[info_df["分类标准"].astype(str).str.contains(standard, na=False)]
            industry = self._extract_latest_industry_name(filtered)
            if industry:
                return industry

        return self._extract_latest_industry_name(info_df)

    def _extract_latest_industry_name(self, df: pd.DataFrame) -> Optional[str]:
        """从行业变更记录中提取最新有效行业名称。"""
        if df is None or df.empty:
            return None

        sorted_df = df.sort_values("变更日期", ascending=False) if "变更日期" in df.columns else df
        for _, row in sorted_df.iterrows():
            for column in ["行业大类", "行业中类", "行业次类", "行业门类"]:
                value = row.get(column)
                if pd.notna(value) and str(value).strip():
                    return str(value).strip()
        return None

    def _load_industry_map_from_spot(self, pending_codes: set[str]) -> None:
        """从全市场实时行情中提取行业字段。"""
        try:
            import akshare as ak
        except ImportError as exc:
            raise ValueError("AkShare 未安装，无法获取真实行业分类") from exc

        try:
            spot_df = ak.stock_zh_a_spot_em()
        except Exception:
            return

        if spot_df is None or spot_df.empty:
            return

        code_column = self._find_first_existing_column(
            spot_df,
            ["代码", "股票代码", "证券代码", "symbol", "Symbol"],
        )
        industry_column = self._find_first_existing_column(
            spot_df,
            ["行业", "所属行业", "所处行业", "industry", "Industry"],
        )

        if code_column is None or industry_column is None:
            return

        mapped = spot_df[[code_column, industry_column]].dropna()
        mapped[code_column] = mapped[code_column].astype(str).map(self._normalize_stock_code)
        mapped[industry_column] = mapped[industry_column].astype(str).str.strip()

        valid_rows = mapped[
            mapped[code_column].isin(pending_codes) &
            mapped[industry_column].ne("")
        ]

        for _, row in valid_rows.iterrows():
            self._industry_cache[row[code_column]] = row[industry_column]

    def _load_industry_from_individual_info(self, stock_code: str) -> Optional[str]:
        """从个股资料接口补齐行业字段。"""
        try:
            import akshare as ak
        except ImportError as exc:
            raise ValueError("AkShare 未安装，无法获取真实行业分类") from exc

        try:
            info_df = ak.stock_individual_info_em(symbol=stock_code)
        except Exception:
            return None

        if info_df is None or info_df.empty:
            return None

        item_column = self._find_first_existing_column(info_df, ["item", "项目", "Item"])
        value_column = self._find_first_existing_column(info_df, ["value", "值", "Value"])
        if item_column is None or value_column is None:
            return None

        info_df[item_column] = info_df[item_column].astype(str).str.strip()
        mask = info_df[item_column].str.contains("行业", na=False)
        if not mask.any():
            return None

        industry_value = info_df.loc[mask, value_column].dropna().astype(str).str.strip()
        if industry_value.empty:
            return None

        return industry_value.iloc[0]

    def _find_first_existing_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """返回候选列名中第一个存在的列。"""
        for column in candidates:
            if column in df.columns:
                return column
        return None


# 全局因子中性化服务实例
factor_neutralization_service = FactorNeutralizationService()
