"""
因子导入服务 - 外部因子值导入预校验与模板
"""
import pandas as pd
from typing import Dict

IMPORT_DISABLED_MESSAGE = (
    "导入工具当前未开放“外部因子值直接入库执行”能力。"
    "系统仅支持可直接执行的麦语言或 Python 因子表达式；"
    "为避免生成不可执行条目，CSV / DataFrame 导入已禁用。"
)


class FactorImportService:
    """因子导入服务类。当前仅保留模板与格式校验能力。"""

    def __init__(self):
        pass

    def import_from_csv(
        self,
        csv_file_path: str,
        factor_name: str,
        description: str = "",
        category: str = "导入",
        date_column: str = "date",
        factor_column: str = "factor_value",
    ) -> Dict:
        """
        从CSV文件导入因子

        Args:
            csv_file_path: CSV文件路径
            factor_name: 因子名称
            description: 因子描述
            category: 因子分类
            date_column: 日期列名
            factor_column: 因子值列名

        Returns:
            导入结果信息
        """
        try:
            raise RuntimeError(IMPORT_DISABLED_MESSAGE)

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": str(e),
            }

    def import_from_dataframe(
        self,
        df: pd.DataFrame,
        factor_name: str,
        description: str = "",
        category: str = "导入",
        date_column: str = "date",
        factor_column: str = "factor_value",
    ) -> Dict:
        """
        从DataFrame导入因子

        Args:
            df: 包含日期和因子值的DataFrame
            factor_name: 因子名称
            description: 因子描述
            category: 因子分类
            date_column: 日期列名
            factor_column: 因子值列名

        Returns:
            导入结果信息
        """
        try:
            raise RuntimeError(IMPORT_DISABLED_MESSAGE)

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": str(e),
            }

    def validate_csv_format(
        self,
        csv_file_path: str,
        date_column: str = "date",
        factor_column: str = "factor_value",
    ) -> Dict:
        """
        验证CSV文件格式

        Args:
            csv_file_path: CSV文件路径
            date_column: 期望的日期列名
            factor_column: 期望的因子值列名

        Returns:
            验证结果
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file_path)

            result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "info": {},
            }

            # 检查列
            if date_column not in df.columns:
                result["valid"] = False
                result["errors"].append(f"缺少日期列: {date_column}")

            if factor_column not in df.columns:
                result["valid"] = False
                result["errors"].append(f"缺少因子值列: {factor_column}")

            # 检查数据
            if len(df) == 0:
                result["valid"] = False
                result["errors"].append("CSV文件为空")

            # 检查空值
            if date_column in df.columns:
                null_count = df[date_column].isnull().sum()
                if null_count > 0:
                    result["warnings"].append(f"日期列有 {null_count} 个空值")

            if factor_column in df.columns:
                null_count = df[factor_column].isnull().sum()
                if null_count > 0:
                    result["warnings"].append(f"因子值列有 {null_count} 个空值")

                # 检查是否为数值类型
                try:
                    pd.to_numeric(df[factor_column], errors="coerce")
                except Exception:
                    result["warnings"].append("因子值列包含非数值数据")

            # 添加基本信息
            result["info"] = {
                "row_count": len(df),
                "columns": list(df.columns),
                "import_enabled": False,
                "import_notice": IMPORT_DISABLED_MESSAGE,
            }

            return result

        except Exception as e:
            return {
                "valid": False,
                "errors": [f"文件读取失败: {e}"],
                "warnings": [],
                "info": {},
            }

    def get_import_template(self) -> pd.DataFrame:
        """
        获取导入模板

        Returns:
            示例DataFrame
        """
        import pandas as pd
        from datetime import datetime, timedelta

        # 创建示例数据
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        factor_values = [0.5 + i * 0.1 for i in range(10)]

        df = pd.DataFrame({
            "date": dates,
            "factor_value": factor_values,
        })

        return df


# 全局因子导入服务实例
factor_import_service = FactorImportService()
