"""
自定义因子管理
"""
import pandas as pd
import numpy as np
import importlib.util
import ast
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from utils.logger import logger
from utils.config import FACTORS_DIR
from utils.helpers import save_to_cache, load_from_cache

class CustomFactor:
    """自定义因子类"""

    def __init__(self,
                 name: str,
                 description: str = "",
                 category: str = "custom",
                 code: str = "",
                 created_at: datetime = None):
        self.name = name
        self.description = description
        self.category = category
        self.code = code
        self.created_at = created_at or datetime.now()
        self.version = 1
        self.enabled = True

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "code": self.code,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "enabled": self.enabled
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CustomFactor':
        """从字典创建"""
        factor = cls(
            name=data["name"],
            description=data.get("description", ""),
            category=data.get("category", "custom"),
            code=data.get("code", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now()
        )
        factor.version = data.get("version", 1)
        factor.enabled = data.get("enabled", True)
        return factor

class CustomFactorManager:
    """自定义因子管理器"""

    def __init__(self):
        self.logger = logger
        self.custom_factors = {}
        self.storage_file = FACTORS_DIR / "custom_factors.json"
        self._load_factors()

    def _load_factors(self):
        """加载已保存的因子"""
        try:
            import json
            if self.storage_file.exists():
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for factor_data in data:
                        factor = CustomFactor.from_dict(factor_data)
                        self.custom_factors[factor.name] = factor
                self.logger.info(f"加载自定义因子: {len(self.custom_factors)}个")
        except Exception as e:
            self.logger.error(f"加载自定义因子失败: {str(e)}")

    def _save_factors(self):
        """保存因子到文件"""
        try:
            import json
            data = [factor.to_dict() for factor in self.custom_factors.values()]
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存自定义因子失败: {str(e)}")

    def add_factor_from_code(self,
                           name: str,
                           description: str,
                           category: str,
                           code: str) -> bool:
        """从代码添加因子"""
        try:
            # 验证代码语法
            ast.parse(code)

            # 创建因子对象
            factor = CustomFactor(
                name=name,
                description=description,
                category=category,
                code=code
            )

            # 保存因子
            self.custom_factors[name] = factor
            self._save_factors()

            self.logger.info(f"添加自定义因子: {name}")
            return True

        except SyntaxError as e:
            self.logger.error(f"因子代码语法错误: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"添加自定义因子失败: {str(e)}")
            return False

    def add_factor_from_csv(self,
                          name: str,
                          description: str,
                          category: str,
                          csv_file_path: str) -> bool:
        """从CSV文件添加因子"""
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file_path)

            # 验证格式
            required_columns = ['symbol', 'date', 'value']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"CSV文件格式错误，需要列: {required_columns}")
                return False

            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['symbol', 'date'])

            # 保存到文件
            factor_file = FACTORS_DIR / f"{name}.csv"
            df.to_csv(factor_file, index=False)

            # 创建因子对象
            factor = CustomFactor(
                name=name,
                description=description,
                category=category,
                code=f"# CSV因子文件: {csv_file_path}"
            )

            self.custom_factors[name] = factor
            self._save_factors()

            self.logger.info(f"从CSV添加自定义因子: {name}")
            return True

        except Exception as e:
            self.logger.error(f"从CSV添加因子失败: {str(e)}")
            return False

    def get_factor(self, name: str) -> Optional[CustomFactor]:
        """获取自定义因子"""
        return self.custom_factors.get(name)

    def list_factors(self) -> List[CustomFactor]:
        """列出所有自定义因子"""
        return list(self.custom_factors.values())

    def remove_factor(self, name: str) -> bool:
        """移除因子"""
        if name in self.custom_factors:
            # 删除CSV文件
            csv_file = FACTORS_DIR / f"{name}.csv"
            if csv_file.exists():
                csv_file.unlink()

            del self.custom_factors[name]
            self._save_factors()
            self.logger.info(f"移除自定义因子: {name}")
            return True
        return False

    def enable_factor(self, name: str) -> bool:
        """启用因子"""
        if name in self.custom_factors:
            self.custom_factors[name].enabled = True
            self._save_factors()
            return True
        return False

    def disable_factor(self, name: str) -> bool:
        """禁用因子"""
        if name in self.custom_factors:
            self.custom_factors[name].enabled = False
            self._save_factors()
            return True
        return False

    def execute_factor(self,
                      name: str,
                      data: pd.DataFrame,
                      symbol_column: str = 'symbol',
                      date_column: str = 'date') -> pd.DataFrame:
        """执行自定义因子计算"""
        if name not in self.custom_factors:
            self.logger.error(f"自定义因子不存在: {name}")
            return pd.DataFrame()

        factor = self.custom_factors[name]
        if not factor.enabled:
            self.logger.warning(f"因子已禁用: {name}")
            return pd.DataFrame()

        try:
            # 检查是否有CSV文件
            csv_file = FACTORS_DIR / f"{name}.csv"
            if csv_file.exists():
                # 从CSV文件加载因子值
                factor_df = pd.read_csv(csv_file)
                factor_df['date'] = pd.to_datetime(factor_df['date'])
                factor_df = factor_df.rename(columns={'value': name})

                # 合并到原数据
                result_df = data.merge(factor_df[['symbol', 'date', name]],
                                     on=['symbol', 'date'],
                                     how='left')
                return result_df

            # 执行代码
            elif factor.code:
                return self._execute_factor_code(factor, data, symbol_column, date_column)

        except Exception as e:
            self.logger.error(f"执行自定义因子{name}失败: {str(e)}")
            return pd.DataFrame()

    def _execute_factor_code(self,
                           factor: CustomFactor,
                           data: pd.DataFrame,
                           symbol_column: str,
                           date_column: str) -> pd.DataFrame:
        """执行因子代码"""
        try:
            # 创建本地命名空间
            local_vars = {
                'pd': pd,
                'np': np,
                'data': data,
                'symbol_column': symbol_column,
                'date_column': date_column
            }

            # 执行代码
            exec(factor.code, globals(), local_vars)

            # 获取返回值
            if 'result' in local_vars:
                result = local_vars['result']
                if isinstance(result, pd.DataFrame):
                    if factor.name in result.columns:
                        # 合并到原数据
                        merge_columns = [symbol_column, date_column]
                        return data.merge(result[[factor.name] + merge_columns],
                                        on=merge_columns,
                                        how='left')
            else:
                self.logger.error(f"因子代码没有返回'result'变量")

        except Exception as e:
            self.logger.error(f"执行因子代码失败: {str(e)}")

        return pd.DataFrame()

    def validate_factor_code(self, code: str) -> Dict[str, Any]:
        """验证因子代码"""
        result = {
            "valid": False,
            "errors": [],
            "warnings": []
        }

        try:
            # 语法检查
            ast.parse(code)

            # 基本内容检查
            if 'def' not in code and 'result' not in code:
                result["warnings"].append("建议定义函数或返回result变量")

            if 'pd.DataFrame' not in code and 'result' not in code:
                result["warnings"].append("建议返回DataFrame格式结果")

            result["valid"] = True

        except SyntaxError as e:
            result["errors"].append(f"语法错误: {str(e)}")
        except Exception as e:
            result["errors"].append(f"其他错误: {str(e)}")

        return result

    def get_factor_history(self, name: str) -> Dict:
        """获取因子历史记录"""
        if name not in self.custom_factors:
            return {"error": "因子不存在"}

        factor = self.custom_factors[name]
        return {
            "name": factor.name,
            "description": factor.description,
            "category": factor.category,
            "created_at": factor.created_at.isoformat(),
            "version": factor.version,
            "enabled": factor.enabled
        }

    def backup_factors(self, backup_path: str) -> bool:
        """备份因子"""
        try:
            import shutil
            if self.storage_file.exists():
                shutil.copy2(self.storage_file, backup_path)
                self.logger.info(f"因子备份到: {backup_path}")
                return True
        except Exception as e:
            self.logger.error(f"备份因子失败: {str(e)}")
        return False

    def restore_factors(self, backup_path: str) -> bool:
        """恢复因子"""
        try:
            import shutil
            if Path(backup_path).exists():
                shutil.copy2(backup_path, self.storage_file)
                self._load_factors()
                self.logger.info(f"从备份恢复因子: {backup_path}")
                return True
        except Exception as e:
            self.logger.error(f"恢复因子失败: {str(e)}")
        return False