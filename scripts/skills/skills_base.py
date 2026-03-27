"""
Skills基础工具类 - 封装HTTP API调用和输出格式化
"""
import requests
import json
import sys
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from urllib.parse import urljoin
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class SkillsClient:
    """Skills HTTP客户端"""

    def __init__(self, base_url: str = "http://localhost:8000/api"):
        """
        初始化客户端

        Args:
            base_url: API基地址，默认http://localhost:8000/api
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })

    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """发送GET请求"""
        url = urljoin(self.base_url, endpoint.lstrip("/"))
        try:
            response = self.session.get(url, params=params, timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"请求超时: {endpoint}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"连接失败: {self.base_url}\n请确保API服务运行在 {self.base_url}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP错误 {response.status_code}: {response.text}")

    def post(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """发送POST请求"""
        url = urljoin(self.base_url, endpoint.lstrip("/"))
        try:
            response = self.session.post(url, json=data, timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"请求超时: {endpoint}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"连接失败: {self.base_url}\n请确保API服务运行在 {self.base_url}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP错误 {response.status_code}: {response.text}")


class SkillOutput:
    """技能输出格式化器"""

    @staticmethod
    def section(title: str) -> str:
        """创建分区标题"""
        return f"\n{'='*60}\n{title}\n{'='*60}"

    @staticmethod
    def subsection(title: str) -> str:
        """创建小标题"""
        return f"\n{title}"

    @staticmethod
    def table(headers: List[str], rows: List[List[str]], widths: Optional[List[int]] = None) -> str:
        """格式化表格"""
        if not rows:
            return "（无数据）"

        if widths is None:
            # 自动计算列宽
            widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    widths[i] = max(widths[i], len(str(cell)))

        # 构建分隔线
        separator = "┌" + "┬".join(f"{'─' * (w + 2)}" for w in widths) + "┐"
        end_line = "└" + "┴".join(f"{'─' * (w + 2)}" for w in widths) + "┘"
        header_sep = "├" + "┼".join(f"{'─' * (w + 2)}" for w in widths) + "┤"

        lines = [separator]

        # 添加表头
        header_row = "│ " + " │ ".join(
            str(h).ljust(w) for h, w in zip(headers, widths)
        ) + " │"
        lines.append(header_row)
        lines.append(header_sep)

        # 添加数据行
        for row in rows:
            data_row = "│ " + " │ ".join(
                str(cell).ljust(w) for cell, w in zip(row, widths)
            ) + " │"
            lines.append(data_row)

        lines.append(end_line)
        return "\n".join(lines)

    @staticmethod
    def metric(label: str, value: Any, unit: str = "", emoji: str = "") -> str:
        """格式化单个指标"""
        formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
        if emoji:
            return f"├─ {label}: {formatted_value} {unit} {emoji}"
        else:
            return f"├─ {label}: {formatted_value} {unit}"

    @staticmethod
    def status_line(message: str, status: str = "info") -> str:
        """显示状态行"""
        icons = {
            "success": "✅",
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️",
            "progress": "⏳",
        }
        icon = icons.get(status, "•")
        return f"{icon} {message}"

    @staticmethod
    def error(message: str) -> str:
        """格式化错误消息"""
        return f"\n❌ 错误: {message}\n"

    @staticmethod
    def success(message: str) -> str:
        """格式化成功消息"""
        return f"\n✅ {message}\n"


def parse_date_range(start_str: Optional[str], end_str: Optional[str]) -> Tuple[str, str]:
    """
    解析和验证日期范围

    Args:
        start_str: 开始日期字符串 (YYYY-MM-DD)
        end_str: 结束日期字符串 (YYYY-MM-DD)

    Returns:
        (start_date, end_date) 元组

    Raises:
        ValueError: 如果日期格式不正确
    """
    try:
        if start_str:
            start_date = datetime.strptime(start_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        else:
            start_date = "2020-01-01"

        if end_str:
            end_date = datetime.strptime(end_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        else:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # 验证日期顺序
        if start_date > end_date:
            raise ValueError("开始日期不能晚于结束日期")

        return start_date, end_date
    except ValueError as e:
        raise ValueError(f"日期格式错误，请使用 YYYY-MM-DD 格式: {e}")


def create_parser(name: str, description: str) -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog=name,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    return parser


if __name__ == "__main__":
    # 测试连接
    client = SkillsClient()
    try:
        result = client.get("/factors", params={"limit": 1})
        print(SkillOutput.success(f"API连接成功! 获得 {len(result.get('data', []))} 个因子"))
    except Exception as e:
        print(SkillOutput.error(str(e)))
