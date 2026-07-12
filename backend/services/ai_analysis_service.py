"""
AI 因子分析流式解读服务。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Iterator
from urllib import error, request

from backend.services.ai_factor_service import AIFactorService, ai_factor_service


logger = logging.getLogger(__name__)


class AIFactorAnalysisService:
    """基于大模型的因子分析流式解读服务。"""

    def __init__(self, config_service: AIFactorService | None = None):
        self.config_service = config_service or ai_factor_service

    def stream_analysis_report(self, payload: Dict[str, Any]) -> Iterator[str]:
        factor = payload.get("factor") or {}
        analysis_context = payload.get("analysis_context") or {}
        stock_context = {
            "stock_code": payload.get("stock_code"),
            "start_date": payload.get("start_date"),
            "end_date": payload.get("end_date"),
            "chart_period": payload.get("chart_period"),
        }

        if not factor.get("name"):
            raise ValueError("缺少因子名称，无法生成 AI 解读")

        if not analysis_context:
            raise ValueError("缺少分析结果数据，请先执行因子分析")

        config = self.config_service.load_runtime_config()
        messages = self._build_messages(
            factor=factor,
            stock_context=stock_context,
            analysis_context=analysis_context,
        )

        yield self._format_sse(
            "metadata",
            {
                "model": config.model,
                "request_url": self._build_request_url(config),
                "factor_name": factor.get("name"),
            },
        )

        chunk_count = 0
        try:
            for chunk in self._coalesce_stream_chunks(
                self._stream_chat_completion(
                    config=config,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2200,
                )
            ):
                chunk_count += 1
                yield self._format_sse("chunk", {"content": chunk})

            if chunk_count == 0:
                raise ValueError("模型未返回任何流式内容")

            yield self._format_sse("done", {"chunk_count": chunk_count})
        except Exception as exc:  # noqa: BLE001
            logger.exception("AI 因子分析流式解读失败")
            yield self._format_sse("error", {"message": str(exc)})

    def _build_messages(
        self,
        *,
        factor: Dict[str, Any],
        stock_context: Dict[str, Any],
        analysis_context: Dict[str, Any],
    ) -> list[Dict[str, str]]:
        prompt_payload = {
            "factor": {
                "name": factor.get("name"),
                "category": factor.get("category"),
                "description": factor.get("description"),
                "formula_type": factor.get("formula_type"),
                "code": factor.get("code"),
            },
            "analysis_scope": stock_context,
            "analysis_context": self._sanitize_for_prompt(analysis_context),
        }

        return [
            {
                "role": "system",
                "content": (
                    "你是 FactorHub 的资深量化研究员与投研写作助手。"
                    "请基于输入的因子背景与分析结果，输出一份适合研究员阅读的 Markdown 中文报告。"
                    "要求实事求是，不能编造缺失数据；遇到数据不足时要明确指出。"
                    "请重点解释因子含义、有效性、风险、适用场景和后续优化建议。"
                    "输出必须是纯 Markdown 正文，不要输出 JSON，不要加代码围栏包裹整篇报告。"
                    "建议至少包含这些一级或二级标题："
                    "总体结论、因子定义与业务逻辑、关键指标解读、稳定性与风险、适用场景、优化建议。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "以下是因子背景信息和分析结果，请写成结构化 Markdown 报告：\n"
                    f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
                ),
            },
        ]

    def _sanitize_for_prompt(
        self,
        value: Any,
        *,
        depth: int = 0,
        max_depth: int = 6,
        max_items: int = 20,
        max_string: int = 1200,
    ) -> Any:
        if depth >= max_depth:
            return "...(已截断)"

        if isinstance(value, dict):
            sanitized: Dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= max_items:
                    sanitized["__truncated__"] = f"其余 {len(value) - max_items} 个字段已省略"
                    break
                sanitized[str(key)] = self._sanitize_for_prompt(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string=max_string,
                )
            return sanitized

        if isinstance(value, list):
            sanitized_items = [
                self._sanitize_for_prompt(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string=max_string,
                )
                for item in value[:max_items]
            ]
            if len(value) > max_items:
                sanitized_items.append(f"...(其余 {len(value) - max_items} 项已省略)")
            return sanitized_items

        if isinstance(value, tuple):
            return self._sanitize_for_prompt(
                list(value),
                depth=depth,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
            )

        if isinstance(value, str):
            return value if len(value) <= max_string else f"{value[:max_string]}...(已截断)"

        if isinstance(value, (int, float, bool)) or value is None:
            return value

        return str(value)

    def _build_request_url(self, config: Any) -> str:
        request_path = (config.request_path or "").strip()
        if request_path.startswith(("http://", "https://")):
            return request_path
        base_url = (config.base_url or "").strip().rstrip("/")
        normalized_path = request_path if request_path.startswith("/") else f"/{request_path}"
        return f"{base_url}{normalized_path}"

    def _stream_chat_completion(
        self,
        *,
        config: Any,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterable[str]:
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        req = request.Request(
            self._build_request_url(config),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {config.api_key}",
                "api-key": config.api_key,
                "Cache-Control": "no-cache",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=180) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload_text = line[5:].strip()
                    if payload_text == "[DONE]":
                        break
                    try:
                        chunk_payload = json.loads(payload_text)
                    except json.JSONDecodeError:
                        logger.debug("跳过非 JSON 流式分片: %s", payload_text)
                        continue
                    content = self._extract_stream_content(chunk_payload)
                    if content:
                        yield content
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise ValueError(f"模型流式请求失败（HTTP {exc.code}）：{detail or exc.reason}") from exc
        except error.URLError as exc:
            raise ValueError(f"模型流式连接失败：{exc.reason}") from exc

    def _coalesce_stream_chunks(
        self,
        chunks: Iterable[str],
        *,
        flush_chars: int = 120,
        soft_flush_chars: int = 48,
    ) -> Iterable[str]:
        """合并上游过碎的 token 分片，降低前端高频重渲染压力。"""
        buffer: list[str] = []
        buffer_length = 0

        for chunk in chunks:
            if not chunk:
                continue

            buffer.append(chunk)
            buffer_length += len(chunk)

            should_flush = buffer_length >= flush_chars
            if not should_flush and buffer_length >= soft_flush_chars:
                should_flush = "\n\n" in chunk or "\n" in chunk
            if not should_flush and buffer_length >= soft_flush_chars:
                should_flush = chunk.endswith(("。", "！", "？", ".", "!", "?", "：", ":"))

            if should_flush:
                yield "".join(buffer)
                buffer = []
                buffer_length = 0

        if buffer:
            yield "".join(buffer)

    def _extract_stream_content(self, chunk_payload: Dict[str, Any]) -> str:
        choices = chunk_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = first_choice.get("delta") or first_choice.get("message") or {}
        if not isinstance(delta, dict):
            return ""

        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text") or ""))
            return "".join(text_parts)
        return ""

    def _format_sse(self, event_type: str, data: Dict[str, Any]) -> str:
        payload = {"type": event_type, **data}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


ai_factor_analysis_service = AIFactorAnalysisService()
