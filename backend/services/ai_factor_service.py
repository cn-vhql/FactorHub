"""
AI 因子生成与模型配置服务。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, parse, request

from backend.core.settings import settings
from backend.formula_engine.runtime import infer_formula_type, normalize_formula_type


logger = logging.getLogger(__name__)


@dataclass
class AIModelConfig:
    """OpenAI 协议模型配置。"""

    base_url: str = ""
    model: str = ""
    api_key: str = ""
    request_path: str = "/chat/completions"


class AIFactorService:
    """AI 因子配置、验证与代码生成服务。"""

    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or (settings.DATA_DIR / "ai_model_config.json")

    def get_config_view(self) -> Dict[str, Any]:
        config = self._load_config()
        return self._serialize_config(config)

    def load_runtime_config(self) -> AIModelConfig:
        """获取包含真实密钥的运行时配置。"""
        config = self._load_config()
        self._validate_config_fields(config)
        return config

    def save_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        config = self._merge_config(payload, allow_keep_existing_key=True)
        self._validate_config_fields(config)
        self._save_config(config)
        return self._serialize_config(config)

    def validate_config(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        config = self._merge_config(payload or {}, allow_keep_existing_key=True)
        self._validate_config_fields(config)

        response = self._request_chat_completion(
            config,
            messages=[
                {
                    "role": "system",
                    "content": "你是模型连通性检测助手。请只回复 OK。",
                },
                {
                    "role": "user",
                    "content": "请只返回 OK，用于验证接口是否生效。",
                },
            ],
            temperature=0,
            max_tokens=16,
        )

        content = self._extract_message_content(response).strip()
        if not content:
            raise ValueError("模型返回内容为空，无法确认接口是否可用")

        return {
            "success": True,
            "model": config.model,
            "request_url": self._build_request_url(config),
            "reply_preview": content[:200],
        }

    def generate_factor(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        requirement = (payload.get("requirement") or "").strip()
        if not requirement:
            raise ValueError("请先提供因子需求描述")

        preferred_formula_type = (payload.get("formula_type") or "auto").strip() or "auto"
        suggested_name = (payload.get("suggested_name") or "").strip()
        description_hint = (payload.get("description_hint") or "").strip()
        max_rounds = int(payload.get("max_rounds") or 4)
        max_rounds = max(1, min(max_rounds, 6))

        config = self._merge_config({}, allow_keep_existing_key=True)
        self._validate_config_fields(config)

        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(),
            },
            {
                "role": "user",
                "content": self._build_generation_prompt(
                    requirement=requirement,
                    preferred_formula_type=preferred_formula_type,
                    suggested_name=suggested_name,
                    description_hint=description_hint,
                ),
            },
        ]

        last_error = ""
        for attempt in range(1, max_rounds + 1):
            response = self._request_chat_completion(
                config,
                messages=messages,
                temperature=0.2,
                max_tokens=1400,
            )
            content = self._extract_message_content(response)
            candidate = self._parse_generation_response(content)
            resolved = self._normalize_generated_factor(candidate, preferred_formula_type)

            from backend.services.factor_service import factor_service

            is_valid, validation_message = factor_service.validate_factor_code(
                resolved["code"],
                formula_type=resolved["formula_type"],
            )
            if is_valid:
                return {
                    "name": resolved["name"],
                    "description": resolved["description"],
                    "formula_type": resolved["formula_type"],
                    "code": resolved["code"],
                    "validation_message": validation_message,
                    "attempts": attempt,
                    "request_url": self._build_request_url(config),
                }

            last_error = validation_message
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "上一个候选因子未通过校验，请根据以下错误修正，并继续只返回 JSON：\n"
                        f"{validation_message}\n"
                        "请保持原始业务意图不变，生成可直接执行、可通过校验的最终版本。"
                    ),
                }
            )

        raise ValueError(f"AI 连续 {max_rounds} 轮修正后仍未生成有效因子：{last_error or '未知错误'}")

    def _build_system_prompt(self) -> str:
        return (
            "你是 FactorHub 的量化因子工程师。"
            "你的任务是根据自然语言需求，生成可以直接在 FactorHub 执行并通过校验的因子。"
            "必须只返回一个 JSON 对象，禁止输出 Markdown、解释文字或代码块围栏。"
            'JSON 键固定为 name、description、formula_type、code。'
            "formula_type 只能是 mylanguage 或 python。"
            "如果使用 Python 因子，优先生成单个向量化表达式，尽量不要生成 def calculate_factor 包装。"
            "Python 因子禁止使用 lambda、rolling(...).apply(...)、for、while、推导式、from ... import ...。"
            "Python 可用字段包括 close、open、high、low、volume、amount、turnover、amplitude、pct_change、np、pd，"
            "也可调用 MA、EMA、SMA、REF、HHV、LLV、RANGEPOS、CROSS、IF、RSI、MACD、ATR、BBANDS 等系统函数。"
            "如果使用麦语言，可使用 OPEN、HIGH、LOW、CLOSE、VOL、AMOUNT、TURNOVER、AMPLITUDE、PCT_CHG，"
            "以及 MA、EMA、SMA、REF、HHV、LLV、RANGEPOS、IF、CROSS 等函数。"
            "麦语言支持多输出，主输出优先使用 XG:，没有 XG: 时最后一个 : 输出会被视为主序列。"
            "生成结果必须以生产可用、可执行、数据含义明确为目标。"
        )

    def _build_generation_prompt(
        self,
        *,
        requirement: str,
        preferred_formula_type: str,
        suggested_name: str,
        description_hint: str,
    ) -> str:
        return (
            f"因子需求：{requirement}\n"
            f"公式类型偏好：{preferred_formula_type}\n"
            f"建议名称：{suggested_name or '无'}\n"
            f"补充说明：{description_hint or '无'}\n"
            "请输出一个可直接保存到因子库的因子，并确保 name 简洁、description 准确。"
        )

    def _normalize_generated_factor(self, candidate: Dict[str, Any], preferred_formula_type: str) -> Dict[str, str]:
        code = str(candidate.get("code") or "").strip()
        if not code:
            raise ValueError("模型返回缺少 code 字段")

        requested_type = str(candidate.get("formula_type") or preferred_formula_type or "auto").strip() or "auto"
        try:
            formula_type = normalize_formula_type(requested_type, code)
        except Exception:  # noqa: BLE001
            formula_type = infer_formula_type(code)

        name = str(candidate.get("name") or "").strip() or "AI_生成因子"
        description = str(candidate.get("description") or "").strip()

        return {
            "name": name,
            "description": description,
            "formula_type": formula_type,
            "code": code,
        }

    def _parse_generation_response(self, content: str) -> Dict[str, Any]:
        stripped = (content or "").strip()
        if not stripped:
            raise ValueError("模型返回内容为空")

        json_candidates = [stripped]
        if "```" in stripped:
            segments = stripped.split("```")
            for segment in segments:
                candidate = segment.strip()
                if candidate.lower().startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    json_candidates.append(candidate)

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_candidates.append(stripped[start : end + 1])

        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        raise ValueError("模型返回结果不是有效 JSON，请检查模型输出格式")

    def _merge_config(self, payload: Dict[str, Any], *, allow_keep_existing_key: bool) -> AIModelConfig:
        existing = self._load_config()
        base_url = (payload.get("base_url") if payload else None)
        model = (payload.get("model") if payload else None)
        api_key = (payload.get("api_key") if payload else None)
        request_path = (payload.get("request_path") if payload else None)

        merged = AIModelConfig(
            base_url=str(base_url).strip() if base_url is not None else existing.base_url,
            model=str(model).strip() if model is not None else existing.model,
            api_key=str(api_key).strip() if api_key is not None else existing.api_key,
            request_path=str(request_path).strip() if request_path is not None else existing.request_path,
        )

        if allow_keep_existing_key and api_key is not None and not str(api_key).strip():
            merged.api_key = existing.api_key

        if not merged.request_path:
            merged.request_path = "/chat/completions"

        return merged

    def _validate_config_fields(self, config: AIModelConfig) -> None:
        if not config.base_url:
            raise ValueError("请先配置模型服务地址")
        if not config.model:
            raise ValueError("请先配置模型 ID")
        if not config.api_key:
            raise ValueError("请先配置模型密钥")
        if not config.request_path:
            raise ValueError("请先配置模型请求地址")

    def _save_config(self, config: AIModelConfig) -> None:
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(
            json.dumps(asdict(config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_config(self) -> AIModelConfig:
        if not self.config_file.exists():
            return AIModelConfig()
        try:
            payload = json.loads(self.config_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"AI 模型配置文件已损坏：{exc}") from exc
        return AIModelConfig(
            base_url=str(payload.get("base_url") or "").strip(),
            model=str(payload.get("model") or "").strip(),
            api_key=str(payload.get("api_key") or "").strip(),
            request_path=str(payload.get("request_path") or "/chat/completions").strip() or "/chat/completions",
        )

    def _serialize_config(self, config: AIModelConfig) -> Dict[str, Any]:
        return {
            "base_url": config.base_url,
            "model": config.model,
            "request_path": config.request_path,
            "api_key": "",
            "has_api_key": bool(config.api_key),
            "api_key_masked": self._mask_key(config.api_key),
            "configured": bool(config.base_url and config.model and config.api_key and config.request_path),
        }

    def _build_request_url(self, config: AIModelConfig) -> str:
        request_path = config.request_path.strip()
        if request_path.startswith(("http://", "https://")):
            return request_path

        base_url = config.base_url.strip().rstrip("/")
        normalized_path = request_path if request_path.startswith("/") else f"/{request_path}"
        return f"{base_url}{normalized_path}"

    def _request_chat_completion(
        self,
        config: AIModelConfig,
        *,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self._build_request_url(config),
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
                "api-key": config.api_key,
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            logger.error("AI 模型请求失败: %s", detail)
            raise ValueError(f"模型请求失败（HTTP {exc.code}）：{detail or exc.reason}") from exc
        except error.URLError as exc:
            raise ValueError(f"模型服务连接失败：{exc.reason}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"模型返回了非 JSON 响应：{body[:300]}") from exc

    def _extract_message_content(self, response: Dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("模型响应缺少 choices 字段")

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise ValueError("模型响应缺少 message 字段")

        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text") or ""))
            return "".join(text_parts)
        return str(content or "")

    def _mask_key(self, api_key: str) -> str:
        if not api_key:
            return ""
        if len(api_key) <= 8:
            return "*" * len(api_key)
        return f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}"


ai_factor_service = AIFactorService()
