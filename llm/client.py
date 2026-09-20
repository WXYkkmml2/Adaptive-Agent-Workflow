"""LLM 客户端：向兼容 Chat Completions 的服务请求 JSON。"""

import json
import os
import re
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, source: str = "unknown", max_tokens: int = None) -> str:
        raise NotImplementedError

    def complete_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, source: str = "unknown", max_tokens: int = None) -> dict:
        try:
            raw = self.complete(system_prompt, user_prompt, temperature, source=source, max_tokens=max_tokens)
        except TypeError:
            try:
                raw = self.complete(system_prompt, user_prompt, temperature, source=source)
            except TypeError:
                raw = self.complete(system_prompt, user_prompt, temperature)

        if isinstance(raw, dict):
            if raw.get("error") == "LLM_ERROR":
                return raw
            return raw

        cleaned = re.sub(r"^```(?:json)?\s*", "", str(raw).strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        # 如果清理后内容为空（例如 LLM 返回空字符串或仅包含代码块标记），视为解析失败
        if not cleaned or cleaned.strip() == "":
            logger.error(f"[LLM] JSON 解析失败: 清理后响应为空\n原始响应: {raw}")
            msg = "LLM 返回空响应或仅包含代码块标记，无法解析为 JSON"
            return {"error": "LLM_ERROR", "message": msg, "raw": str(raw)}
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"[LLM] JSON 解析失败: {e}\n原始响应: {raw}")
            msg = f"LLM JSON 解析失败: {e}"
            # 返回结构化错误以便上层代码和测试能够检测到解析失败
            return {"error": "LLM_ERROR", "message": msg, "raw": str(raw)}


class RealLLMClient(LLMClient):
    """真实 LLM 客户端（可选，需要 API Key）。"""

    def __init__(self):
        # 从环境读取 API Key 并进行容错清洗：去除左右花引号、直引号和首尾空白。
        
        raw_key = os.environ.get("LLM_API_KEY", "")
        if isinstance(raw_key, str):
            cleaned = raw_key.strip()
            # 常见的“花引号”清理（U+201C, U+201D）以及普通引号
            cleaned = cleaned.replace('\u201c', '').replace('\u201d', '')
            if (cleaned.startswith('"') and cleaned.endswith('"')) or (
                    cleaned.startswith("'") and cleaned.endswith("'")):
                cleaned = cleaned[1:-1]
            cleaned = cleaned.strip()
        else:
            cleaned = ""

        self.api_key = cleaned
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model = os.environ.get("LLM_MODEL", "deepseek-flash")
        if not self.api_key:
            raise ValueError("未设置 LLM_API_KEY")
        if raw_key and raw_key != self.api_key:
            logger.info("检测到并清洗了环境中的 LLM_API_KEY（可能包含不合法引号或空白）")
        logger.info("使用 RealLLMClient: base_url=%s, model=%s", self.base_url, self.model)

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, source: str = "unknown", max_tokens: int = None) -> str:
        """请求服务端 JSON 模式；返回 JSON 文本，供 complete_json 统一解析。"""
        import socket
        import time
        import urllib.error
        import urllib.request
        from llm.ssl_utils import get_ssl_context

        def error(message: str) -> str:
            return json.dumps({"error": "LLM_ERROR", "message": message, "source": source}, ensure_ascii=False)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "请只输出一个有效的 JSON 对象，不要解释或 Markdown。\n" + (system_prompt or "")},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": min(max(float(temperature), 0.0), 2.0),
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "max_tokens": int(max_tokens) if max_tokens is not None else 2048,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            retries = max(1, int(os.environ.get("LLM_MAX_RETRIES", "2")))
            timeout = max(1, int(os.environ.get("LLM_TIMEOUT", "30")))
        except ValueError:
            return error("LLM_MAX_RETRIES 和 LLM_TIMEOUT 必须为整数")

        context = get_ssl_context()
        for attempt in range(1, retries + 1):
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                    data = json.load(response)
                choices = data.get("choices") or []
                if not choices:
                    return error("LLM 响应缺少 choices")
                choice = choices[0]
                content = (choice.get("message") or {}).get("content")
                if choice.get("finish_reason") == "length":
                    usage = data.get("usage") or {}
                    logger.warning(
                        "[LLM] output truncated: source=%s, max_tokens=%s, usage=%s",
                        source, payload["max_tokens"], usage,
                    )
                    return error(
                        f"LLM JSON 输出被截断（max_tokens={payload['max_tokens']}, "
                        f"completion_tokens={usage.get('completion_tokens', '未知')}）；"
                        "请增大 max_tokens 或缩短提示词"
                    )
                if not isinstance(content, str) or not content.strip():
                    return error("LLM 返回空 content；请调整提示词或重试")
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as exc:
                    return error(f"LLM 返回无效 JSON: {exc}")
                if not isinstance(parsed, dict):
                    return error("LLM 返回的 JSON 顶层必须为对象")
                logger.info("[LLM] success: source=%s, attempt=%d, elapsed=%.2fs",
                            source, attempt, time.perf_counter() - started)
                return content
            except urllib.error.HTTPError as exc:
                detail = exc.read(1000).decode("utf-8", errors="replace")
                message = f"LLM HTTP {exc.code}: {detail}"
                logger.warning("[LLM] %s: source=%s", message, source)
                if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                    return error(message)
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                message = f"LLM 网络错误/超时: {exc}"
                logger.warning("[LLM] %s: source=%s, attempt=%d/%d", message, source, attempt, retries)
                if attempt == retries:
                    return error(message)
        return error("LLM 调用失败")


def create_llm_client() -> LLMClient:
    """真实运行必须显式配置 API Key，避免无声切换为模拟响应。"""
    return RealLLMClient()
