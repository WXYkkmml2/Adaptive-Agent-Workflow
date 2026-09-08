"""
LLM 客户端。（Step 3 更新：MockLLM 支持重规划场景）

新增：
- MockLLMClient 识别"重规划"上下文，返回更强的修正方案
- 支持注入失败场景用于测试 S4
"""

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
            return {
                "error": "LLM_ERROR",
                "message": f"LLM JSON 解析失败: {e}",
                "source": source,
            }


class MockLLMClient(LLMClient):
    """
    模拟 LLM 客户端。
    
    Step 3 新增 failure_mode:
      设为 True 时，对"生成调整方案"类任务首次返回一个不够强的修正，
      使仿真/约束校验失败，从而触发 S4 重规划。
      重规划时（prompt 中包含"重规划"关键词）返回更强的修正。
    """

    def __init__(self, failure_mode: bool = False):
        self.failure_mode = failure_mode
        # 追踪每个任务类型的调用次数，用于在重规划时返回不同结果
        self._call_counts = {}
        logger.info(
            f"使用 MockLLMClient（failure_mode={failure_mode}）"
        )

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, source: str = "unknown", max_tokens: int = None) -> str:
        if "任务分解专家" in system_prompt:
            return self._planner_response(user_prompt)
        elif "编排智能体" in system_prompt:
            return self._orchestration_response(user_prompt)
        elif "执行智能体" in system_prompt:
            return self._execution_response(user_prompt)
        else:
            return json.dumps({"message": "mock fallback"}, ensure_ascii=False)

    def _planner_response(self, user_prompt: str) -> str:
        bus_match = re.search(r"Bus\s*(\d+)", user_prompt)
        target_bus = int(bus_match.group(1)) if bus_match else 13

        tasks = {
            "tasks": [
                {"id": "t1", "description": "查询目标节点电压",
                 "dependencies": [], "devices": [target_bus],
                 "device_type": "bus", "voltage_level": "MV"},
                {"id": "t2", "description": "查询邻近节点状态",
                 "dependencies": [], "devices": [target_bus],
                 "device_type": "bus", "voltage_level": "MV"},
                {"id": "t3", "description": "查询相关线路负载",
                 "dependencies": [], "devices": [target_bus],
                 "device_type": "line", "voltage_level": "MV"},
                {"id": "t4", "description": "生成调整方案",
                 "dependencies": ["t1", "t2", "t3"], "devices": [target_bus],
                 "device_type": "gen", "voltage_level": "MV"},
                {"id": "t5", "description": "仿真验证方案",
                 "dependencies": ["t4"], "devices": [target_bus],
                 "device_type": "gen", "voltage_level": "MV"},
                {"id": "t6", "description": "验证全网约束",
                 "dependencies": ["t5"], "devices": [target_bus],
                 "device_type": "bus", "voltage_level": "MV"},
            ]
        }
        return json.dumps(tasks, ensure_ascii=False)

    def _orchestration_response(self, user_prompt: str) -> str:
        bus_match = re.search(r"设备.*?(\d+)", user_prompt)
        target_bus = int(bus_match.group(1)) if bus_match else 13

        is_replan = "重规划" in user_prompt or "replan" in user_prompt.lower()

        # ---- 根据任务关键词路由 ----
        # 优先识别"调整/生成方案"的任务，避免被 prior_results 中的关键词误导
        if "调整方案" in user_prompt or ("生成" in user_prompt and "方案" in user_prompt):
            # ---- failure_mode 关键逻辑 ----
            # 首次调用：如果 failure_mode=True，返回过小的调整量（故意不够）
            # 重规划调用：返回更大的调整量
            count_key = "adjust_plan"
            self._call_counts[count_key] = self._call_counts.get(count_key, 0) + 1

            if self.failure_mode and self._call_counts[count_key] == 1 and not is_replan:
                # 首次：微弱调整，不足以修复电压（故意设置为更保守的 1.00）
                # 且把仿真操作设为无效类型，保证 simulate_action 返回 failure，触发 S4
                vm_target = 1.00
                invalid_action = True
                logger.info("  [MockLLM] failure_mode: 首次返回不足的调整量")
            else:
                # 正常或重规划：足够的调整
                vm_target = 1.08
                if is_replan:
                    logger.info("  [MockLLM] 重规划: 返回更强的调整量")
            # 根据是否需要故意触发失败，生成不同的仿真 action
            if 'invalid_action' in locals() and invalid_action:
                sim_action = {"type": "invalid_action", "gen_id": 0, "vm_pu": vm_target}
            else:
                sim_action = {"type": "set_gen_voltage", "gen_id": 0, "vm_pu": vm_target}

            return json.dumps({"instructions": [
                {"tool": "get_generator_state", "params": {"gen_id": 0},
                 "description": "查询 Gen 0 当前状态",
                 "expected_result": "获取发电机参数"},
                {"tool": "simulate_action",
                 "params": {"action": sim_action},
                 "description": f"仿真提高 Gen 0 电压至 {vm_target}",
                 "expected_result": "仿真收敛且目标母线电压改善"},
            ]}, ensure_ascii=False)

        elif "仿真" in user_prompt:
            # t5 本身就是验证，用已在 t4 中 simulate 过的方案
            # 再做一次仿真确认
            count_key = "simulate"
            self._call_counts[count_key] = self._call_counts.get(count_key, 0) + 1

            if self.failure_mode and self._call_counts.get("adjust_plan", 0) <= 1 and not is_replan:
                vm_target = 1.02
            else:
                vm_target = 1.08

            return json.dumps({"instructions": [{
                "tool": "simulate_action",
                "params": {"action": {"type": "set_gen_voltage",
                                       "gen_id": 0, "vm_pu": vm_target}},
                "description": "仿真验证调压方案",
                "expected_result": "仿真收敛且目标母线电压改善",
            }]}, ensure_ascii=False)

        elif "约束" in user_prompt or "验证" in user_prompt:
            return json.dumps({"instructions": [{
                "tool": "check_constraints",
                "params": {},
                "description": "校验全网运行约束",
                "expected_result": "所有母线电压和线路负载率在限值内",
            }]}, ensure_ascii=False)

        # 其次处理其他查询类任务
        elif "查询目标节点电压" in user_prompt or ("目标" in user_prompt and "电压" in user_prompt):
            return json.dumps({"instructions": [{
                "tool": "get_bus_voltage",
                "params": {"bus_id": target_bus},
                "description": f"查询 Bus {target_bus} 电压",
                "expected_result": "获取电压数据",
            }]}, ensure_ascii=False)

        elif "邻近" in user_prompt or "邻居" in user_prompt:
            return json.dumps({"instructions": [{
                "tool": "get_neighbor_buses",
                "params": {"bus_id": target_bus},
                "description": f"查询 Bus {target_bus} 的相邻母线",
                "expected_result": "获取邻近节点列表",
            }]}, ensure_ascii=False)

        elif "线路" in user_prompt and "负载" in user_prompt:
            return json.dumps({"instructions": [{
                "tool": "get_line_loading",
                "params": {"line_id": 0},
                "description": "查询线路负载率",
                "expected_result": "获取线路负载数据",
            }]}, ensure_ascii=False)

        else:
            return json.dumps({"instructions": [{
                "tool": "get_bus_voltage",
                "params": {"bus_id": target_bus},
                "description": "查询目标母线电压",
                "expected_result": "获取电压数据",
            }]}, ensure_ascii=False)

    def _execution_response(self, user_prompt: str) -> str:
        tool_match = re.search(r'"tool":\s*"(\w+)"', user_prompt)
        tool_name = tool_match.group(1) if tool_match else "get_bus_voltage"

        if tool_name.startswith("get_") or tool_name == "check_constraints":
            strategy = "direct"
        elif tool_name == "simulate_action":
            strategy = "simulate"
        else:
            strategy = "simulate"

        params_match = re.search(r'"params":\s*(\{[^}]*\})', user_prompt)
        if params_match:
            try:
                params = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                params = {}
        else:
            params = {}

        return json.dumps({
            "tool_sequence": [{"tool": tool_name, "params": params}],
            "strategy": strategy,
        }, ensure_ascii=False)


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
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise ValueError("未设置 LLM_API_KEY")
        if raw_key and raw_key != self.api_key:
            logger.info("检测到并清洗了环境中的 LLM_API_KEY（可能包含不合法引号或空白）")
        logger.info(f"使用 RealLLMClient: model={self.model}")

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, source: str = "unknown", max_tokens: int = None) -> str:
        import urllib.request
        import urllib.error
        from llm.ssl_utils import get_ssl_context
        import time
        import socket

        # 强制模型只输出有效 JSON 的系统指令，放在用户提供的 system_prompt 之前
        json_enforcer = (
            "你是一个格式化助手：严格只输出有效的 JSON，不要任何解释或多余文本。"
            " 如果无法生成有效 JSON，请返回 {\"error\": \"...\"} 形式的 JSON。"
        )
        enforced_system = json_enforcer + "\n" + (system_prompt or "")

        payload_obj = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": enforced_system},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": min(max(float(temperature), 0.1), 0.2),
        }
        if max_tokens is not None:
            payload_obj["max_tokens"] = int(max_tokens)
        payload = json.dumps(payload_obj)

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload.encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            max_retries = int(os.environ.get("LLM_MAX_RETRIES", "2"))
        except Exception:
            max_retries = 2
        try:
            base_timeout = int(os.environ.get("LLM_TIMEOUT", "20"))
        except Exception:
            base_timeout = 20

        # 可配置的重写（reformat）请求超时，避免在短超时下直接失败
        try:
            reformat_timeout = int(os.environ.get("LLM_REFORMAT_TIMEOUT", str(min(base_timeout, 10))))
        except Exception:
            reformat_timeout = min(base_timeout, 10)

        logger.info(f"[LLM] start: source={source}, model={self.model}, timeout={base_timeout}s, retries={max_retries}")
        start_time = time.perf_counter()

        for attempt in range(1, max_retries + 1):
            attempt_start = time.perf_counter()
            try:
                context = get_ssl_context()
                if context is not None:
                    with urllib.request.urlopen(req, timeout=base_timeout, context=context) as resp:
                        status = resp.getcode()
                        headers = dict(resp.getheaders())
                        raw = resp.read().decode("utf-8")
                else:
                    with urllib.request.urlopen(req, timeout=base_timeout) as resp:
                        status = resp.getcode()
                        headers = dict(resp.getheaders())
                        raw = resp.read().decode("utf-8")

                # 解析响应并取出消息文本
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as e:
                    raw_excerpt = raw[:2000]
                    logger.error(f"[LLM] 非 JSON 响应或空响应: {e}; status={status}; headers={headers}; raw_excerpt={raw_excerpt}")
                    return json.dumps({
                        "error": "LLM_ERROR",
                        "message": f"LLM 返回非 JSON 响应或空响应: {e}",
                        "status": status,
                        "headers": headers,
                        "raw": raw_excerpt,
                    }, ensure_ascii=False)
                # 更稳健地提取 message 内容：优先取 content，其次尝试 reasoning_content、text、delta.content 等备用字段
                choice = data.get("choices", [])[0] if data.get("choices") else {}
                message = choice.get("message", {}) if isinstance(choice, dict) else {}
                content = message.get("content") if isinstance(message, dict) else None
                if not content:
                    # 尝试备用字段
                    content = message.get("reasoning_content") or choice.get("text") or (message.get("delta") or {}).get("content") or ""
                    if content:
                        logger.info(f"[LLM] using alternate message field for content: source={source}")

                # 尝试从 content 中提取代码块内的 JSON 并验证
                extracted = re.sub(r"^```(?:json)?\s*", "", str(content).strip())
                extracted = re.sub(r"\s*```$", "", extracted)
                try:
                    json.loads(extracted)
                    # 如果提取出的内容是合法 JSON，直接返回提取后的文本
                    elapsed = time.perf_counter() - start_time
                    logger.info(f"[LLM] success(json-extract): source={source}, attempt={attempt}/{max_retries}, elapsed={elapsed:.2f}s")
                    return extracted
                except Exception:
                    # 继续后续处理
                    pass

                # 如果 content 为空或仅包含空白，记录更多 HTTP 上下文
                if not content or str(content).strip() == "":
                    raw_excerpt = raw[:2000]
                    logger.error(f"[LLM] empty_content: source={source}, status={status}, headers={headers}, raw_excerpt={raw_excerpt}")
                    err_msg = "LLM 返回空的 content 字段"
                    return json.dumps({
                        "error": "LLM_ERROR",
                        "message": err_msg,
                        "raw": raw,
                        "status": status,
                        "headers": headers,
                        "source": source,
                    }, ensure_ascii=False)

                # 若 content 不是合法 JSON，尝试自动向模型发起一次“仅输出 JSON”的重写请求
                try:
                    reformat_payload = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "你是一个格式化助手：严格只输出有效的 JSON，不要任何解释或多余文本。"},
                            {"role": "user", "content": f"请将下面的文本仅转换为有效的 JSON（不要解释）：\n\n{content}"},
                        ],
                        "temperature": 0.0,
                    }
                    req2 = urllib.request.Request(
                        f"{self.base_url}/chat/completions",
                        data=json.dumps(reformat_payload).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                    )
                    # 只做一次短重试（同步调用），超时可通过 LLM_REFORMAT_TIMEOUT 配置
                    if context is not None:
                        with urllib.request.urlopen(req2, timeout=reformat_timeout, context=context) as resp2:
                            raw2 = resp2.read().decode("utf-8")
                    else:
                        with urllib.request.urlopen(req2, timeout=reformat_timeout) as resp2:
                            raw2 = resp2.read().decode("utf-8")

                    # 尝试从重写结果中提取 JSON
                    try:
                        data2 = json.loads(raw2)
                        formatted = data2["choices"][0]["message"]["content"]
                    except Exception:
                        formatted = raw2

                    formatted_extracted = re.sub(r"^```(?:json)?\s*", "", str(formatted).strip())
                    formatted_extracted = re.sub(r"\s*```$", "", formatted_extracted)
                    def _try_extract_json_fragment(text: str):
                        if not text or not isinstance(text, str):
                            return None
                        # 找到第一个 '{' 与最后一个 '}' 的子串，尝试解析
                        start = text.find("{")
                        end = text.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            candidate = text[start:end+1]
                            try:
                                json.loads(candidate)
                                return candidate
                            except Exception:
                                return None
                        return None

                    try:
                        json.loads(formatted_extracted)
                        logger.info(f"[LLM] reformat success: source={source}")
                        return formatted_extracted
                    except Exception:
                        logger.error(f"[LLM] reformat failed: source={source}, formatted_preview={formatted_extracted[:1000]}")
                        # 在放弃之前尝试从原始 content 中提取 JSON 片段
                        fragment = _try_extract_json_fragment(content) or _try_extract_json_fragment(formatted_extracted)
                        if fragment:
                            logger.info(f"[LLM] extracted JSON fragment from non-JSON content: source={source}")
                            return fragment
                        # 返回结构化错误，包含原始 content 以便上层能统一处理
                        elapsed = time.perf_counter() - start_time
                        logger.info(f"[LLM] success (non-json): source={source}, attempt={attempt}/{max_retries}, elapsed={elapsed:.2f}s")
                        return json.dumps({
                            "error": "LLM_ERROR",
                            "message": "LLM 返回非 JSON 内容且重写后仍非 JSON",
                            "raw": content,
                            "source": source,
                        }, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"[LLM] reformat request failed: {e}")
                    # 在放弃之前尝试从 content 中提取 JSON 片段
                    def _try_extract_json_fragment(text: str):
                        if not text or not isinstance(text, str):
                            return None
                        start = text.find("{")
                        end = text.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            candidate = text[start:end+1]
                            try:
                                json.loads(candidate)
                                return candidate
                            except Exception:
                                return None
                        return None

                    fragment = _try_extract_json_fragment(content)
                    if fragment:
                        logger.info(f"[LLM] extracted JSON fragment after reformat failure: source={source}")
                        return fragment

                    elapsed = time.perf_counter() - start_time
                    logger.info(f"[LLM] success (non-json, no-reformat): source={source}, attempt={attempt}/{max_retries}, elapsed={elapsed:.2f}s")
                    return json.dumps({
                        "error": "LLM_ERROR",
                        "message": f"LLM 返回非 JSON 内容，重写请求失败: {e}",
                        "raw": content,
                        "source": source,
                    }, ensure_ascii=False)

            except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
                elapsed = time.perf_counter() - attempt_start
                logger.warning(f"[LLM] failure: source={source}, attempt={attempt}/{max_retries}, elapsed={elapsed:.2f}s, error={e}")
                if attempt < max_retries:
                    continue
                err_msg = f"LLM API 网络错误/超时: {e}"
                logger.error(f"[LLM] final_error: source={source}, elapsed={time.perf_counter() - start_time:.2f}s, error={err_msg}")
                return json.dumps({"error": "LLM_ERROR", "message": err_msg, "source": source}, ensure_ascii=False)

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                elapsed = time.perf_counter() - attempt_start
                raw_excerpt = raw[:1000] if 'raw' in locals() else "<empty>"
                logger.error(f"[LLM] parse_error: source={source}, attempt={attempt}/{max_retries}, elapsed={elapsed:.2f}s, error={e}, raw={raw_excerpt}")
                err_msg = f"LLM JSON 解析失败: {e}"
                return json.dumps({"error": "LLM_ERROR", "message": err_msg, "source": source}, ensure_ascii=False)

        elapsed = time.perf_counter() - start_time
        logger.error(f"[LLM] exhausted: source={source}, elapsed={elapsed:.2f}s, retries={max_retries}")
        return json.dumps({"error": "LLM_ERROR", "message": "LLM 调用失败且重试耗尽", "source": source}, ensure_ascii=False)


def create_llm_client(failure_mode: bool = False) -> LLMClient:
    """工厂函数。有 API Key 就用真实客户端，没有就用 Mock。"""
    if os.environ.get("LLM_API_KEY"):
        try:
            return RealLLMClient()
        except ValueError:
            pass
    return MockLLMClient(failure_mode=failure_mode)