"""
LLM 客户端。

提供两种实现：
1. MockLLMClient：不需要 API Key，返回预定义响应，用于测试和演示
2. RealLLMClient：调用真实 API（需要 API Key，可选）

默认用 Mock，设置环境变量 LLM_API_KEY 后自动切换到真实 API。
"""

import json
import os
import re
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM 客户端基类。"""

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """调用 LLM，返回文本响应。"""
        raise NotImplementedError

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> dict:
        """调用 LLM 并解析 JSON 响应。"""
        raw = self.complete(system_prompt, user_prompt, temperature)
        # 清理可能的 markdown 代码块包裹
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}\n原始响应: {raw}")
            return {}


class MockLLMClient(LLMClient):
    """
    模拟 LLM 客户端。
    
    根据 prompt 中的关键词判断当前在 S1/S2/S3 的哪个阶段，
    返回对应的预定义 JSON 响应。
    
    预定义的场景是"范围"文档中的标准场景：
    "Bus 14 电压过低，请分析并恢复"
    """

    def __init__(self):
        logger.info("使用 MockLLMClient（无需 API Key）")

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        # 根据 system_prompt 判断调用场景
        if "任务分解专家" in system_prompt:
            return self._planner_response(user_prompt)
        elif "编排智能体" in system_prompt:
            return self._orchestration_response(user_prompt)
        elif "执行智能体" in system_prompt:
            return self._execution_response(user_prompt)
        else:
            return json.dumps({"message": "mock fallback"}, ensure_ascii=False)

    def _planner_response(self, user_prompt: str) -> str:
        """
        S1 规划器的模拟响应。
        生成标准的 6 步任务 DAG（对应"范围"文档）。
        """
        # 从 prompt 中提取目标母线（简单解析）
        bus_match = re.search(r"Bus\s*(\d+)", user_prompt)
        target_bus = int(bus_match.group(1)) if bus_match else 13

        tasks = {
            "tasks": [
                {
                    "id": "t1",
                    "description": "查询目标节点电压",
                    "dependencies": [],
                    "devices": [target_bus],
                    "device_type": "bus",
                    "voltage_level": "MV",
                },
                {
                    "id": "t2",
                    "description": "查询邻近节点状态",
                    "dependencies": [],
                    "devices": [target_bus],
                    "device_type": "bus",
                    "voltage_level": "MV",
                },
                {
                    "id": "t3",
                    "description": "查询相关线路负载",
                    "dependencies": [],
                    "devices": [target_bus],
                    "device_type": "line",
                    "voltage_level": "MV",
                },
                {
                    "id": "t4",
                    "description": "生成调整方案",
                    "dependencies": ["t1", "t2", "t3"],
                    "devices": [target_bus],
                    "device_type": "gen",
                    "voltage_level": "MV",
                },
                {
                    "id": "t5",
                    "description": "仿真验证方案",
                    "dependencies": ["t4"],
                    "devices": [target_bus],
                    "device_type": "gen",
                    "voltage_level": "MV",
                },
                {
                    "id": "t6",
                    "description": "验证全网约束",
                    "dependencies": ["t5"],
                    "devices": [target_bus],
                    "device_type": "bus",
                    "voltage_level": "MV",
                },
            ]
        }
        return json.dumps(tasks, ensure_ascii=False)

    def _orchestration_response(self, user_prompt: str) -> str:
        """
        S2 编排智能体的模拟响应。
        根据任务描述关键词返回对应的设备级指令。
        """
        prompt_lower = user_prompt.lower()

        # 从 prompt 中提取目标母线
        bus_match = re.search(r"设备.*?(\d+)", user_prompt)
        target_bus = int(bus_match.group(1)) if bus_match else 13

        if "查询目标节点电压" in user_prompt or "目标" in user_prompt and "电压" in user_prompt:
            instructions = {
                "instructions": [
                    {
                        "tool": "get_bus_voltage",
                        "params": {"bus_id": target_bus},
                        "description": f"查询 Bus {target_bus} 的电压幅值和相角",
                        "expected_result": "获取电压数据",
                    }
                ]
            }
        elif "邻近" in user_prompt or "邻居" in user_prompt:
            instructions = {
                "instructions": [
                    {
                        "tool": "get_neighbor_buses",
                        "params": {"bus_id": target_bus},
                        "description": f"查询 Bus {target_bus} 的相邻母线",
                        "expected_result": "获取邻近节点列表",
                    }
                ]
            }
        elif "线路" in user_prompt and "负载" in user_prompt:
            instructions = {
                "instructions": [
                    {
                        "tool": "get_line_loading",
                        "params": {"line_id": 0},
                        "description": "查询相关线路的负载率",
                        "expected_result": "获取线路负载数据",
                    }
                ]
            }
        elif "调整方案" in user_prompt or "生成" in user_prompt and "方案" in user_prompt:
            # t4: 基于前置结果生成调整方案
            # Mock 策略：提高最近的发电机电压设定值
            instructions = {
                "instructions": [
                    {
                        "tool": "get_generator_state",
                        "params": {"gen_id": 0},
                        "description": "查询 Gen 0 当前状态",
                        "expected_result": "获取发电机参数",
                    },
                    {
                        "tool": "set_gen_voltage",
                        "params": {"gen_id": 0, "vm_pu": 1.06},
                        "description": "提高 Gen 0 电压设定值以抬升末端电压",
                        "expected_result": "发电机电压设定值更新为 1.06 p.u.",
                    },
                ]
            }
        elif "仿真" in user_prompt:
            instructions = {
                "instructions": [
                    {
                        "tool": "simulate_action",
                        "params": {
                            "action": {
                                "type": "set_gen_voltage",
                                "gen_id": 0,
                                "vm_pu": 1.06,
                            }
                        },
                        "description": "在仿真副本上验证调压方案",
                        "expected_result": "仿真收敛且目标母线电压改善",
                    }
                ]
            }
        elif "约束" in user_prompt or "验证" in user_prompt:
            instructions = {
                "instructions": [
                    {
                        "tool": "check_constraints",
                        "params": {},
                        "description": "校验全网运行约束",
                        "expected_result": "所有母线电压和线路负载率在限值内",
                    }
                ]
            }
        else:
            # 兜底
            instructions = {
                "instructions": [
                    {
                        "tool": "get_bus_voltage",
                        "params": {"bus_id": target_bus},
                        "description": "查询目标母线电压",
                        "expected_result": "获取电压数据",
                    }
                ]
            }

        return json.dumps(instructions, ensure_ascii=False)

    def _execution_response(self, user_prompt: str) -> str:
        """
        S3 执行智能体的模拟响应。
        根据指令中的工具名称生成调用序列。
        """
        # 从编排智能体传来的指令中提取工具名
        tool_match = re.search(r'"tool":\s*"(\w+)"', user_prompt)
        tool_name = tool_match.group(1) if tool_match else "get_bus_voltage"

        # 确定执行策略：查询类 → direct，操作类 → simulate
        if tool_name.startswith("get_") or tool_name == "check_constraints":
            strategy = "direct"
        elif tool_name == "simulate_action":
            strategy = "simulate"
        else:
            strategy = "simulate"

        # 提取 params
        params_match = re.search(r'"params":\s*(\{[^}]*\})', user_prompt)
        if params_match:
            try:
                params = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                params = {}
        else:
            params = {}

        response = {
            "tool_sequence": [
                {"tool": tool_name, "params": params}
            ],
            "strategy": strategy,
        }
        return json.dumps(response, ensure_ascii=False)


class RealLLMClient(LLMClient):
    """
    真实 LLM 客户端（可选）。
    支持 OpenAI 兼容 API（包括 Anthropic 的兼容接口）。
    
    使用方式：
        export LLM_API_KEY="your-api-key"
        export LLM_BASE_URL="https://api.openai.com/v1"   # 或其他兼容端点
        export LLM_MODEL="gpt-4o-mini"                     # 或其他模型
    """

    def __init__(self):
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise ValueError("未设置 LLM_API_KEY 环境变量")
        logger.info(f"使用 RealLLMClient: model={self.model}")

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        })

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload.encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError) as e:
            logger.error(f"LLM API 调用失败: {e}")
            return "{}"


def create_llm_client() -> LLMClient:
    """
    工厂函数：有 API Key 就用真实客户端，没有就用 Mock。
    """
    if os.environ.get("LLM_API_KEY"):
        try:
            return RealLLMClient()
        except ValueError:
            pass
    return MockLLMClient()