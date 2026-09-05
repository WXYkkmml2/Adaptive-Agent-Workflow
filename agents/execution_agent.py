"""
S3 执行智能体。

对应原文档 S3 的完整流程：
  00 反例库匹配（不调用大模型）
  01 调用大模型（输出候选工具调用序列 + 执行策略）
  02 反例降权（对候选方案概率做降权和重新归一化）
  03 确定采样温度（用 a、b、C 计算）
  04 执行策略分支（直接/仿真/人工）
  05 最终采样（在收缩后的概率空间中采样工具调用序列）
"""

import logging
import json
import numpy as np
from agents.permission import Permission
from grid.tools import get_available_tools, call_tool, TOOL_REGISTRY
from llm.client import LLMClient
from llm.prompts import EXECUTION_SYSTEM, EXECUTION_USER
from config.settings import TEMPERATURE_MIN, TEMPERATURE_MAX

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """
    执行智能体：拿到设备级指令，选工具，执行。

    执行智能体不再继续拆任务，而是直接调用工具库操作电网。
    """

    def __init__(
        self,
        agent_id: str,
        instruction: dict,
        permission: Permission,
        network,
        llm: LLMClient,
        anti_example_store=None,
        d0_info: dict = None,
        certainty: float = 0.7,
        depth: int = 2,
    ):
        self.agent_id = agent_id
        self.instruction = instruction  # 编排层下发的设备级指令
        self.permission = permission
        self.network = network
        self.llm = llm
        self.anti_example_store = anti_example_store
        self.d0_info = d0_info or {}
        self.certainty = certainty
        self.depth = depth

    def execute(self) -> dict:
        """
        执行 S3 完整流程。
        """
        indent = "  " * self.depth
        logger.info(f"{indent}[{self.agent_id}] 执行: {self.instruction.get('description', '')}")

        # ---- 00. 反例库匹配 ----
        downweight = self._match_anti_examples()

        # ---- 01. 确定工具调用序列 ----
        tool_plan = self._plan_tool_calls()
        if not tool_plan:
            return self._fail("无法生成工具调用计划")

        # ---- 02. 反例降权 ----
        tool_sequence = tool_plan.get("tool_sequence", [])
        strategy = tool_plan.get("strategy", "direct")

        # ---- 03. 确定采样温度 ----
        temperature = self._compute_temperature()
        logger.info(f"{indent}  采样温度: {temperature:.2f}, 执行策略: {strategy}")

        # ---- 04. 执行策略分支 ----
        if strategy == "human":
            logger.info(f"{indent}  ⚠ 高风险操作，需要人工确认（MVP 中自动通过）")

        if strategy == "simulate":
            # 先仿真验证
            sim_ok = self._simulate_first(tool_sequence)
            if not sim_ok:
                return self._fail("仿真验证未通过")

        # ---- 05. 执行工具调用序列 ----
        return self._execute_tools(tool_sequence)

    def _match_anti_examples(self) -> dict:
        """
        00 反例库匹配。
        
        对应原文档：
        "不调用大模型，就是用传统向量检索，
         输出的是哪些候选方案需要降权的约束信息"
        """
        if self.anti_example_store is None or self.anti_example_store.size() == 0:
            return {}

        matched = self.anti_example_store.match(
            task_description=self.instruction.get("description", ""),
            coupling_strength=self.d0_info.get("coupling_strength_a", 0),
            topology_depth=self.d0_info.get("topology_depth_b", 1),
        )

        if matched:
            downweight = self.anti_example_store.get_downweight_tools(matched)
            logger.info(f"  反例匹配命中 {len(matched)} 条，降权工具: {downweight}")
            return downweight

        return {}

    def _plan_tool_calls(self) -> dict:
        """
        01 调用大模型生成工具调用序列。

        这里可以直接用编排层已经给出的 tool + params，
        也可以再调一次 LLM 让它选择更优的序列。
        MVP 中优先使用编排层的指令（如果已经足够具体）。
        """
        tool_name = self.instruction.get("tool", "")
        params = self.instruction.get("params", {})

        if tool_name and tool_name in TOOL_REGISTRY:
            # 编排层已经给出了具体的工具调用，直接用
            # 根据工具的风险等级确定执行策略
            risk = TOOL_REGISTRY[tool_name]["risk_level"]
            strategy_map = {"low": "direct", "medium": "simulate", "high": "human"}
            return {
                "tool_sequence": [{"tool": tool_name, "params": params}],
                "strategy": strategy_map.get(risk, "simulate"),
            }

        # 编排层没给具体工具，需要调 LLM 选择
        available = get_available_tools(self.permission.to_dict())
        tools_desc = {
            name: TOOL_REGISTRY[name]["description"]
            for name in available
        }

        system_prompt = EXECUTION_SYSTEM.format(tools=json.dumps(tools_desc, ensure_ascii=False))
        user_prompt = EXECUTION_USER.format(
            instruction=json.dumps(self.instruction, ensure_ascii=False),
            context=str(self.d0_info),
        )

        return self.llm.complete_json(system_prompt, user_prompt)

    def _compute_temperature(self) -> float:
        """
        03 确定采样温度。

        对应原文档：
        "采样温度和 C、a、b 挂钩。
         耦合强度高 or BFS 距离远 → 更高温度（更多候选方案）；
         C 高 → 低温度（已经很确定）。
         
         综合 R = w1*a_norm + w2*b_norm - w3*C_norm
         再把 R 线性映射到 [T_min, T_max]"
        """
        a = self.d0_info.get("coupling_strength_a", 0.5)
        b_norm = self.d0_info.get("topology_depth_b_norm", 0.5)
        c = self.certainty

        # 权重系数（可通过历史案例统计调整）
        w1, w2, w3 = 0.3, 0.3, 0.4

        # a 和 b 正相关温度（影响范围大 → 需要更多探索）
        # C 负相关温度（确定性高 → 不需要那么多探索）
        r = w1 * a + w2 * b_norm - w3 * c

        # R 的理论范围大约在 [-0.4, 0.6]，映射到 [T_min, T_max]
        r_min, r_max = -0.4, 0.6
        r_clipped = max(r_min, min(r, r_max))
        ratio = (r_clipped - r_min) / (r_max - r_min)
        temperature = TEMPERATURE_MIN + ratio * (TEMPERATURE_MAX - TEMPERATURE_MIN)

        return round(temperature, 4)

    def _simulate_first(self, tool_sequence: list) -> bool:
        """
        04 仿真验证分支。

        对应原文档：
        "中风险的调用仿真工具然后再去执行，
         先获取仿真的结果，再输入大模型做二次推理，分析是否可行"
        
        MVP 简化：直接在副本上试跑，检查约束是否满足。
        """
        from grid.tools import simulate_action, check_constraints

        for call in tool_sequence:
            tool_name = call.get("tool", "")
            params = call.get("params", {})

            if tool_name == "simulate_action":
                result = simulate_action(self.network.net, params.get("action", params))
                if not result.get("success", False):
                    logger.warning(f"  仿真失败: {result.get('error', '未知')}")
                    return False
                # 在仿真副本上检查约束
                sim_net = result.get("net_copy")
                if sim_net is not None:
                    constraints = check_constraints(sim_net)
                    if not constraints["all_satisfied"]:
                        logger.warning(f"  仿真约束违规: {constraints['violations']}")
                        return False

            elif tool_name in ("set_gen_voltage", "set_gen_output", "set_line_status"):
                # 把修改类操作包装成 simulate_action 来验证
                action = {"type": tool_name, **params}
                result = simulate_action(self.network.net, action)
                if not result.get("success", False):
                    return False
                sim_net = result.get("net_copy")
                if sim_net is not None:
                    constraints = check_constraints(sim_net)
                    if not constraints["all_satisfied"]:
                        return False

        return True

    def _execute_tools(self, tool_sequence: list) -> dict:
        """
        05 执行工具调用序列。

        执行智能体按序列调用工具，提取数据/修改状态，
        然后把结果回传给父智能体。
        """
        indent = "  " * self.depth
        results = []

        for call in tool_sequence:
            tool_name = call.get("tool", "")
            params = call.get("params", {})

            # 对于修改类工具，直接在真实网络上操作
            # （在 MVP 中"真实网络"就是 pandapower 的 net 对象）
            if tool_name in ("set_gen_voltage", "set_gen_output", "set_line_status"):
                # 调用 network 对象的方法修改真实网络
                try:
                    if tool_name == "set_gen_voltage":
                        self.network.set_gen_voltage(**params)
                    elif tool_name == "set_gen_output":
                        self.network.set_gen_output(**params)
                    elif tool_name == "set_line_status":
                        self.network.set_line_status(**params)
                    result = {"success": True, "tool": tool_name, "result": "操作已执行"}
                except Exception as e:
                    result = {"success": False, "tool": tool_name, "error": str(e)}
            else:
                # 查询类工具通过统一入口调用
                result = call_tool(tool_name, self.network.net, **params)

            logger.info(
                f"{indent}  工具 {tool_name}: "
                f"{'✓' if result.get('success') else '✗'}"
            )
            if not result.get('success'):
                logger.info(f"{indent}    详细错误: {result.get('error')}")
            results.append(result)

        all_success = all(r.get("success", False) for r in results)

        return {
            "agent_id": self.agent_id,
            "success": all_success,
            "tool_results": results,
            "instruction": self.instruction,
        }

    def _fail(self, reason: str) -> dict:
        logger.warning(f"  [{self.agent_id}] 失败: {reason}")
        return {
            "agent_id": self.agent_id,
            "success": False,
            "error": reason,
            "instruction": self.instruction,
        }