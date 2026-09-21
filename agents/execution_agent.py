"""
S3 执行智能体。

执行流程：
  01 调用大模型（输出候选工具调用序列 + 执行策略）
  02 确定采样温度（用 a、b、C 计算）
  03 执行策略分支（直接/仿真/人工）
  04 最终执行工具调用序列
"""

import logging
import json
from agents.permission import Permission
from grid.tools import get_available_tools, call_tool, TOOL_REGISTRY, validate_tool_params
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
        d0_info: dict = None,
        certainty: float = 0.7,
        depth: int = 2,
        goal=None,
        kernel=None,
    ):
        self.agent_id = agent_id
        self.instruction = instruction  # 编排层下发的设备级指令
        self.permission = permission
        self.network = network
        self.llm = llm
        self.d0_info = d0_info or {}
        self.certainty = certainty
        self.depth = depth
        self.goal = goal
        self.kernel = kernel

    def execute(self) -> dict:
        """
        执行 S3 完整流程。
        """
        if self.kernel is not None:
            return self.kernel.commit_action(self.instruction, self.permission)
        indent = "  " * self.depth
        logger.info(f"{indent}[{self.agent_id}] 执行: {self.instruction.get('description', '')}")

        tool_plan = self._plan_tool_calls()
        if not tool_plan.get("success", True) is not False:
            return self._fail(tool_plan.get("error", "无法生成工具调用计划"))

        tool_sequence = tool_plan.get("tool_sequence", [])
        if not tool_sequence:
            return self._fail("工具调用序列为空")

        strategy = tool_plan.get("strategy", "direct")

        temperature = self._compute_temperature()
        logger.info(f"{indent}  采样温度: {temperature:.2f}, 执行策略: {strategy}")

        if strategy == "human":
            logger.info(f"{indent}  ⚠ 高风险操作，需要人工确认（MVP 中自动通过）")

        if strategy == "simulate":
            self._simulation_error = "仿真验证未通过"
            sim_ok = self._simulate_first(tool_sequence)
            if not sim_ok:
                return self._fail(self._simulation_error)

        return self._execute_tools(tool_sequence)

    def _plan_tool_calls(self) -> dict:
        """
        使用编排层已经给出的 tool + params，禁止再次调用 LLM。
        若 tool 不存在或不在当前权限范围内，直接返回结构化失败。
        """
        tool_name = self.instruction.get("tool", "")
        params = self.instruction.get("params", {})
        available = get_available_tools(self.permission.to_dict())

        if not tool_name:
            return {"success": False, "error": "指令缺少 tool 字段"}

        if tool_name not in TOOL_REGISTRY:
            call_tool(tool_name, self.network.net, permission=self.permission.to_dict())
            return {"success": False, "error": f"工具 '{tool_name}' 不存在", "available_tools": available}

        if tool_name not in available:
            call_tool(tool_name, self.network.net, permission=self.permission.to_dict())
            return {
                "success": False,
                "error": f"工具 '{tool_name}' 不在当前权限对应的 available_tools 中",
                "available_tools": available,
            }

        ok, err = validate_tool_params(tool_name, params, context=self.instruction.get("description", ""))
        if not ok:
            return {"success": False, "error": f"参数校验失败: {err}", "available_tools": available}

        risk = TOOL_REGISTRY[tool_name]["risk_level"]
        strategy_map = {"low": "direct", "medium": "simulate", "high": "human"}
        return {
            "success": True,
            "tool_sequence": [{"tool": tool_name, "params": params}],
            "strategy": strategy_map.get(risk, "simulate"),
        }

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
        from grid.tools import simulate_action, check_constraints, constraints_not_worse

        for call in tool_sequence:
            tool_name = call.get("tool", "")
            params = call.get("params", {})

            if tool_name == "simulate_action":
                action = params.get("action", params)
                result = simulate_action(self.network.net, action, self.goal)
                if not result.get("success", False):
                    self._simulation_error = f"仿真失败: action={action}, error={result.get('error')}"
                    logger.warning(f"  仿真失败: {result.get('error', '未知')}")
                    return False
                # 在仿真副本上检查约束
                sim_net = result.get("net_copy")
                if sim_net is not None:
                    constraints = check_constraints(sim_net)
                    before = check_constraints(self.network.net)
                    if not constraints_not_worse(before, constraints):
                        self._simulation_error = (f"仿真验证未通过: action={action}, "
                                                  f"before={before['violations']}, "
                                                  f"after={constraints['violations']}")
                        logger.warning(f"  仿真约束违规: {constraints['violations']}")
                        return False

            elif tool_name in ("set_gen_voltage", "set_gen_output", "set_line_status"):
                # 把修改类操作包装成 simulate_action 来验证
                action = {"type": tool_name, **params}
                result = simulate_action(self.network.net, action, self.goal)
                if not result.get("success", False):
                    self._simulation_error = f"仿真失败: action={action}, error={result.get('error')}"
                    return False
                sim_net = result.get("net_copy")
                if sim_net is not None:
                    constraints = check_constraints(sim_net)
                    before = check_constraints(self.network.net)
                    if not constraints_not_worse(before, constraints):
                        self._simulation_error = (f"仿真验证未通过: action={action}, "
                                                  f"before={before['violations']}, "
                                                  f"after={constraints['violations']}")
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
            context = self.instruction.get("description", "")
            if tool_name in ("set_gen_voltage", "set_gen_output", "set_line_status"):
                if self.goal and len(self.network.action_log) >= self.goal.max_real_actions:
                    results.append({"success": False, "tool": tool_name,
                                    "error": "真实调整次数已达上限"})
                    break
                # 修改类操作：先校验参数名，再调用 network 方法修改真实网络
                ok, err = validate_tool_params(tool_name, params, context=context)
                if not ok:
                    result = {"success": False, "tool": tool_name, "error": err}
                else:
                    try:
                        cleaned = params.copy()
                        if tool_name == "set_gen_voltage":
                            feedback = self.network.set_gen_voltage(**cleaned)
                            actual = self.network.get_generator_state(cleaned["gen_id"])["vm_pu"]
                            if abs(actual - cleaned["vm_pu"]) > 1e-3:
                                result = {"success": False, "tool": tool_name, "result": feedback,
                                          "error": f"PARAMETER 偏差: 期望 {cleaned['vm_pu']}，实际 {actual}"}
                                results.append(result)
                                break
                        elif tool_name == "set_gen_output":
                            self.network.set_gen_output(**cleaned)
                        elif tool_name == "set_line_status":
                            self.network.set_line_status(**cleaned)
                        result = {"success": True, "tool": tool_name, "result": "操作已执行"}
                    except Exception as e:
                        result = {"success": False, "tool": tool_name, "error": str(e)}
            else:
                # 查询类工具通过统一入口调用
                result = call_tool(
                    tool_name,
                    self.network.net,
                    context=context,
                    permission=self.permission.to_dict(),
                    **params,
                )
                if tool_name == "simulate_action" and result.get("success") and self.goal:
                    from grid.goal import goal_status
                    from grid.tools import check_constraints, constraints_not_worse
                    sim_net = result.get("result", {}).get("net_copy")
                    if sim_net is not None:
                        result["result"].update(goal_status(sim_net, self.goal))
                        result["result"]["not_worse"] = constraints_not_worse(
                            check_constraints(self.network.net), check_constraints(sim_net))

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
